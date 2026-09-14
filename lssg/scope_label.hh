#pragma once
#include "config.hh"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cassert>
#include <random>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <queue>
#include "scope_kernel.hh"
#include "visit_list.hh"
#include "lruk_cache.hh"
#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace lssg {

const lssg_range<double> jd_range        = {0, 1.0};

enum class ThresholdDivision {
  UNIFORM,
  LOGARITHMIC,
  EXPONENTIAL,
  QUADRATIC
};

class LabelSetScopeKernel : public ScopeKernel<labelset_t, lssg_bitset<labelset_id_t>>
{

public:
  LabelSetScopeKernel() : jaccard_distance_cache_(config::kJaccardCacheCapacity, config::kJaccardCacheShards,
                          "JaccardDistanceCache"),
                          threshold_division_(ThresholdDivision::UNIFORM) {}

  LabelSetScopeKernel(size_t max_layers, size_t max_elements, std::string base_label_file = "",
                      ThresholdDivision division = ThresholdDivision::UNIFORM, size_t preload_limit = 0)
      : ScopeKernel<labelset_t, lssg_bitset<labelset_id_t>>(max_layers, max_elements),
        base_label_file_(std::move(base_label_file)),
        jaccard_distance_cache_(config::kJaccardCacheCapacity, config::kJaccardCacheShards, "JaccardDistanceCache"),
        preload_limit_(preload_limit),
        threshold_division_(division)
  {
    vertex_labelsetid_lookup_ = new labelset_id_t[max_elements];
    memset(vertex_labelsetid_lookup_, -1, sizeof(labelset_id_t) * max_elements);

    vertex_locks_ = new std::atomic<uint32_t>[max_elements];
    for (size_t i = 0; i < max_elements; i++) {
      vertex_locks_[i].store(0, std::memory_order_relaxed);
    }
  }

  void Init() override
  {
    init_jaccard_cache(jd_range);
    this->cur_max_layer_ = this->top_layer_;

    for (size_t layer = 0; layer <= this->top_layer_; ++layer) {
      std::cerr << "Layer " << layer << " Jaccard distance threshold: " << jaccard_cache_[layer] << std::endl;
    }

#ifdef USE_MINHASH
    initialize_minhash();
#endif

    if (!base_label_file_.empty()) {
      std::unique_lock<std::shared_mutex> write_lock(data_mutex_);
      try {
        size_t total_loaded = load_labelsets_from_file(base_label_file_);
        if (preload_limit_ != 0 && total_loaded != preload_limit_) {
          throw std::runtime_error("Base label file contains fewer records than the vector input");
        }
        if (!all_labelsets_.empty()) {

          labelset_entries_.clear();
          labelset_entries_.resize(all_labelsets_.size());

          const size_t num_labelsets = all_labelsets_.size();

          std::cerr << "Computing label frequencies for " << num_labelsets << " labelsets..." << std::endl;
          std::unordered_map<labelsetmember_t, size_t> label_freq_map;
          for (const auto &labelset : all_labelsets_) {
            for (labelsetmember_t label : labelset) {
              ++label_freq_map[label];
            }
          }

          size_t num_unique_labels = label_freq_map.size();
          std::cerr << "Found " << num_unique_labels << " unique labels" << std::endl;

          labelsetmember_t max_label = 0;
          for (const auto &[label, freq] : label_freq_map) {
            max_label = std::max(max_label, label);
          }

          label_frequencies_.clear();
          label_frequencies_.resize(max_label + 1, 0);
          for (const auto &[label, freq] : label_freq_map) {
            label_frequencies_[label] = freq;
          }

          const size_t bytes_per_bitset = (num_labelsets + 63) / 64 * 8;
          const size_t max_dense_labels = bytes_per_bitset == 0
              ? 0
              : config::kDenseIndexMemoryBudgetBytes / bytes_per_bitset;

          std::vector<std::pair<size_t, labelsetmember_t>> sorted_labels;
          sorted_labels.reserve(label_freq_map.size());
          for (const auto &[label, freq] : label_freq_map) {
            sorted_labels.emplace_back(freq, label);
          }
          std::sort(sorted_labels.rbegin(), sorted_labels.rend());

          dense_labels_.clear();
          dense_labels_.reserve(sorted_labels.size());
          std::vector<labelsetmember_t> tier1_labels;
          tier1_labels.reserve(std::min(sorted_labels.size(), max_dense_labels));

          size_t tier1_count    = 0;
          size_t tier1_coverage = 0;
          for (const auto &[freq, label] : sorted_labels) {
            if (tier1_count < max_dense_labels) {
              tier1_labels.push_back(label);
              dense_labels_.insert(label);
              tier1_coverage += freq;
              ++tier1_count;
            } else {
              break;
            }
          }

          dense_threshold_ = tier1_labels.empty() ? 0 : sorted_labels[tier1_count - 1].first;

          size_t total_pairs = num_labelsets * config::kLabelsetEntryCapacity;
          std::cerr << "Two-tier index configuration:" << std::endl;
          std::cerr << "  Tier 1 (dense): " << tier1_labels.size() << " labels (freq >= " << dense_threshold_ << ")"
                    << std::endl;
          std::cerr << "  Tier 2 (sparse): " << (num_unique_labels - tier1_labels.size()) << " labels" << std::endl;
          std::cerr << "  Tier 1 coverage: " << tier1_coverage << " / ~" << total_pairs << " label-labelset pairs"
                    << std::endl;
          std::cerr << "  Tier 1 memory: " << (tier1_labels.size() * bytes_per_bitset) / (1024.0 * 1024.0 * 1024.0)
                    << " GB" << std::endl;

          std::cerr << "Building Tier 1 dense bitsets..." << std::endl;

          label_to_labelsets_.clear();
          label_to_labelsets_.resize(max_label + 1);

// Initialize dense-label bitsets.
#pragma omp parallel for schedule(static)
          for (size_t i = 0; i < tier1_labels.size(); ++i) {
            labelsetmember_t label = tier1_labels[i];
            label_to_labelsets_[label].Resize(num_labelsets);
            label_to_labelsets_[label].Clear();
          }

// Fill dense-label bitsets serially: Set() updates packed words and concurrent
// writes to different bits of the same word would race.
          for (labelset_id_t lid = 0; lid < static_cast<labelset_id_t>(all_labelsets_.size()); ++lid) {
            for (labelsetmember_t label : all_labelsets_[lid]) {
              if (is_dense_label(label)) {
                label_to_labelsets_[label].Set(lid);
              }
            }
          }

          std::cerr << "Building Tier 2 sparse index..." << std::endl;
          sparse_label_index_.clear();
          sparse_label_index_.reserve(num_unique_labels > tier1_labels.size() ? num_unique_labels - tier1_labels.size() : 0);
          for (const auto &[label, freq] : label_freq_map) {
            if (!is_dense_label(label)) {
              auto &vec = sparse_label_index_[label];
              vec.reserve(freq);
            }
          }

          for (labelset_id_t lid = 0; lid < static_cast<labelset_id_t>(all_labelsets_.size()); ++lid) {
            for (labelsetmember_t label : all_labelsets_[lid]) {
              if (!is_dense_label(label)) {
                sparse_label_index_[label].push_back(lid);
              }
            }
          }

          std::cerr << "Two-tier index built successfully!" << std::endl;
          std::cerr << "  Sparse index size: " << sparse_label_index_.size() << " labels, "
                    << sparse_label_index_.size() * sizeof(std::vector<labelset_id_t>) / 1024.0 << " KB overhead"
                    << std::endl;

          for (size_t i = 0; i < all_labelsets_.size(); i++) {
            labelset_entries_[i].reserve(config::kLabelsetEntryCapacity);
          }

#ifdef USE_MINHASH
          build_minhash_index();
#endif

          std::cerr << "Preloaded " << total_loaded << " base labelsets (" << all_labelsets_.size() << " unique) from '"
                    << base_label_file_ << "'\n";

          // Initialize labelset vertex counts for selectivity computation
          labelset_vertex_counts_.resize(all_labelsets_.size(), 0);
        }
      } catch (const std::exception &e) {
        std::cerr << "Failed to preload label sets from '" << base_label_file_ << "': " << e.what() << std::endl;
        throw;
      }
    }
  }

  ~LabelSetScopeKernel() override
  {
    delete[] vertex_labelsetid_lookup_;
    delete[] vertex_locks_;
  }

  void Insert(const labelset_t &labelset, tableint inid) override
  {
    labelset_id_t labelset_id  = -1;
    bool          new_labelset = false;

    {
      std::shared_lock<std::shared_mutex> read_lock(data_mutex_);
      auto                                it = labelset_id_map_.find(labelset);
      if (it == labelset_id_map_.end()) {
        read_lock.unlock();
        std::unique_lock<std::shared_mutex> write_lock(data_mutex_);

        it = labelset_id_map_.find(labelset);
        if (it == labelset_id_map_.end()) {
          assert(labelset_entries_.size() == all_labelsets_.size());
          labelset_id = all_labelsets_.size();
          all_labelsets_.emplace_back(labelset);
          labelset_entries_.resize(all_labelsets_.size());
          labelset_id_map_[labelset] = labelset_id;
          labelset_vertex_counts_.resize(all_labelsets_.size(), 0);

          const size_t new_size = all_labelsets_.size();
          for (size_t label = 0; label < label_to_labelsets_.size(); ++label) {
            auto &bitset = label_to_labelsets_[label];
            // Only dense labels own bitsets.  Resizing every sparse label here
            // would turn the sparse index into a full bitmap index after the
            // first dynamic insertion.
            if (bitset.n_ != 0 && bitset.n_ < new_size) {
              lssg_bitset<labelset_id_t> new_bitset(new_size);
              new_bitset.Clear();

              size_t old_n_uint64 = (bitset.n_ + 63) / 64;
              memcpy(new_bitset.data_, bitset.data_, old_n_uint64 * sizeof(uint64_t));
              new_bitset.num_ones_ = bitset.num_ones_;

              bitset = std::move(new_bitset);
            }
          }

          for (labelsetmember_t label : labelset) {
            if (label >= label_to_labelsets_.size()) {
              label_to_labelsets_.resize(static_cast<size_t>(label) + 1);
            }

            if (is_dense_label(label)) {
              auto &bitset = label_to_labelsets_[label];
              if (bitset.n_ == 0) {
                bitset.Resize(new_size);
                bitset.Clear();
              }
              bitset.Set(labelset_id);
            } else {
              sparse_label_index_[label].push_back(labelset_id);
            }

            if (label >= label_frequencies_.size()) {
              label_frequencies_.resize(static_cast<size_t>(label) + 1, 0);
            }
            ++label_frequencies_[label];
          }

          if (minhash_initialized_) {
            if (minhash_signatures_.size() <= labelset_id) {
              minhash_signatures_.resize(labelset_id + 1);
            }

            minhash_signatures_[labelset_id] = compute_minhash_signature(labelset);

            for (size_t band_idx = 0; band_idx < num_bands_; ++band_idx) {
              uint64_t bucket_id = get_lsh_bucket(minhash_signatures_[labelset_id], band_idx);
              lsh_buckets_[band_idx][bucket_id].push_back(labelset_id);
            }
          }

          new_labelset = true;
        } else {
          labelset_id = it->second;
        }
      } else {
        labelset_id = it->second;
      }
    }

    assert(labelset_id != -1);
    spinLock(inid);
    vertex_labelsetid_lookup_[inid] = labelset_id;
    spinUnlock(inid);

    // Keep vector-backed labelset metadata stable while updating the per-labelset
    // counters and entry points. A concurrent insertion may grow these vectors
    // while another insertion is finishing its graph update.
    std::shared_lock<std::shared_mutex> metadata_lock(data_mutex_);

    // Update per-labelset vertex count for selectivity computation
    if (labelset_id < labelset_vertex_counts_.size()) {
      __atomic_fetch_add(&labelset_vertex_counts_[labelset_id], (size_t)1, __ATOMIC_RELAXED);
    }

    const size_t max_entry_num = config::kLabelsetEntryCapacity;
    std::unique_lock<std::mutex> entry_lock(entry_mutexes_[labelset_id % NUM_ENTRY_MUTEXES]);
    if (labelset_entries_[labelset_id].size() < max_entry_num) {
      labelset_entries_[labelset_id].emplace_back(inid);
    } else {
      thread_local std::mt19937 local_gen{std::random_device{}()};
      if (new_labelset || local_gen() % 10 == 0) {
        labelset_entries_[labelset_id][inid % max_entry_num] = inid;
      }
    }
  }

  auto GetScopeForAllLayers(const labelset_t &query_labelset,
      std::vector<std::vector<tableint>>     &entry_points_per_layer) -> std::vector<lssg_bitset<labelset_id_t>>
  {
    std::shared_lock<std::shared_mutex> read_lock(data_mutex_);

    const size_t                           num_layers = this->cur_max_layer_ + 1;
    std::vector<lssg_bitset<labelset_id_t>> scopes(num_layers);
    entry_points_per_layer.resize(num_layers);

    for (size_t i = 0; i < num_layers; ++i) {
      // Use the fixed vertex capacity instead of the current labelset count.
      // Labelset IDs can grow while another insertion is traversing the graph;
      // the larger scope prevents a newly published vertex from indexing past
      // the snapshot returned by this method.
      scopes[i].Resize(this->max_elements_);
    }

    scopes[this->top_layer_].Fill();
    scopes[0].Clear();

    auto exact_it = labelset_id_map_.find(query_labelset);
    const bool has_exact_match = exact_it != labelset_id_map_.end();
    const labelset_id_t exact_match_id = has_exact_match ? exact_it->second : INVALID_TABLEINT;

    if (has_exact_match) {
      scopes[0].Set(exact_match_id);
    }

    const size_t MAX_ENTRY_NUM = config::kQueryEntryPointCapacity;

    if (has_exact_match && !labelset_entries_[exact_match_id].empty()) {
      for (size_t i = 0; i < std::min(MAX_ENTRY_NUM, labelset_entries_[exact_match_id].size()); ++i) {
        entry_points_per_layer[0].emplace_back(labelset_entries_[exact_match_id][i]);
      }
    }
    for (size_t i = 0; i < std::min(MAX_ENTRY_NUM, all_labelsets_.size()); ++i) {
      if (!labelset_entries_[i].empty()) {
        entry_points_per_layer[this->top_layer_].emplace_back(labelset_entries_[i][0]);
      }
    }

    std::vector<std::pair<labelset_id_t, double>> computed_candidates;
    computed_candidates.reserve(4096);
    bool cache_hit = false;

    cache_hit = has_exact_match && jaccard_distance_cache_.get(exact_match_id, computed_candidates);

    if (!cache_hit) {
      computed_candidates.clear();

      double top_threshold  = jaccard_cache_[this->top_layer_ - 1];
      double min_sim_needed = 1.0 - top_threshold - config::kSimilarLabelsetMargin;

      size_t query_size = query_labelset.size();
      double min_size   = (query_size > 0) ? query_size * min_sim_needed : 0;
      double max_size   = (min_sim_needed > 0) ? query_size / min_sim_needed : std::numeric_limits<double>::max();

    #ifdef USE_MINHASH

      lssg_bitset<labelset_id_t> candidate_set(all_labelsets_.size());
      candidate_set.Clear();

      std::vector<uint8_t> query_signature;
      if (has_exact_match) {
        query_signature = minhash_signatures_[exact_match_id];
      } else {
        query_signature = compute_minhash_signature(query_labelset);
      }

      for (size_t band_idx = 0; band_idx < num_bands_; ++band_idx) {
        uint64_t    bucket_id = get_lsh_bucket(query_signature, band_idx);
        const auto &bucket    = lsh_buckets_[band_idx][bucket_id];

        for (auto cand : bucket) {
          if (!has_exact_match || cand != exact_match_id) {
            candidate_set.Set(cand);
          }
        }
      }

      size_t n_uint64 = (candidate_set.n_ + 63) / 64;
      for (size_t word_idx = 0; word_idx < n_uint64; ++word_idx) {
        uint64_t word = candidate_set.data_[word_idx];
        if (word == 0)
          continue;

        while (word != 0) {
          int           bit_pos = __builtin_ctzll(word);
          labelset_id_t cand    = word_idx * 64 + bit_pos;

          word &= word - 1;

          size_t base_size = all_labelsets_[cand].size();
          if (base_size < min_size || base_size > max_size) {
            continue;
          }

          double jd = compute_jaccard_distance_sorted(query_labelset, cand);
          computed_candidates.emplace_back(cand, jd);
        }
      }
#else
      if (!query_labelset.empty()) {
        constexpr size_t CANDIDATE_BUDGET = config::kIvfCandidateBudget;

        std::vector<std::pair<size_t, labelsetmember_t>> label_freq;
        label_freq.reserve(query_labelset.size());

        for (labelsetmember_t label : query_labelset) {
          size_t freq = get_label_frequency(label);
          label_freq.emplace_back(freq, label);
        }

        std::sort(label_freq.begin(), label_freq.end());

        lssg_bitset<labelset_id_t> cand_labelset(all_labelsets_.size());
        cand_labelset.Clear();

        size_t current_candidates = 0;
        for (const auto &[freq, label] : label_freq) {
          size_t estimated_new_total = current_candidates + freq * 0.8;

          if (current_candidates > 0 && estimated_new_total > CANDIDATE_BUDGET) {
            break;
          }

          if (current_candidates == 0) {
            if (is_dense_label(label)) {
              cand_labelset = label_to_labelsets_[label];
            } else {
              auto it = sparse_label_index_.find(label);
              if (it != sparse_label_index_.end()) {
                cand_labelset.Clear();
                for (labelset_id_t lid : it->second) {
                  cand_labelset.Set(lid);
                }
              }
            }
            current_candidates = freq;
          } else {
            if (is_dense_label(label)) {
              cand_labelset.bitunion(label_to_labelsets_[label]);
            } else {
              auto it = sparse_label_index_.find(label);
              if (it != sparse_label_index_.end()) {
                for (labelset_id_t lid : it->second) {
                  cand_labelset.Set(lid);
                }
              }
            }
            current_candidates = cand_labelset.num_ones_;

            if (current_candidates > CANDIDATE_BUDGET) {
              break;
            }
          }
        }

        if (current_candidates > 0) {
          size_t n_uint64 = (cand_labelset.n_ + 63) / 64;

          for (size_t word_idx = 0; word_idx < n_uint64; ++word_idx) {
            uint64_t word = cand_labelset.data_[word_idx];
            if (word == 0)
              continue;

            while (word != 0) {
              int           bit_pos = __builtin_ctzll(word);
              labelset_id_t lid     = word_idx * 64 + bit_pos;

              word &= word - 1;

              if (has_exact_match && lid == exact_match_id) {
                continue;
              }

              size_t base_size = all_labelsets_[lid].size();
              if (all_labelsets_[lid].empty() || base_size < min_size || base_size > max_size ||
                  all_labelsets_[lid].front() > query_labelset.back()) {
                continue;
              }

              double jd = compute_jaccard_distance_sorted(query_labelset, lid);
              computed_candidates.emplace_back(lid, jd);
            }
          }
        }
      }
#endif

      if (has_exact_match) {
        jaccard_distance_cache_.put(exact_match_id, computed_candidates);
      }
    }

    for (layer_t layer = this->top_layer_ - 1; layer >= 1; --layer) {
      scopes[layer].Clear();
    }

    for (const auto &[lid, jd] : computed_candidates) {
      for (layer_t layer = this->top_layer_ - 1; layer >= 1; --layer) {
        auto jaccard_threshold = jaccard_cache_[layer];
        if (jd <= jaccard_threshold) {
          scopes[layer].Set(lid);
          if (entry_points_per_layer[layer].size() < MAX_ENTRY_NUM && !labelset_entries_[lid].empty()) {
            entry_points_per_layer[layer].emplace_back(labelset_entries_[lid][0]);
          }
        } else {
          break;
        }
      }
    }

    return scopes;
  }

  auto Landing(lssg_bitset<labelset_id_t> &scope, std::vector<tableint> &entry_points,
      size_t ep_num = config::kDefaultLandingEntryPoints)
      -> layer_t override
  {
    if (scope.num_ones_ == 1) {
      entry_points.clear();
      auto &entries = labelset_entries_[scope.last_1_];
      size_t limit = std::min(entries.size(), ep_num);
      for (size_t i = 0; i < limit; ++i) {
        entry_points.emplace_back(entries[i]);
      }
      return 0;
    }

    GetRandomEntries(scope, entry_points, ep_num);
    return this->cur_max_layer_;
  }

  auto TryRaiseLayer() -> bool override { return false; }

  inline auto TestInScope(tableint inid, const lssg_bitset<labelset_id_t> &scope) -> bool override
  {
    auto target_set_id = vertex_labelsetid_lookup_[inid];
    return scope.Test(target_set_id);
  }

  // Count the number of vertices matching a given scope (for selectivity computation)
  auto CountMatchingVertices(const lssg_bitset<labelset_id_t> &scope) -> size_t
  {
    std::shared_lock<std::shared_mutex> read_lock(data_mutex_);
    size_t count = 0;
    if (labelset_vertex_counts_.empty()) {
      // Fallback: count matching labelsets (approximation)
      return scope.num_ones_;
    }
    // Iterate set bits in scope, sum per-labelset vertex counts
    size_t n_uint64 = (scope.n_ + 63) / 64;
    for (size_t word_idx = 0; word_idx < n_uint64; ++word_idx) {
      uint64_t word = scope.data_[word_idx];
      if (word == 0) continue;
      while (word != 0) {
        int bit_pos = __builtin_ctzll(word);
        word &= word - 1;
        labelset_id_t lid = word_idx * 64 + bit_pos;
        if (lid < labelset_vertex_counts_.size()) {
          count += labelset_vertex_counts_[lid];
        }
      }
    }
    return count;
  }

  auto FilterCandidates(const std::vector<dist_id_pair> &candidates, tableint center_inid, layer_t layer)
      -> std::vector<dist_id_pair> override
  {
    std::shared_lock<std::shared_mutex> read_lock(data_mutex_);
    if (layer == 0)
      return candidates;
    std::vector<dist_id_pair> pruned_candidates;
    auto                      center_set_id = vertex_labelsetid_lookup_[center_inid];
    for (const auto &candidate : candidates) {
      bool  good = true;
      auto &set1 = all_labelsets_[vertex_labelsetid_lookup_[candidate.id_]];
      if (vertex_labelsetid_lookup_[candidate.id_] == center_set_id) {
        continue;
      }
      for (const auto selected_candidate : pruned_candidates) {
        auto jd = compute_jaccard_distance_sorted(set1, vertex_labelsetid_lookup_[selected_candidate.id_]);
        if (jd < jaccard_cache_[layer]) {
          good = false;
          break;
        }
      }
      if (good) {
        pruned_candidates.push_back(candidate);
      }
    }
    return pruned_candidates;
  }

  auto TransformFilterToScope(const labelset_t &query_labelset, const FilterConfig &filter_config)
      -> lssg_bitset<labelset_id_t>
  {
    std::shared_lock<std::shared_mutex> read_lock(data_mutex_);

    auto                      label_config = dynamic_cast<const LabelFilterConfig &>(filter_config);
    lssg_bitset<labelset_id_t> scope(this->max_elements_);

    if (query_labelset.empty()) {
      scope.Clear();
      if (label_config.type_ == LabelFilterConfig::LABEL_CONTAINMENT) {
        scope.Fill();
      } else if (label_config.type_ == LabelFilterConfig::LABEL_OVERLAP) {
      } else if (label_config.type_ == LabelFilterConfig::LABEL_EQUALITY) {
        const size_t all_sets_size = all_labelsets_.size();
        for (size_t i = 0; i < all_sets_size; ++i) {
          if (all_labelsets_[i].empty()) {
            scope.Set(i);
          }
        }
      }
      return scope;
    }

    // Query labelsets are already sorted by LoadLabelSets; use directly
    scope.Clear();
    if (label_config.type_ == LabelFilterConfig::LABEL_EQUALITY) {
      handle_equality_query(query_labelset, scope);
    } else if (label_config.type_ == LabelFilterConfig::LABEL_CONTAINMENT) {
      handle_containment_query(query_labelset, scope);
      if (scope.num_ones_ == 1) {
        scope.last_1_ = scope.FindFirst1();
      }
    } else if (label_config.type_ == LabelFilterConfig::LABEL_OVERLAP) {
      handle_overlap_query(query_labelset, scope);
      if (scope.num_ones_ == 1) {
        scope.last_1_ = scope.FindFirst1();
      }
    } else {
      throw std::runtime_error("unknown label filter type");
    }

    return scope;
  }

  inline auto GetEmptyScope() -> lssg_bitset<labelset_id_t> override
  {
    lssg_bitset<labelset_id_t> scope(this->max_elements_);
    scope.Clear();
    return scope;
  }

  inline void MarkInnerIDinScope(lssg_bitset<labelset_id_t> &scope, tableint inid) override
  {
    auto target_set_id = vertex_labelsetid_lookup_[inid];
    scope.Set(target_set_id);
  }

  void Serialize(std::ostream &os) override
  {
    std::unique_lock<std::shared_mutex> write_lock(data_mutex_);

    WriteBinaryPOD(os, config::kScopeMagic);
    WriteBinaryPOD(os, config::kFileFormatVersion);
    const uint8_t use_minhash =
#ifdef USE_MINHASH
        1;
#else
        0;
#endif
    WriteBinaryPOD(os, use_minhash);
#ifdef USE_MINHASH
    WriteBinaryPOD(os, num_minhash_funcs_);
    WriteBinaryPOD(os, num_bands_);
    WriteBinaryPOD(os, MINHASH_BITS_PER_HASH);
#else
    constexpr size_t no_minhash_config = 0;
    WriteBinaryPOD(os, no_minhash_config);
    WriteBinaryPOD(os, no_minhash_config);
    WriteBinaryPOD(os, no_minhash_config);
#endif

    WriteBinaryPOD(os, this->cur_max_layer_);
    WriteBinaryPOD(os, this->top_layer_);
    WriteBinaryPOD(os, this->max_elements_);

    os.write((char *)vertex_labelsetid_lookup_, sizeof(labelset_id_t) * this->max_elements_);

    size_t labelset_all_size = 0;
    WriteBinaryPOD(os, all_labelsets_.size());
    for (const auto &ls : all_labelsets_) {
      WriteBinaryPOD(os, ls.size());
      if (ls.size()) {
        os.write((char *)ls.data(), sizeof(labelsetmember_t) * ls.size());
        labelset_all_size += ls.size() * sizeof(labelsetmember_t);
      }
    }
    std::cerr << "Serialized " << all_labelsets_.size() << " labelsets, total size: " << labelset_all_size / (1024.0)
              << " KB\n";
#ifdef USE_MINHASH
    WriteBinaryPOD(os, minhash_initialized_);
    if (minhash_initialized_) {
      WriteBinaryPOD(os, minhash_params_.size());
      if (!minhash_params_.empty()) {
        os.write((char *)minhash_params_.data(), sizeof(MinHashParams) * minhash_params_.size());
      }

      WriteBinaryPOD(os, minhash_signatures_.size());
      size_t total_sig_bytes = 0;
      for (const auto &sig : minhash_signatures_) {
        WriteBinaryPOD(os, sig.size());
        if (!sig.empty()) {
          os.write((char *)sig.data(), sizeof(uint8_t) * sig.size());
          total_sig_bytes += sig.size();
        }
      }

      size_t bytes_per_sig = minhash_signatures_.empty() ? 0 : total_sig_bytes / minhash_signatures_.size();
      std::cerr << "MinHash signatures: " << minhash_signatures_.size() << " labelsets × " << bytes_per_sig
                << " bytes = " << total_sig_bytes / 1024.0 << " KB total\n";
      std::cerr << "  Configuration: " << num_minhash_funcs_ << " hash functions, " << MINHASH_BITS_PER_HASH
                << " bits/hash, " << num_bands_ << " bands\n";

      WriteBinaryPOD(os, lsh_buckets_.size());
      for (const auto &band : lsh_buckets_) {
        WriteBinaryPOD(os, band.size());
        for (const auto &[bucket_id, labelset_ids] : band) {
          WriteBinaryPOD(os, bucket_id);
          WriteBinaryPOD(os, labelset_ids.size());
          if (!labelset_ids.empty()) {
            os.write((char *)labelset_ids.data(), sizeof(labelset_id_t) * labelset_ids.size());
          }
        }
      }
    }
#endif
  }

  void Deserialize(std::istream &is) override
  {
    std::unique_lock<std::shared_mutex> write_lock(data_mutex_);

    uint64_t magic = 0;
    uint32_t version = 0;
    uint8_t stored_use_minhash = 0;
    size_t stored_num_minhash_funcs = 0;
    size_t stored_num_bands = 0;
    size_t stored_bits_per_hash = 0;
    ReadBinaryPOD(is, magic);
    ReadBinaryPOD(is, version);
    ReadBinaryPOD(is, stored_use_minhash);
    ReadBinaryPOD(is, stored_num_minhash_funcs);
    ReadBinaryPOD(is, stored_num_bands);
    ReadBinaryPOD(is, stored_bits_per_hash);
    if (!is || magic != config::kScopeMagic || version != config::kFileFormatVersion) {
      throw std::runtime_error("Invalid or unsupported LSSG scope file");
    }
#ifdef USE_MINHASH
    if (stored_use_minhash == 0 || stored_num_minhash_funcs != num_minhash_funcs_ ||
        stored_num_bands != num_bands_ || stored_bits_per_hash != MINHASH_BITS_PER_HASH) {
      throw std::runtime_error("Scope file MinHash configuration does not match this build");
    }
#else
    if (stored_use_minhash != 0) {
      throw std::runtime_error("Scope file was built with MinHash; rebuild this target with LSSG_USE_MINHASH=1");
    }
#endif

    labelset_id_map_.clear();
    all_labelsets_.clear();
    labelset_entries_.clear();
    label_to_labelsets_.clear();

    ReadBinaryPOD(is, this->cur_max_layer_);
    ReadBinaryPOD(is, this->top_layer_);
    ReadBinaryPOD(is, this->max_elements_);
    if (!is || this->top_layer_ == 0 || this->cur_max_layer_ > this->top_layer_ || this->max_elements_ == 0) {
      throw std::runtime_error("Invalid LSSG scope metadata");
    }
    if (vertex_labelsetid_lookup_) {
      delete[] vertex_labelsetid_lookup_;
    }

    if (vertex_locks_) {
      delete[] vertex_locks_;
    }

    vertex_labelsetid_lookup_ = new labelset_id_t[this->max_elements_];
    is.read((char *)vertex_labelsetid_lookup_, sizeof(labelset_id_t) * this->max_elements_);
    if (!is) {
      throw std::runtime_error("Truncated vertex label mapping in LSSG scope file");
    }

    vertex_locks_ = new std::atomic<uint32_t>[this->max_elements_];
    for (size_t i = 0; i < this->max_elements_; i++) {
      vertex_locks_[i].store(0, std::memory_order_relaxed);
    }

    size_t labelset_count = 0;
    ReadBinaryPOD(is, labelset_count);
    if (!is || labelset_count == 0 || labelset_count > this->max_elements_) {
      throw std::runtime_error("Invalid labelset count in LSSG scope file");
    }
    all_labelsets_.resize(labelset_count);
    for (uint64_t i = 0; i < labelset_count; ++i) {
      size_t ls_size = 0;
      ReadBinaryPOD(is, ls_size);
      if (ls_size) {
        all_labelsets_[i].resize(ls_size);
        is.read((char *)all_labelsets_[i].data(), sizeof(labelsetmember_t) * ls_size);
      }
    }
    if (!is) {
      throw std::runtime_error("Truncated labelset data in LSSG scope file");
    }

    labelset_id_map_.reserve(all_labelsets_.size());
    for (size_t i = 0; i < all_labelsets_.size(); ++i) {
      labelset_id_map_[all_labelsets_[i]] = static_cast<labelset_id_t>(i);
    }

    const size_t max_entry_num = config::kLabelsetEntryCapacity;
    labelset_entries_.resize(all_labelsets_.size());
    labelset_vertex_counts_.resize(all_labelsets_.size(), 0);
    for (tableint vid = 0; vid < static_cast<tableint>(this->max_elements_); ++vid) {
      auto lid = vertex_labelsetid_lookup_[vid];
      if (static_cast<size_t>(lid) < labelset_entries_.size()) {
        auto &vec = labelset_entries_[lid];
        if (vec.size() < max_entry_num) {
          vec.emplace_back(vid);
        } else {
          vec[vid % max_entry_num] = vid;
        }
        // Count per-labelset vertex frequency for selectivity computation
        ++labelset_vertex_counts_[lid];
      }
    }

    const size_t num_labelsets = all_labelsets_.size();

    std::cerr << "Rebuilding two-tier inverted index for " << num_labelsets << " labelsets..." << std::endl;

    std::unordered_map<labelsetmember_t, size_t> label_freq_map;
    for (const auto &labelset : all_labelsets_) {
      for (labelsetmember_t label : labelset) {
        ++label_freq_map[label];
      }
    }

    size_t num_unique_labels = label_freq_map.size();
    std::cerr << "Found " << num_unique_labels << " unique labels" << std::endl;

    labelsetmember_t max_label = 0;
    for (const auto &[label, freq] : label_freq_map) {
      max_label = std::max(max_label, label);
    }

    label_frequencies_.clear();
    label_frequencies_.resize(max_label + 1, 0);
    for (const auto &[label, freq] : label_freq_map) {
      label_frequencies_[label] = freq;
    }

    const size_t bytes_per_bitset = (num_labelsets + 63) / 64 * 8;
    const size_t max_dense_labels = bytes_per_bitset == 0
        ? 0
        : config::kDenseIndexMemoryBudgetBytes / bytes_per_bitset;

    std::vector<std::pair<size_t, labelsetmember_t>> sorted_labels;
    sorted_labels.reserve(label_freq_map.size());
    for (const auto &[label, freq] : label_freq_map) {
      sorted_labels.emplace_back(freq, label);
    }
    std::sort(sorted_labels.rbegin(), sorted_labels.rend());

    dense_labels_.clear();
    dense_labels_.reserve(sorted_labels.size());
    std::vector<labelsetmember_t> tier1_labels;
    tier1_labels.reserve(std::min(sorted_labels.size(), max_dense_labels));

    size_t tier1_count    = 0;
    size_t tier1_coverage = 0;
    for (const auto &[freq, label] : sorted_labels) {
      if (tier1_count < max_dense_labels) {
        tier1_labels.push_back(label);
        dense_labels_.insert(label);
        tier1_coverage += freq;
        ++tier1_count;
      } else {
        break;
      }
    }

    dense_threshold_ = tier1_labels.empty() ? 0 : sorted_labels[tier1_count - 1].first;

    std::cerr << "Two-tier index configuration:" << std::endl;
    std::cerr << "  Tier 1 (dense): " << tier1_labels.size() << " labels (freq >= " << dense_threshold_ << ")"
              << std::endl;
    std::cerr << "  Tier 2 (sparse): " << (num_unique_labels - tier1_labels.size()) << " labels" << std::endl;
    std::cerr << "  Tier 1 memory: " << (tier1_labels.size() * bytes_per_bitset) / (1024.0 * 1024.0 * 1024.0) << " GB"
              << std::endl;

    std::cerr << "Building Tier 1 dense bitsets..." << std::endl;

    label_to_labelsets_.clear();
    label_to_labelsets_.resize(max_label + 1);

// Initialize dense-label bitsets.
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tier1_labels.size(); ++i) {
      labelsetmember_t label = tier1_labels[i];
      label_to_labelsets_[label].Resize(num_labelsets);
      label_to_labelsets_[label].Clear();
    }

// Fill dense-label bitsets serially for the same packed-word race reason as in
// the build path above.
    for (labelset_id_t lid = 0; lid < static_cast<labelset_id_t>(all_labelsets_.size()); ++lid) {
      for (labelsetmember_t label : all_labelsets_[lid]) {
        if (is_dense_label(label)) {
          label_to_labelsets_[label].Set(lid);
        }
      }
    }

    std::cerr << "Building Tier 2 sparse index..." << std::endl;
    sparse_label_index_.clear();
    sparse_label_index_.reserve(num_unique_labels > tier1_labels.size() ? num_unique_labels - tier1_labels.size() : 0);
    for (const auto &[label, freq] : label_freq_map) {
      if (!is_dense_label(label)) {
        auto &vec = sparse_label_index_[label];
        vec.reserve(freq);
      }
    }

    for (labelset_id_t lid = 0; lid < static_cast<labelset_id_t>(all_labelsets_.size()); ++lid) {
      for (labelsetmember_t label : all_labelsets_[lid]) {
        if (!is_dense_label(label)) {
          sparse_label_index_[label].push_back(lid);
        }
      }
    }

    std::cerr << "Two-tier index rebuilt successfully!" << std::endl;
    if (this->cur_max_layer_ > this->top_layer_) {
      this->cur_max_layer_ = this->top_layer_;
    }

    init_jaccard_cache(jd_range);
#ifdef USE_MINHASH
    ReadBinaryPOD(is, minhash_initialized_);
    if (minhash_initialized_) {
      size_t params_size = 0;
      ReadBinaryPOD(is, params_size);
      minhash_params_.resize(params_size);
      if (params_size > 0) {
        is.read((char *)minhash_params_.data(), sizeof(MinHashParams) * params_size);
      }

      size_t signatures_size = 0;
      ReadBinaryPOD(is, signatures_size);
      minhash_signatures_.resize(signatures_size);
      for (size_t i = 0; i < signatures_size; ++i) {
        size_t sig_size = 0;
        ReadBinaryPOD(is, sig_size);
        minhash_signatures_[i].resize(sig_size);
        if (sig_size > 0) {
          is.read((char *)minhash_signatures_[i].data(), sizeof(uint8_t) * sig_size);
        }
      }

      size_t num_bands = 0;
      ReadBinaryPOD(is, num_bands);
      lsh_buckets_.resize(num_bands);
      for (size_t band_idx = 0; band_idx < num_bands; ++band_idx) {
        size_t num_buckets = 0;
        ReadBinaryPOD(is, num_buckets);
        for (size_t bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx) {
          uint64_t bucket_id = 0;
          ReadBinaryPOD(is, bucket_id);

          size_t labelset_ids_size = 0;
          ReadBinaryPOD(is, labelset_ids_size);

          std::vector<labelset_id_t> labelset_ids(labelset_ids_size);
          if (labelset_ids_size > 0) {
            is.read((char *)labelset_ids.data(), sizeof(labelset_id_t) * labelset_ids_size);
          }

          lsh_buckets_[band_idx][bucket_id] = std::move(labelset_ids);
        }
      }
    } else {
      build_minhash_index();
    }
#endif
  }

private:
  static inline void ltrim(std::string &s) { s.erase(0, s.find_first_not_of(" \t\n\r\f\v")); }
  static inline void rtrim(std::string &s)
  {
    auto pos = s.find_last_not_of(" \t\n\r\f\v");
    if (pos == std::string::npos)
      s.clear();
    else
      s.erase(pos + 1);
  }
  static inline void trim(std::string &s)
  {
    ltrim(s);
    rtrim(s);
  }

  inline bool is_dense_label(labelsetmember_t label) const { return dense_labels_.find(label) != dense_labels_.end(); }

  inline size_t get_label_frequency(labelsetmember_t label) const
  {
    if (label < label_frequencies_.size()) {
      return label_frequencies_[label];
    }
    return 0;
  }

  // Load label sets from a text file (one line per set, comma-separated integers).
  size_t load_labelsets_from_file(const std::string &filename)
  {
    std::ifstream in(filename);
    if (!in.is_open()) {
      throw std::runtime_error("Cannot open base label file: " + filename);
    }

    all_labelsets_.clear();
    labelset_id_map_.clear();

    size_t      total_count = 0;
    std::string line;

    while (std::getline(in, line)) {
      trim(line);
      if (line.empty())
        continue;

      labelset_t        labels;
      std::stringstream ss(line);
      std::string       token;
      while (std::getline(ss, token, ',')) {
        trim(token);
        if (token.empty())
          continue;
        try {
          size_t parsed = 0;
          const unsigned long long value = std::stoull(token, &parsed);
          if (parsed != token.size() || token.front() == '-' ||
              value > std::numeric_limits<labelsetmember_t>::max()) {
            throw std::invalid_argument("invalid label token");
          }
          labels.emplace_back(static_cast<labelsetmember_t>(value));
        } catch (...) {
          throw std::runtime_error("Invalid token '" + token + "' in base label file");
        }
      }

      if (!labels.empty()) {
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

        total_count++;

        if (labelset_id_map_.find(labels) == labelset_id_map_.end()) {
          labelset_id_t new_id     = static_cast<labelset_id_t>(all_labelsets_.size());
          labelset_id_map_[labels] = new_id;
          all_labelsets_.emplace_back(labels);
        }

        if (preload_limit_ != 0 && total_count >= preload_limit_) {
          break;
        }
      }
    }

    return total_count;
  }

  void GetRandomEntries(const lssg_bitset<labelset_id_t> &scope, std::vector<tableint> &eps, size_t ep_num)
  {
    eps.clear();

    if (ep_num == 0) {
      return;
    }

    const size_t n_uint64 = (scope.n_ + 63) / 64;
    std::vector<labelset_id_t> active_lids;
    active_lids.reserve(std::min(scope.num_ones_, ep_num));

    // Collect active labelset IDs first; this enables a second round to add
    // more representatives from each labelset when scope is very selective.
    for (size_t word_idx = 0; word_idx < n_uint64; ++word_idx) {
      uint64_t word = scope.data_[word_idx];
      if (word == 0) {
        continue;
      }

      while (word != 0) {
        int           bit_pos = __builtin_ctzll(word);
        labelset_id_t lid     = static_cast<labelset_id_t>(word_idx * 64 + bit_pos);
        word &= (word - 1);

        if (lid < labelset_entries_.size() && !labelset_entries_[lid].empty()) {
          active_lids.emplace_back(lid);
        }
      }
    }

    if (active_lids.empty()) {
      return;
    }

    // Stage 1 (diversity first): ensure one seed per labelset when possible.
    for (labelset_id_t lid : active_lids) {
      if (eps.size() >= ep_num) {
        break;
      }
      const auto &entries = labelset_entries_[lid];
      eps.emplace_back(entries[0]);
    }

    // Stage 2 (uniqueness second): fill with remaining unique filtered entries.
    if (eps.size() < ep_num) {
      std::unordered_set<tableint> selected;
      selected.reserve(ep_num * 2);
      for (auto id : eps) {
        selected.insert(id);
      }

      // 2a) Consume remaining cached entries from each active labelset.
      for (size_t offset = 1; eps.size() < ep_num; ++offset) {
        bool added_any = false;
        for (labelset_id_t lid : active_lids) {
          if (eps.size() >= ep_num) {
            break;
          }
          const auto &entries = labelset_entries_[lid];
          if (offset >= entries.size()) {
            continue;
          }

          tableint candidate = entries[offset];
          if (selected.insert(candidate).second) {
            eps.emplace_back(candidate);
            added_any = true;
          }
        }
        if (!added_any) {
          break;
        }
      }

      // 2b) If still short, scan vertices of active labelsets for more entries.
      if (eps.size() < ep_num) {
        for (labelset_id_t lid : active_lids) {
          if (eps.size() >= ep_num) break;
          const auto &entries = labelset_entries_[lid];
          for (tableint inid : entries) {
            if (eps.size() >= ep_num) break;
            if (selected.insert(inid).second) {
              eps.emplace_back(inid);
            }
          }
        }
      }
    }
  }

  void handle_equality_query(const labelset_t &query_labelset, lssg_bitset<labelset_id_t> &scope)
  {
    auto it = labelset_id_map_.find(query_labelset);
    if (it != labelset_id_map_.end()) {
      scope.Set(it->second);
    }
  }

  void handle_containment_query(const labelset_t &query_sorted, lssg_bitset<labelset_id_t> &scope)
  {
    if (query_sorted.size() == 1) {
      labelsetmember_t label = query_sorted[0];

      if (is_dense_label(label)) {
        const auto &label_bitset = label_to_labelsets_[label];
        if (label_bitset.n_ > 0) {
          scope = label_bitset;
        }
      } else {
        auto it = sparse_label_index_.find(label);
        if (it != sparse_label_index_.end()) {
          scope.Clear();
          for (labelset_id_t lid : it->second) {
            scope.Set(lid);
          }
        } else {
          scope.Clear();
        }
      }
      return;
    }

    if (query_sorted.empty()) {
      scope.Clear();
      return;
    }

    labelsetmember_t first_label = query_sorted[0];
    if (is_dense_label(first_label)) {
      const auto &first_bitset = label_to_labelsets_[first_label];
      if (first_bitset.n_ == 0) {
        scope.Clear();
        return;
      }
      scope = first_bitset;
    } else {
      auto it = sparse_label_index_.find(first_label);
      if (it == sparse_label_index_.end() || it->second.empty()) {
        scope.Clear();
        return;
      }
      scope.Clear();
      for (labelset_id_t lid : it->second) {
        scope.Set(lid);
      }
    }

    // AND with remaining labels' bitsets (tier-aware)
    lssg_bitset<labelset_id_t> sparse_intersection(scope.n_);
    for (size_t i = 1; i < query_sorted.size(); ++i) {
      labelsetmember_t label = query_sorted[i];

      if (is_dense_label(label)) {
        const auto &label_bitset = label_to_labelsets_[label];
        if (label_bitset.n_ == 0) {
          scope.Clear();
          return;
        }
        scope.bitintersect(label_bitset);
      } else {
        // Sparse label: intersect by scanning posting list and probing scope.
        auto it = sparse_label_index_.find(label);
        if (it == sparse_label_index_.end() || it->second.empty()) {
          scope.Clear();
          return;
        }

        sparse_intersection.Clear();
        for (labelset_id_t lid : it->second) {
          if (scope.Test(lid)) {
            sparse_intersection.Set(lid);
          }
        }
        // Swap instead of copy-assign: O(1) pointer swap vs O(N) alloc+memcpy
        std::swap(scope, sparse_intersection);
      }

      if (scope.num_ones_ == 0) {
        return;
      }
    }
  }

  void handle_overlap_query(const labelset_t &query_sorted, lssg_bitset<labelset_id_t> &scope)
  {
    if (query_sorted.size() == 1) {
      handle_containment_query(query_sorted, scope);
      return;
    }

    if (query_sorted.empty()) {
      scope.Clear();
      return;
    }

    labelsetmember_t first_label = query_sorted[0];
    if (is_dense_label(first_label)) {
      const auto &first_bitset = label_to_labelsets_[first_label];
      if (first_bitset.n_ == 0) {
        scope.Clear();
      } else {
        scope = first_bitset;
      }
    } else {
      auto it = sparse_label_index_.find(first_label);
      if (it == sparse_label_index_.end() || it->second.empty()) {
        scope.Clear();
      } else {
        scope.Clear();
        for (labelset_id_t lid : it->second) {
          scope.Set(lid);
        }
      }
    }

    for (size_t i = 1; i < query_sorted.size(); ++i) {
      labelsetmember_t label = query_sorted[i];

      if (is_dense_label(label)) {
        const auto &label_bitset = label_to_labelsets_[label];
        if (label_bitset.n_ > 0) {
          scope.bitunion(label_bitset);
        }
      } else {
        auto it = sparse_label_index_.find(label);
        if (it != sparse_label_index_.end()) {
          for (labelset_id_t lid : it->second) {
            scope.Set(lid);
          }
        }
      }
    }
  }

  void init_jaccard_cache(lssg_range<double> jd_range = lssg_range<double>{0.0, 1.0})
  {
    jaccard_cache_.assign(this->top_layer_ + 1, 0.0);
    if (!jaccard_cache_.empty()) {
      jaccard_cache_[0]                = jd_range.l_;
      jaccard_cache_[this->top_layer_] = jd_range.u_;

      const double range = jd_range.u_ - jd_range.l_;
      const size_t n = this->top_layer_;

      switch (threshold_division_) {
        case ThresholdDivision::UNIFORM:
          {
            const double delta = range / n;
            for (size_t i = 1; i < n; ++i) {
              jaccard_cache_[i] = jd_range.l_ + i * delta;
            }
          }
          break;

        case ThresholdDivision::LOGARITHMIC:
          {
            const double log_base = std::log(n + 1);
            for (size_t i = 1; i < n; ++i) {
              jaccard_cache_[i] = jd_range.l_ + range * (std::log(i + 1) / log_base);
            }
          }
          break;

        case ThresholdDivision::EXPONENTIAL:
          {
            const double exp_factor = std::exp(1.0) - 1.0;
            for (size_t i = 1; i < n; ++i) {
              double normalized = static_cast<double>(i) / n;
              jaccard_cache_[i] = jd_range.l_ + range * ((std::exp(normalized) - 1.0) / exp_factor);
            }
          }
          break;

        case ThresholdDivision::QUADRATIC:
          {
            for (size_t i = 1; i < n; ++i) {
              double normalized = static_cast<double>(i) / n;
              jaccard_cache_[i] = jd_range.l_ + range * (normalized * normalized);
            }
          }
          break;
      }
    }
  }

  void initialize_minhash()
  {
    constexpr size_t total_bits    = num_minhash_funcs_ * MINHASH_BITS_PER_HASH;
    constexpr size_t bytes_per_sig = (total_bits + 7) / 8;
    constexpr size_t bits_per_band = rows_per_band_ * MINHASH_BITS_PER_HASH;

    std::cerr << "\n========== MinHash Configuration ==========\n";
    std::cerr << "Hash functions: " << num_minhash_funcs_ << "\n";
    std::cerr << "Bits per hash:  " << MINHASH_BITS_PER_HASH << " bits\n";
    std::cerr << "Signature size: " << bytes_per_sig << " bytes per labelset\n";
    std::cerr << "LSH bands:      " << num_bands_ << " bands × " << rows_per_band_ << " hashes/band\n";
    std::cerr << "Bucket space:   2^" << bits_per_band << " = " << (1ULL << std::min(size_t(bits_per_band), size_t(20)))
              << (bits_per_band > 20 ? " (approx)" : "") << " buckets per band\n";

    double sig_memory_mb = (600000.0 * bytes_per_sig) / (1024.0 * 1024.0);
    std::cerr << "Est. memory:    " << sig_memory_mb << " MB for 600K labelsets\n";
    std::cerr << "===========================================\n\n";
    std::random_device                      rd;
    std::mt19937_64                         gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1, MINHASH_PRIME - 1);

    minhash_params_.resize(num_minhash_funcs_);
    for (size_t i = 0; i < num_minhash_funcs_; ++i) {
      minhash_params_[i].a = dist(gen);
      minhash_params_[i].b = dist(gen);
    }

    lsh_buckets_.resize(num_bands_);

    minhash_initialized_ = true;
  }

  void build_minhash_index()
  {
    if (!minhash_initialized_) {
      initialize_minhash();
    }

    minhash_signatures_.resize(all_labelsets_.size());

#pragma omp parallel for schedule(dynamic, omp_get_max_threads())
    for (size_t i = 0; i < all_labelsets_.size(); ++i) {
      minhash_signatures_[i] = compute_minhash_signature(all_labelsets_[i]);
    }

#pragma omp parallel for schedule(dynamic, omp_get_max_threads())
    for (size_t band_idx = 0; band_idx < num_bands_; ++band_idx) {
      for (size_t i = 0; i < all_labelsets_.size(); ++i) {
        uint64_t bucket_id = get_lsh_bucket(minhash_signatures_[i], band_idx);
        lsh_buckets_[band_idx][bucket_id].push_back(i);
      }
    }
  }

  std::vector<uint8_t> compute_minhash_signature(const labelset_t &labelset) const
  {
    constexpr size_t  bits_per_func  = MINHASH_BITS_PER_HASH;
    constexpr size_t  total_bits     = num_minhash_funcs_ * bits_per_func;
    constexpr size_t  num_bytes      = (total_bits + 7) / 8;
    constexpr size_t  funcs_per_byte = 8 / bits_per_func;
    constexpr uint8_t bit_mask       = (1 << bits_per_func) - 1;

    std::vector<uint8_t> signature(num_bytes, 0);

    if (labelset.empty()) {
      std::fill(signature.begin(), signature.end(), 0xFF);
      return signature;
    }

    size_t hash_idx = 0;
    for (size_t byte_idx = 0; byte_idx < num_bytes; ++byte_idx) {
      uint8_t byte_val = 0;

      size_t funcs_in_this_byte = std::min(funcs_per_byte, num_minhash_funcs_ - hash_idx);
      for (size_t func_pos = 0; func_pos < funcs_in_this_byte; ++func_pos, ++hash_idx) {
        uint32_t min_hash = std::numeric_limits<uint32_t>::max();

        for (labelsetmember_t label : labelset) {
          uint64_t hash = (static_cast<uint64_t>(minhash_params_[hash_idx].a) * label + minhash_params_[hash_idx].b) %
                          MINHASH_PRIME;
          min_hash = std::min(min_hash, static_cast<uint32_t>(hash));
        }

        uint8_t extracted_bits = min_hash & bit_mask;
        byte_val |= (extracted_bits << (func_pos * bits_per_func));
      }

      signature[byte_idx] = byte_val;
    }

    return signature;
  }

  // Returns bucket id for one LSH band.
  uint64_t get_lsh_bucket(const std::vector<uint8_t> &signature, size_t band_idx) const
  {
    constexpr size_t bits_per_func = MINHASH_BITS_PER_HASH;
    size_t           start_bit     = band_idx * rows_per_band_ * bits_per_func;
    size_t           end_bit       = start_bit + rows_per_band_ * bits_per_func;

    uint64_t hash = 0;

    size_t start_byte        = start_bit / 8;
    size_t start_bit_in_byte = start_bit % 8;
    size_t end_byte          = end_bit / 8;
    size_t end_bit_in_byte   = end_bit % 8;

    if (start_bit_in_byte == 0 && end_bit_in_byte == 0) {
      for (size_t byte_idx = start_byte; byte_idx < end_byte; ++byte_idx) {
        hash = (hash << 8) | signature[byte_idx];
      }
    } else {
      for (size_t bit_idx = start_bit; bit_idx < end_bit; ++bit_idx) {
        size_t byte_idx = bit_idx >> 3;
        size_t bit_pos  = bit_idx & 7;

        uint8_t bit_val = (signature[byte_idx] >> bit_pos) & 1;
        hash            = (hash << 1) | bit_val;
      }
    }

    return hash;
  }

  double compute_jaccard_distance_sorted(const labelset_t &query_labelset, tableint base_labelset_id) const
  {
    if (query_labelset.empty()) {
      if (base_labelset_id >= static_cast<tableint>(all_labelsets_.size()))
        return 1.0;
      return all_labelsets_[base_labelset_id].empty() ? 0.0 : 1.0;
    }

    const auto &base_labelset = all_labelsets_[base_labelset_id];

    if (base_labelset.empty())
      return 1.0;

    size_t intersection_size = 0;
    size_t i = 0, j = 0;

    while (i < query_labelset.size() && j < base_labelset.size()) {
      if (query_labelset[i] == base_labelset[j]) {
        intersection_size++;
        i++;
        j++;
      } else if (query_labelset[i] < base_labelset[j]) {
        i++;
      } else {
        j++;
      }
    }

    size_t union_size = query_labelset.size() + base_labelset.size() - intersection_size;
    return 1.0 - static_cast<double>(intersection_size) / union_size;
  }

  double compute_dice_distance_sorted(const labelset_t &query_labelset, tableint base_labelset_id) const
  {
    if (query_labelset.empty()) {
      if (base_labelset_id >= static_cast<tableint>(all_labelsets_.size()))
        return 1.0;
      return all_labelsets_[base_labelset_id].empty() ? 0.0 : 1.0;
    }

    const auto &base_labelset = all_labelsets_[base_labelset_id];

    if (base_labelset.empty())
      return 1.0;

    size_t intersection_size = 0;
    size_t i = 0, j = 0;

    while (i < query_labelset.size() && j < base_labelset.size()) {
      if (query_labelset[i] == base_labelset[j]) {
        intersection_size++;
        i++;
        j++;
      } else if (query_labelset[i] < base_labelset[j]) {
        i++;
      } else {
        j++;
      }
    }

    size_t total_size = query_labelset.size() + base_labelset.size();
    if (total_size == 0)
      return 0.0;
    double dice_similarity = 2.0 * intersection_size / total_size;
    return 1.0 - dice_similarity;
  }

  double compute_overlap_distance_sorted(const labelset_t &query_labelset, tableint base_labelset_id) const
  {
    if (query_labelset.empty()) {
      if (base_labelset_id >= static_cast<tableint>(all_labelsets_.size()))
        return 1.0;
      return all_labelsets_[base_labelset_id].empty() ? 0.0 : 1.0;
    }

    const auto &base_labelset = all_labelsets_[base_labelset_id];

    if (base_labelset.empty())
      return 1.0;

    size_t intersection_size = 0;
    size_t i = 0, j = 0;

    while (i < query_labelset.size() && j < base_labelset.size()) {
      if (query_labelset[i] == base_labelset[j]) {
        intersection_size++;
        i++;
        j++;
      } else if (query_labelset[i] < base_labelset[j]) {
        i++;
      } else {
        j++;
      }
    }

    size_t min_size = std::min(query_labelset.size(), base_labelset.size());
    if (min_size == 0)
      return 0.0;
    double overlap_similarity = static_cast<double>(intersection_size) / min_size;
    return 1.0 - overlap_similarity;
  }

  double compute_cosine_distance_sorted(const labelset_t &query_labelset, tableint base_labelset_id) const
  {
    if (query_labelset.empty() || base_labelset_id >= static_cast<tableint>(all_labelsets_.size())) {
      return 1.0;
    }

    const auto &base_labelset = all_labelsets_[base_labelset_id];

    if (base_labelset.empty())
      return 1.0;

    size_t intersection_size = 0;
    size_t i = 0, j = 0;

    while (i < query_labelset.size() && j < base_labelset.size()) {
      if (query_labelset[i] == base_labelset[j]) {
        intersection_size++;
        i++;
        j++;
      } else if (query_labelset[i] < base_labelset[j]) {
        i++;
      } else {
        j++;
      }
    }

    double magnitude_product = std::sqrt(static_cast<double>(query_labelset.size()) * base_labelset.size());
    if (magnitude_product == 0.0)
      return 0.0;
    double cosine_similarity = intersection_size / magnitude_product;
    return 1.0 - cosine_similarity;
  }

private:
  inline void spinLock(tableint id)
  {
    uint32_t expected = 0;
    while (
        !vertex_locks_[id].compare_exchange_weak(expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
      expected = 0;
      std::this_thread::yield();
    }
  }

  inline void spinUnlock(tableint id) { vertex_locks_[id].store(0, std::memory_order_release); }

  static constexpr size_t NUM_ENTRY_MUTEXES = config::kEntryMutexCount;

  labelset_id_t         *vertex_labelsetid_lookup_{nullptr};
  std::atomic<uint32_t> *vertex_locks_{nullptr};

  std::vector<dist_t> jaccard_cache_;

  std::shared_mutex data_mutex_;

  mutable std::mutex entry_mutexes_[NUM_ENTRY_MUTEXES];

  std::unordered_map<labelset_t, labelset_id_t> labelset_id_map_;
  std::vector<labelset_t>                       all_labelsets_;

  std::vector<std::vector<tableint>> labelset_entries_;

  // Per-labelset vertex counts for selectivity computation
  std::vector<size_t> labelset_vertex_counts_;

  std::vector<lssg_bitset<labelset_id_t>> label_to_labelsets_;
  std::unordered_map<labelsetmember_t, std::vector<labelset_id_t>> sparse_label_index_;
  std::unordered_set<labelsetmember_t> dense_labels_;
  std::vector<size_t>                  label_frequencies_;
  size_t                               dense_threshold_ = config::kDenseLabelFrequencyFallback;

  std::string base_label_file_;

  static constexpr size_t   num_minhash_funcs_ = config::kNumMinhashFunctions;
  static constexpr size_t   num_bands_         = config::kNumBands;
  static constexpr size_t   rows_per_band_     = num_minhash_funcs_ / num_bands_;
  static constexpr uint64_t MINHASH_PRIME      = config::kMinhashPrime;

  static constexpr size_t MINHASH_BITS_PER_HASH = config::kMinhashBitsPerHash;
  struct MinHashParams
  {
    uint32_t a;
    uint32_t b;
  };

  bool                       minhash_initialized_{false};
  std::vector<MinHashParams> minhash_params_;

  std::vector<std::vector<uint8_t>> minhash_signatures_;

  std::vector<std::unordered_map<uint64_t, std::vector<labelset_id_t>>> lsh_buckets_;

  mutable LRUCache<labelset_id_t, std::vector<std::pair<labelset_id_t, double>>> jaccard_distance_cache_;

  size_t preload_limit_{0};
  ThresholdDivision threshold_division_;
};
}  // namespace lssg
