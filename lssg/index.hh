#include "utils.hh"
#include "visit_list.hh"
#include "space_dist.hh"
#include "memory.hh"
#include "disk.hh"
#include "scope_label.hh"
#include <atomic>

namespace lssg {

class SearchCache
{
public:
  std::vector<float> occlude_factor;

  explicit SearchCache(size_t reserve_size) { occlude_factor.reserve(reserve_size); }
};

template <typename att_t, typename vec_t, typename scope_kernel_t>
class LSSGIndex
{
public:
  LSSGIndex(size_t max_elements, size_t vec_d, size_t M, size_t efc, std::string space_name,
      std::unique_ptr<scope_kernel_t> &scope_kernel)
      : max_elements_(max_elements), vec_d_(vec_d), M_(M), efc_(efc)
  {
    scope_kernel_     = std::move(scope_kernel);
    sizelinks_per_element_ =
        sizeof(label_t) + sizeof(vec_t) * vec_d_ + sizeof(tableint) * (M_ + 1) * (scope_kernel_->top_layer_ + 1);
    sizelinklistsmem_ = max_elements_ * sizelinks_per_element_;
    linklistsmemory_  = (char *)glass::alloc2M(sizelinklistsmem_);
    if (linklistsmemory_ == nullptr) {
      throw std::runtime_error("Not enough memory: LSSGIndex failed to allocate linklist");
    }
    offset_label_     = 0;
    offset_vec_       = offset_label_ + sizeof(label_t);
    offset_linklists_ = offset_vec_ + sizeof(vec_t) * vec_d_;
    if (space_name == "l2") {
      space_ = new lssg::L2Space(vec_d);
    } else if (space_name == "ip") {
      space_ = new lssg::InnerProductSpace(vec_d);
    } else {
      throw std::runtime_error("unsupported space type " + space_name + ", supported: l2, ip");
    }
    fstdistfunc_     = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();

    visited_pool_.Init(max_elements);
  }

  LSSGIndex(std::string index_path, std::string scope_kernel_path, std::string space_name)
  {
    std::ifstream sk_ifs(scope_kernel_path, std::ios::binary);
    if (!sk_ifs.is_open()) {
      throw std::runtime_error("Failed to open scope kernel file: " + scope_kernel_path);
    }
    scope_kernel_ = std::make_unique<scope_kernel_t>();
    scope_kernel_->Deserialize(sk_ifs);
    sk_ifs.close();

    std::ifstream ifs(index_path, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open index file: " + index_path);
    }

    uint64_t magic = 0;
    uint32_t version = 0;
    ReadBinaryPOD(ifs, magic);
    ReadBinaryPOD(ifs, version);
    if (!ifs || magic != config::kIndexMagic || version != config::kFileFormatVersion) {
      throw std::runtime_error("Invalid or unsupported LSSG index file: " + index_path);
    }

    ReadBinaryPOD(ifs, max_elements_);
    ReadBinaryPOD(ifs, vec_d_);
    ReadBinaryPOD(ifs, M_);
    ReadBinaryPOD(ifs, efc_);
    ReadBinaryPOD(ifs, curvec_num_);

    if (!ifs || max_elements_ == 0 || curvec_num_ > max_elements_ || M_ == 0 || vec_d_ == 0) {
      throw std::runtime_error("Invalid LSSG index metadata: " + index_path);
    }
    if (max_elements_ != scope_kernel_->max_elements_) {
      throw std::runtime_error("Index and scope capacities do not match: " + index_path);
    }

    offset_label_     = 0;
    offset_vec_       = offset_label_ + sizeof(label_t);
    offset_linklists_ = offset_vec_ + sizeof(vec_t) * vec_d_;
    sizelinks_per_element_ =
        sizeof(label_t) + sizeof(vec_t) * vec_d_ + sizeof(tableint) * (M_ + 1) * (scope_kernel_->top_layer_ + 1);
    sizelinklistsmem_ = max_elements_ * sizelinks_per_element_;

    linklistsmemory_ = (char *)glass::alloc2M(sizelinklistsmem_);
    if (linklistsmemory_ == nullptr) {
      throw std::runtime_error("Failed to allocate memory for linklistsmemory_");
    }

    memset(linklistsmemory_, 0, sizelinklistsmem_);

    for (tableint id = 0; id < curvec_num_; ++id) {
      label_t *label_ptr = GetLabelByInternalID(id);
      ifs.read(reinterpret_cast<char *>(label_ptr), sizeof(label_t));

      vec_t *vec_ptr = GetVecByInternalID(id);
      ifs.read(reinterpret_cast<char *>(vec_ptr), sizeof(vec_t) * vec_d_);

      for (layer_t layer = 0; layer <= static_cast<layer_t>(scope_kernel_->top_layer_); ++layer) {
        auto     ll = GetLinkListByInternalID(id, layer);
        tableint link_count;

        ReadBinaryPOD(ifs, link_count);

        if (!ifs || link_count > M_) {
          throw std::runtime_error(
              "Invalid link count: " + std::to_string(link_count) + " exceeds M=" + std::to_string(M_));
        }

        ll[M_] = link_count;

        for (tableint i = 0; i < link_count; ++i) {
          ReadBinaryPOD(ifs, ll[i]);
          if (!ifs || ll[i] >= curvec_num_) {
            throw std::runtime_error("Invalid neighbor ID in LSSG index file: " + index_path);
          }
        }
      }
    }

    ifs.close();

    visited_pool_.Init(max_elements_);

    if (space_name == "l2") {
      space_ = new lssg::L2Space(vec_d_);
    } else if (space_name == "ip") {
      space_ = new lssg::InnerProductSpace(vec_d_);
    } else {
      throw std::runtime_error("unsupported space type " + space_name + ", supported: l2, ip");
    }
    fstdistfunc_     = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();
  }

  void save(const std::string &index_path, const std::string &scope_kernel_path)
  {
    std::ofstream sk_ofs(scope_kernel_path, std::ios::binary);
    if (!sk_ofs.is_open()) {
      throw std::runtime_error("Failed to open scope kernel file: " + scope_kernel_path);
    }
    scope_kernel_->Serialize(sk_ofs);
    if (!sk_ofs.good()) {
      throw std::runtime_error("Failed while writing scope kernel file: " + scope_kernel_path);
    }
    sk_ofs.close();

    std::ofstream ofs(index_path, std::ios::binary);
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open index file: " + index_path);
    }

    WriteBinaryPOD(ofs, config::kIndexMagic);
    WriteBinaryPOD(ofs, config::kFileFormatVersion);

    WriteBinaryPOD(ofs, max_elements_);
    WriteBinaryPOD(ofs, vec_d_);
    WriteBinaryPOD(ofs, M_);
    WriteBinaryPOD(ofs, efc_);
    WriteBinaryPOD(ofs, curvec_num_);

    for (tableint id = 0; id < curvec_num_; ++id) {
      label_t *label_ptr = GetLabelByInternalID(id);
      ofs.write(reinterpret_cast<const char *>(label_ptr), sizeof(label_t));

      vec_t *vec_ptr = GetVecByInternalID(id);
      ofs.write(reinterpret_cast<const char *>(vec_ptr), sizeof(vec_t) * vec_d_);

      for (layer_t layer = 0; layer <= static_cast<layer_t>(scope_kernel_->top_layer_); ++layer) {
        auto     ll         = GetLinkListByInternalID(id, layer);
        tableint link_count = ll[M_];

        WriteBinaryPOD(ofs, link_count);

        for (tableint i = 0; i < link_count; ++i) {
          WriteBinaryPOD(ofs, ll[i]);
        }
      }
    }

    if (!ofs.good()) {
      throw std::runtime_error("Failed while writing index file: " + index_path);
    }

    ofs.close();
  }

  void insert(const label_t label, const vec_t *v, const att_t &attribute, bool replace_deleted = false)
  {
    (void)replace_deleted;
    int      max_level_copy = -1;
    tableint cur_num        = INVALID_TABLEINT;
    {
      std::unique_lock<std::mutex> lock(max_layer_lock_);
      cur_num = curvec_num_++;
      {
        if (cur_num == 0) {
          auto label_mem = GetLabelByInternalID(cur_num);
          auto vec_mem   = GetVecByInternalID(cur_num);
          memcpy(label_mem, &label, sizeof(label_t));
          memcpy(vec_mem, v, sizeof(vec_t) * vec_d_);
          for (layer_t layer = 0; layer <= static_cast<layer_t>(scope_kernel_->top_layer_); ++layer) {
            auto ll = GetLinkListByInternalID(cur_num, layer);
            ll[M_]  = 0;
          }
          scope_kernel_->Insert(attribute, cur_num);
          return;
        }
      }
      max_level_copy = scope_kernel_->cur_max_layer_;
    }
    if (max_level_copy < 0 || cur_num == INVALID_TABLEINT) {
      throw std::runtime_error("Index insertion initialization failed");
    }
    std::vector<std::vector<dist_id_pair>> tmp_linklist(max_level_copy + 1);
    std::vector<dist_id_pair>              cur_allc;
    auto                                   curc_record = visited_pool_.Get();
    curc_record->Clear();

    std::vector<std::vector<tableint>> entry_points_per_layer;
    auto query_scopes = scope_kernel_->GetScopeForAllLayers(attribute, entry_points_per_layer);

    for (layer_t layer = max_level_copy; layer >= 0; --layer) {
      const auto &entry_points = entry_points_per_layer[layer];
      auto       &query_scope  = query_scopes[layer];

      for (auto ep_id : entry_points) {
        auto d = fstdistfunc_(v, GetVecByInternalID(ep_id), dist_func_param_);
        metric_dist_comps_++;
        cur_allc.emplace_back(d, ep_id);
      }
      std::vector<dist_id_pair> filtered_curc;
      filtered_curc.reserve(cur_allc.size());
      for (const auto &[d, i] : cur_allc) {
        if (scope_kernel_->TestInScope(i, query_scope)) {
          filtered_curc.emplace_back(d, i);
          curc_record->Set(i);
        }
      }
      cur_allc = std::move(filtered_curc);
      if (cur_allc.size() < M_) {
        using scope_type = std::remove_reference_t<decltype(query_scope)>;
        auto new_c       = SearchCandidates<true>(cur_allc,
            v,
            query_scope,
                  {layer, max_level_copy},
            efc_,
            static_cast<scope_type *>(nullptr),
            static_cast<VisitedList<tableint> *>(nullptr),
            cur_num);
        for (const auto &[d, i] : new_c) {
          if (i == cur_num) {
            throw std::runtime_error("repeated internal id");
          }
          if (!curc_record->Test(i)) {
            cur_allc.emplace_back(d, i);
          }
        }
      }
      auto pruned         = PruneByHeuristic(cur_allc, M_ / 2);
      tmp_linklist[layer] = std::move(pruned);
    }
    visited_pool_.Return(curc_record);

    {
      auto label_mem = GetLabelByInternalID(cur_num);
      auto vec_mem   = GetVecByInternalID(cur_num);
      memcpy(label_mem, &label, sizeof(label_t));
      memcpy(vec_mem, v, sizeof(vec_t) * vec_d_);
      for (layer_t layer = max_level_copy; layer >= 0; --layer) {
        auto ll = GetLinkListByInternalID(cur_num, layer);
        ll[M_]  = (tableint)tmp_linklist[layer].size();
        for (tableint i = 0; i < ll[M_]; ++i) {
          if (tmp_linklist[layer][i].id_ == cur_num) {
            throw std::runtime_error("pruned[i].id_ == cur_num");
          }
          if (ll[i]) {
            throw std::runtime_error("newly added point should have blank link list");
          }
          ll[i] = tmp_linklist[layer][i].id_;
        }
      }
      scope_kernel_->Insert(attribute, cur_num);
    }
    for (layer_t layer = max_level_copy; layer >= 0; --layer) {
      for (const auto &[nn_d, nn_i] : tmp_linklist[layer]) {
        auto nn_ll = GetLinkListByInternalID(nn_i, layer);

        tableint *size_ptr = &nn_ll[M_];
        tableint  old      = __atomic_fetch_add(size_ptr, 1, __ATOMIC_ACQ_REL);
        if (old < (tableint)M_) {
          __atomic_store_n(&nn_ll[old], cur_num, __ATOMIC_RELEASE);
        } else {
          __atomic_fetch_sub(size_ptr, 1, __ATOMIC_ACQ_REL);

          tableint                  nn_ll_sz = __atomic_load_n(size_ptr, __ATOMIC_ACQUIRE);
          std::vector<dist_id_pair> nn_allc;
          nn_allc.reserve(nn_ll_sz + 1);
          for (tableint i = 0; i < nn_ll_sz; ++i) {
            tableint neighbor_id = __atomic_load_n(&nn_ll[i], __ATOMIC_ACQUIRE);
            nn_allc.emplace_back(
                fstdistfunc_(GetVecByInternalID(nn_i), GetVecByInternalID(neighbor_id), dist_func_param_), neighbor_id);
          }

          nn_allc = scope_kernel_->FilterCandidates(nn_allc, nn_i, layer);
          nn_allc.emplace_back(nn_d, cur_num);
          auto nn_pruned = PruneByHeuristic(nn_allc, M_);

          tableint new_sz = (tableint)nn_pruned.size();
          for (tableint i = 0; i < new_sz; ++i) {
            __atomic_store_n(&nn_ll[i], nn_pruned[i].id_, __ATOMIC_RELEASE);
          }
          __atomic_store_n(size_ptr, new_sz, __ATOMIC_RELEASE);
        }
      }
    }
  }

  template <bool collect_tier_stats = false, typename filter_t>
  auto searchKNN(const vec_t *query_vec, size_t efs, size_t k, const filter_t &filter,
      const LabelFilterConfig &filter_config = {}, TierUsageStats *tier_stats = nullptr)
      -> std::vector<std::pair<dist_t, label_t>>
  {
    lssg_range<layer_t>        layer_rng;
    std::vector<dist_id_pair> ep_dist_id_pairs;
    std::vector<tableint>     eps;
    auto                      query_scope = scope_kernel_->TransformFilterToScope(filter, filter_config);
    size_t                    ep_num      = std::max(config::kSearchMinEntryPoints,
        std::min(config::kSearchMaxEntryPoints, efs / 2));
    layer_rng = {0, scope_kernel_->Landing(query_scope, eps, ep_num)};
    ep_dist_id_pairs.reserve(eps.size());

    // Initialize tier-usage stats if requested (compile-time branch)
    if constexpr (collect_tier_stats) {
      if (tier_stats) {
        tier_stats->Reset(layer_rng.u_);
        size_t matching_vertices = scope_kernel_->CountMatchingVertices(query_scope);
        tier_stats->selectivity  = static_cast<double>(matching_vertices) / static_cast<double>(curvec_num_);
      }
    }

    for (auto ep_id : eps) {
      auto d = fstdistfunc_(query_vec, GetVecByInternalID(ep_id), dist_func_param_);
      metric_dist_comps_++;
      ep_dist_id_pairs.emplace_back(d, ep_id);
    }

    auto global_visited = visited_pool_.Get();
    global_visited->Clear();

    // Equality fast path: only one matching labelset, so scope_visited tracking
    // is unnecessary. Skip the allocation and pass nullptr.
    bool                        is_equality = (filter_config.type_ == LabelFilterConfig::LABEL_EQUALITY);
    auto                        scope_visited_storage =
        is_equality ? lssg_bitset<labelset_id_t>(0) : scope_kernel_->GetEmptyScope();
    lssg_bitset<labelset_id_t> *scope_visited_ptr = is_equality ? nullptr : &scope_visited_storage;

    auto result = SearchCandidates<false, collect_tier_stats>(
        ep_dist_id_pairs, query_vec, query_scope, layer_rng, efs, scope_visited_ptr, global_visited, -1, tier_stats);

    visited_pool_.Return(global_visited);

    while (result.size() > k) {
      POP_HEAP(result);
    }
    std::vector<std::pair<dist_t, label_t>> final_res(result.size());
    for (size_t i = 0; i < final_res.size(); ++i) {
      final_res[i].first  = result[i].dist_;
      final_res[i].second = *GetLabelByInternalID(result[i].id_);
    }
    return final_res;
  }

  LSSGIndex(const LSSGIndex &)            = delete;
  LSSGIndex &operator=(const LSSGIndex &) = delete;
  LSSGIndex(LSSGIndex &&)                 = delete;
  LSSGIndex &operator=(LSSGIndex &&)      = delete;
  LSSGIndex()                             = delete;

  ~LSSGIndex()
  {
    free(linklistsmemory_);
    linklistsmemory_ = nullptr;
    if (space_ != nullptr) {
      delete space_;
    }
  }

private:
  inline __attribute__((always_inline)) auto GetLabelByInternalID(tableint internal_id) -> label_t *
  {
    return (label_t *)(linklistsmemory_ + internal_id * sizelinks_per_element_ + offset_label_);
  }

  inline __attribute__((always_inline)) auto GetAttByInternalID(tableint internal_id) -> att_t &
  {
    return attrmemory_[internal_id];
  }

  inline __attribute__((always_inline)) auto GetVecByInternalID(tableint internal_id) -> vec_t *
  {
    return (vec_t *)(linklistsmemory_ + internal_id * sizelinks_per_element_ + offset_vec_);
  }

  inline __attribute__((always_inline)) auto GetLinkListByInternalID(tableint internal_id, layer_t layer) -> tableint *
  {
    // Reverse layer layout improves prefetch behavior during top-down traversal.
    return (tableint *)(linklistsmemory_ + internal_id * sizelinks_per_element_ + offset_linklists_ +
                        (scope_kernel_->top_layer_ - layer) * (M_ + 1) * sizeof(tableint));
  }

  template <bool is_build, bool collect_tier_stats = false, typename filter_t, typename scope_t>
  auto SearchCandidates(std::vector<dist_id_pair> &eps, const vec_t *v, const filter_t &scope,
      const lssg_range<layer_t> &layer_rng, const size_t ef, scope_t *scope_visited = nullptr,
      VisitedList<tableint> *passed_visited = nullptr, tableint ignore = -1,
      TierUsageStats *tier_stats = nullptr) -> std::vector<dist_id_pair>
  {
    if (eps.empty())
      return {};
    auto visited = passed_visited ? passed_visited : visited_pool_.Get();
    if (!passed_visited)
      visited->Clear();
    if (is_build && ignore != INVALID_TABLEINT) {
      visited->Set(ignore);
    }

    // Tier-usage stats: track which layer each vertex was first discovered in
    [[maybe_unused]] std::vector<layer_t> discovery_layer;
    if constexpr (collect_tier_stats) {
      discovery_layer.assign(max_elements_, static_cast<layer_t>(-1));
    }

    std::vector<dist_id_pair> result;
    std::vector<dist_id_pair> candidates;
    for (auto ep : eps) {
      PUSH_HEAP(candidates, -ep.dist_, ep.id_);
      PUSH_HEAP(result, ep.dist_, ep.id_);
      visited->Set(ep.id_);
      if (scope_visited)
        scope_kernel_->MarkInnerIDinScope(*scope_visited, ep.id_);
      // Entry points are discovered at the top search layer
      if constexpr (collect_tier_stats) {
        discovery_layer[ep.id_] = layer_rng.u_;
        tier_stats->visited_per_layer[layer_rng.u_]++;
        tier_stats->total_visited++;
      }
    }
    auto res_max_dist = TOP_HEAP(result).dist_;
    while (!candidates.empty()) {
      auto [dist, id] = TOP_HEAP(candidates);
      if constexpr (is_build) {
        if (((-dist) > res_max_dist) && (result.size() == ef)) {
          break;
        }
      } else {
        if ((-dist) > res_max_dist) {
          break;
        }
      }

#ifdef USE_SSE
      _mm_prefetch((char *)GetLinkListByInternalID(id, layer_rng.u_), _MM_HINT_T2);
#endif
      POP_HEAP(candidates);
      metric_hops_++;

      size_t neighbor_cnt = 0;

      for (layer_t layer = layer_rng.u_; layer >= layer_rng.l_; --layer) {
        if (neighbor_cnt >= M_) {
          break;
        }
        auto     ll    = GetLinkListByInternalID(id, layer);
        tableint ll_sz = __atomic_load_n(&ll[M_], __ATOMIC_ACQUIRE);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited->GetData(ll[0])), _MM_HINT_T0);
        _mm_prefetch((char *)(visited->GetData(ll[0]) + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(GetVecByInternalID(ll[0])), _MM_HINT_T0);
        _mm_prefetch((char *)(ll + 1), _MM_HINT_T0);
#endif
        for (tableint i = 0; i < ll_sz; ++i) {
          if (neighbor_cnt >= M_) {
            break;
          }
          auto nn_id = __atomic_load_n(&ll[i], __ATOMIC_ACQUIRE);
#ifdef USE_SSE
          _mm_prefetch((char *)(visited->GetData(ll[i + 1])), _MM_HINT_T0);
          _mm_prefetch((char *)(GetVecByInternalID(ll[i + 1])), _MM_HINT_T0);
#endif
          // Filter-valid expansion ratio: count every scanned neighbor
          if constexpr (collect_tier_stats) {
            tier_stats->total_scanned_per_layer[layer]++;
          }

          if (!scope_kernel_->TestInScope(nn_id, scope)) {
            continue;
          }

          // Count filter-valid neighbor (passes TestInScope)
          if constexpr (collect_tier_stats) {
            tier_stats->valid_scanned_per_layer[layer]++;
          }
          if (visited->Test(nn_id)) {
            continue;
          }
          visited->Set(nn_id);
          if (scope_visited)
            scope_kernel_->MarkInnerIDinScope(*scope_visited, nn_id);

          // Track tier-usage: record discovery layer and per-layer visited count
          if constexpr (collect_tier_stats) {
            if (discovery_layer[nn_id] == static_cast<layer_t>(-1)) {
              discovery_layer[nn_id] = layer;
              tier_stats->visited_per_layer[layer]++;
              tier_stats->total_visited++;
            }
          }

          auto nn_dist = fstdistfunc_(v, GetVecByInternalID(nn_id), dist_func_param_);
          metric_dist_comps_++;
          neighbor_cnt++;
          if (result.size() < ef || nn_dist < res_max_dist) {
            PUSH_HEAP(candidates, -nn_dist, nn_id);
#ifdef USE_SSE
            _mm_prefetch((char *)linklistsmemory_ + TOP_HEAP(candidates).id_ * sizelinks_per_element_, _MM_HINT_T2);
#endif
            PUSH_HEAP(result, nn_dist, nn_id);
            if (result.size() > ef) {
              POP_HEAP(result);
            }
            res_max_dist = TOP_HEAP(result).dist_;
          }
        }
      }
    }
    if (!passed_visited)
      visited_pool_.Return(visited);

    // Populate per-layer candidate counts from result heap
    if constexpr (collect_tier_stats) {
      for (const auto &[d, id] : result) {
        layer_t dl = discovery_layer[id];
        if (dl >= 0 && dl < static_cast<layer_t>(tier_stats->candidates_per_layer.size())) {
          tier_stats->candidates_per_layer[dl]++;
        }
      }
    }

    return result;
  }

  template <bool use_knn = false>
  auto PruneByHeuristic(std::vector<dist_id_pair> &candidates, const size_t M) -> std::vector<dist_id_pair>
  {
    if (candidates.size() <= M) {
      return candidates;
    }
    if (M == 0) {
      return {};
    }
    if (M == 1) {
      return {candidates[0]};
    }
    std::sort(candidates.begin(), candidates.end());
    std::vector<dist_id_pair> pruned;
    if (use_knn) {
      pruned = candidates;
      pruned.resize(M);
      return pruned;
    }
    for (const auto &[db, ib] : candidates) {
      if (pruned.size() >= M) {
        break;
      }
      bool good = true;
      for (const auto &[da, ia] : pruned) {
        auto curdist = fstdistfunc_(GetVecByInternalID(ib), GetVecByInternalID(ia), dist_func_param_);
        metric_dist_comps_++;
        if (curdist < db) {
          good = false;
          break;
        }
      }
      if (good) {
        pruned.emplace_back(db, ib);
      }
    }
    return pruned;
  }

  auto RobustPrune(std::vector<dist_id_pair> &candidates, const size_t M, tableint id = -1) -> std::vector<dist_id_pair>
  {
    if (candidates.size() <= M) {
      return candidates;
    }
    if (M == 0) {
      return {};
    }
    if (M == 1) {
      return {candidates[0]};
    }

    thread_local SearchCache search_cache(candidates.size());

    std::sort(candidates.begin(), candidates.end());

    std::vector<dist_id_pair> pruned;
    pruned.reserve(M);

    const float alpha = 1.2f;

    const size_t candidate_size = std::min(candidates.size(), 2 * M);

    auto &occlude_factor = search_cache.occlude_factor;
    occlude_factor.clear();
    occlude_factor.resize(candidate_size, 0.0f);

    float cur_alpha = 1.0f;
    while (cur_alpha <= alpha && pruned.size() < M) {
      for (size_t i = 0; i < candidate_size && pruned.size() < M; ++i) {
        if (occlude_factor[i] > cur_alpha)
          continue;

        occlude_factor[i] = std::numeric_limits<float>::max();

        if (candidates[i].id_ != id) {
          pruned.emplace_back(candidates[i].dist_, candidates[i].id_);
        }

        for (size_t j = i + 1; j < candidate_size; ++j) {
          if (occlude_factor[j] > alpha)
            continue;

          auto distance_ij = fstdistfunc_(
              GetVecByInternalID(candidates[i].id_), GetVecByInternalID(candidates[j].id_), dist_func_param_);
          metric_dist_comps_++;

          occlude_factor[j] = (distance_ij == 0) ? std::numeric_limits<float>::max()
                                                 : std::max(occlude_factor[j], candidates[j].dist_ / distance_ij);
        }
      }
      cur_alpha *= 1.2f;
    }

    return pruned;
  }

public:
  std::atomic<size_t> metric_dist_comps_{0};
  std::atomic<size_t> metric_hops_{0};

  // Get the maximum HNSW layer (top layer) for tier-usage reporting
  auto GetTopLayer() const -> layer_t { return scope_kernel_->top_layer_; }
  auto GetDimension() const -> size_t { return vec_d_; }
  auto GetCurrentCount() const -> size_t { return curvec_num_; }
  auto GetMaxElements() const -> size_t { return max_elements_; }

private:
  size_t max_elements_{0};
  size_t vec_d_;
  size_t o_{4};
  size_t M_{24};
  size_t efc_{256};

  size_t curvec_num_{0};
  size_t cur_max_layer_{0};

  size_t sizelinks_per_element_{0};
  size_t sizelinklistsmem_{0};

  size_t offset_label_{0};
  size_t offset_vec_{0};
  size_t offset_linklists_{0};

  char              *linklistsmemory_{nullptr};
  std::vector<att_t> attrmemory_;

  std::mutex max_layer_lock_;
  lssg::SpaceInterface<vec_t> *space_{nullptr};
  lssg::DISTFUNC<vec_t>        fstdistfunc_{nullptr};
  void                          *dist_func_param_{nullptr};

  std::unique_ptr<scope_kernel_t> scope_kernel_{nullptr};

  VisitedPool<VisitedList<tableint>> visited_pool_;
};
}  // namespace lssg
