#pragma once

#include "../lssg/space_dist.hh"
#include "../lssg/utils.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace benchmark {

inline float *fvecs_read(const std::string &filename, size_t &dimension, size_t &count)
{
  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open vector file: " + filename);
  }

  std::int32_t dimension_on_disk = 0;
  input.read(reinterpret_cast<char *>(&dimension_on_disk), sizeof(dimension_on_disk));
  if (!input || dimension_on_disk <= 0) {
    throw std::runtime_error("Invalid vector dimension in: " + filename);
  }

  const size_t disk_dimension = static_cast<size_t>(dimension_on_disk);
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0) {
    throw std::runtime_error("Cannot determine vector file size: " + filename);
  }
  const size_t record_size = sizeof(std::int32_t) + disk_dimension * sizeof(float);
  if (record_size == 0 || static_cast<std::uintmax_t>(file_size) % record_size != 0) {
    throw std::runtime_error("Vector file has a truncated or inconsistent record: " + filename);
  }

  count = static_cast<size_t>(static_cast<std::uintmax_t>(file_size) / record_size);
  dimension = disk_dimension;
  auto *data = new float[dimension * count];
  input.seekg(0, std::ios::beg);
  for (size_t i = 0; i < count; ++i) {
    std::int32_t record_dimension = 0;
    input.read(reinterpret_cast<char *>(&record_dimension), sizeof(record_dimension));
    if (!input || record_dimension != dimension_on_disk) {
      delete[] data;
      throw std::runtime_error("Vector dimensions are inconsistent in: " + filename);
    }
    input.read(reinterpret_cast<char *>(data + i * dimension), dimension * sizeof(float));
    if (!input) {
      delete[] data;
      throw std::runtime_error("Could not read vector record from: " + filename);
    }
  }
  std::cerr << "Loaded " << count << " vectors of dimension " << dimension << " from " << filename << '\n';
  return data;
}

inline std::vector<lssg::labelset_t> LoadLabelSets(const std::string &filename)
{
  std::ifstream input(filename);
  if (!input) {
    throw std::runtime_error("Cannot open label file: " + filename);
  }

  std::vector<lssg::labelset_t> label_sets;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }

    lssg::labelset_t labels;
    std::stringstream line_stream(line);
    std::string token;
    while (std::getline(line_stream, token, ',')) {
      const auto first = token.find_first_not_of(" \t\n\r\f\v");
      const auto last = token.find_last_not_of(" \t\n\r\f\v");
      if (first == std::string::npos) {
        continue;
      }
      token = token.substr(first, last - first + 1);
      try {
        size_t parsed = 0;
        const unsigned long long value = std::stoull(token, &parsed);
        if (parsed != token.size() || token.front() == '-') {
          throw std::invalid_argument("invalid label token");
        }
        if (value > std::numeric_limits<lssg::labelsetmember_t>::max()) {
          throw std::out_of_range("label value out of range");
        }
        labels.push_back(static_cast<lssg::labelsetmember_t>(value));
      } catch (const std::exception &) {
        throw std::runtime_error("Invalid label '" + token + "' on line " + std::to_string(line_number) +
            " in " + filename);
      }
    }

    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    if (!labels.empty()) {
      label_sets.emplace_back(std::move(labels));
    }
  }
  std::cerr << "Loaded " << label_sets.size() << " label sets from " << filename << '\n';
  return label_sets;
}

inline std::vector<std::vector<lssg::label_t>> LoadGroundTruth(const std::string &filename)
{
  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open ground-truth file: " + filename);
  }

  std::vector<std::vector<lssg::label_t>> ground_truth;
  while (true) {
    std::int32_t result_count = 0;
    input.read(reinterpret_cast<char *>(&result_count), sizeof(result_count));
    if (input.eof()) {
      break;
    }
    if (!input || result_count < 0) {
      throw std::runtime_error("Invalid ground-truth record in: " + filename);
    }

    std::vector<lssg::label_t> result(static_cast<size_t>(result_count));
    for (auto &id : result) {
      std::uint32_t stored_id = 0;
      input.read(reinterpret_cast<char *>(&stored_id), sizeof(stored_id));
      if (!input) {
        throw std::runtime_error("Truncated ground-truth record in: " + filename);
      }
      id = stored_id;
    }
    ground_truth.emplace_back(std::move(result));
  }
  std::cerr << "Loaded " << ground_truth.size() << " ground-truth queries from " << filename << '\n';
  return ground_truth;
}

inline bool LabelFilterMatch(const lssg::labelset_t &base_set, const lssg::labelset_t &query_set,
    lssg::LabelFilterConfig::Type query_type)
{
  switch (query_type) {
    case lssg::LabelFilterConfig::LABEL_CONTAINMENT: {
      size_t query_index = 0;
      size_t base_index = 0;
      while (query_index < query_set.size() && base_index < base_set.size()) {
        if (query_set[query_index] == base_set[base_index]) {
          ++query_index;
          ++base_index;
        } else if (query_set[query_index] < base_set[base_index]) {
          return false;
        } else {
          ++base_index;
        }
      }
      return query_index == query_set.size();
    }
    case lssg::LabelFilterConfig::LABEL_EQUALITY:
      return base_set == query_set;
    case lssg::LabelFilterConfig::LABEL_OVERLAP: {
      size_t base_index = 0;
      size_t query_index = 0;
      while (base_index < base_set.size() && query_index < query_set.size()) {
        if (base_set[base_index] == query_set[query_index]) {
          return true;
        }
        if (base_set[base_index] < query_set[query_index]) {
          ++base_index;
        } else {
          ++query_index;
        }
      }
      return false;
    }
    default:
      throw std::runtime_error("Unknown label filter type");
  }
}

inline float CalculateRecall(const std::vector<std::vector<lssg::label_t>> &ground_truth,
    const std::vector<std::vector<lssg::label_t>> &results)
{
  const size_t query_count = std::min(ground_truth.size(), results.size());
  size_t expected = 0;
  size_t found = 0;
  for (size_t i = 0; i < query_count; ++i) {
    expected += ground_truth[i].size();
    for (const auto id : results[i]) {
      if (std::find(ground_truth[i].begin(), ground_truth[i].end(), id) != ground_truth[i].end()) {
        ++found;
      }
    }
  }
  return expected == 0 ? 1.0F : static_cast<float>(found) / static_cast<float>(expected);
}

inline std::vector<std::vector<lssg::label_t>> GenLabelSetGT(size_t base_count, size_t query_count, size_t dimension,
    size_t k, const float *base_vectors, const float *query_vectors,
    const std::vector<lssg::labelset_t> &base_sets, const std::vector<lssg::labelset_t> &query_sets,
    lssg::LabelFilterConfig::Type query_type, const std::string &space)
{
  if (base_sets.size() != base_count || query_sets.size() != query_count || base_count == 0 || query_count == 0) {
    throw std::runtime_error("Invalid vector/label counts for ground-truth generation");
  }

  std::unique_ptr<lssg::SpaceInterface<float>> space_ptr;
  if (space == "l2") {
    space_ptr = std::make_unique<lssg::L2Space>(dimension);
  } else if (space == "ip") {
    space_ptr = std::make_unique<lssg::InnerProductSpace>(dimension);
  } else {
    throw std::runtime_error("Unsupported space type: " + space);
  }

  const auto distance_function = space_ptr->get_dist_func();
  const auto distance_parameters = space_ptr->get_dist_func_param();
  std::vector<std::vector<lssg::label_t>> ground_truth(query_count);
  const auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic)
  for (size_t query_id = 0; query_id < query_count; ++query_id) {
    std::vector<lssg::dist_id_pair> candidates;
    candidates.reserve(k + 1);
    for (size_t base_id = 0; base_id < base_count; ++base_id) {
      if (!LabelFilterMatch(base_sets[base_id], query_sets[query_id], query_type)) {
        continue;
      }
      const auto distance = distance_function(query_vectors + query_id * dimension,
          base_vectors + base_id * dimension, distance_parameters);
      PUSH_HEAP(candidates, distance, static_cast<lssg::tableint>(base_id));
      if (candidates.size() > k) {
        POP_HEAP(candidates);
      }
    }
    ground_truth[query_id].resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      ground_truth[query_id][i] = candidates[i].id_;
    }
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  std::cerr << "Generated label ground truth for " << query_count << " queries in " << seconds << " seconds\n";
  return ground_truth;
}

inline void ValidateLabelSetGTSample(size_t base_count, size_t query_count, size_t dimension, size_t k,
    const float *base_vectors, const float *query_vectors, const std::vector<lssg::labelset_t> &base_sets,
    const std::vector<lssg::labelset_t> &query_sets, const std::vector<std::vector<lssg::label_t>> &ground_truth,
    size_t sample_count, lssg::LabelFilterConfig::Type query_type, const std::string &space,
    const std::string &ground_truth_file)
{
  if (sample_count == 0) {
    return;
  }
  if (base_count == 0 || query_count == 0 || k == 0 || base_vectors == nullptr || query_vectors == nullptr ||
      base_sets.size() != base_count || query_sets.size() != query_count || ground_truth.size() < query_count) {
    throw std::runtime_error("Invalid inputs for ground-truth validation");
  }

  const size_t queries_to_validate = std::min(sample_count, query_count);
  const std::vector<lssg::labelset_t> sampled_query_sets(query_sets.begin(),
      query_sets.begin() + static_cast<std::vector<lssg::labelset_t>::difference_type>(queries_to_validate));
  const auto expected = GenLabelSetGT(base_count, queries_to_validate, dimension, k, base_vectors, query_vectors,
      base_sets, sampled_query_sets, query_type, space);

  for (size_t query_id = 0; query_id < queries_to_validate; ++query_id) {
    auto actual_ids = ground_truth[query_id];
    auto expected_ids = expected[query_id];
    std::sort(actual_ids.begin(), actual_ids.end());
    std::sort(expected_ids.begin(), expected_ids.end());
    if (actual_ids != expected_ids) {
      throw std::runtime_error("Ground-truth validation failed for query " + std::to_string(query_id) +
          " in " + ground_truth_file + "; supplied results do not match brute-force results");
    }
  }

  std::cerr << "Validated " << queries_to_validate << " ground-truth queries against brute force for "
            << ground_truth_file << '\n';
}

} // namespace benchmark
