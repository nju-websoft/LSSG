#include "../lssg/config.hh"
#include "../lssg/poindex.hh"
#include "bench_utils.hh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGroundTruthValidationSamples = 3;

struct QueryScenario
{
  lssg::LabelFilterConfig::Type type;
  std::string name;
  std::string gt_file;
  std::vector<std::vector<lssg::label_t>> gt_data;
};

std::string next_value(int &index, int argc, char **argv, const std::string &option)
{
  if (index + 1 >= argc) {
    throw std::runtime_error("Missing value for " + option);
  }
  return argv[++index];
}

std::vector<size_t> parse_efs(const std::string &value)
{
  std::vector<size_t> efs;
  size_t begin = 0;
  while (begin < value.size()) {
    size_t end = value.find(',', begin);
    std::string token = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (token.empty()) {
      throw std::runtime_error("Invalid empty EF value in --efs");
    }
    size_t ef = std::stoull(token);
    if (ef == 0) {
      throw std::runtime_error("EF values must be greater than zero");
    }
    efs.push_back(ef);
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  if (efs.empty()) {
    throw std::runtime_error("--efs must contain at least one value");
  }
  return efs;
}

void print_usage(const char *program)
{
  std::cout << "Usage: " << program << " --base_vec FILE --base_labelset FILE"
            << " --query_vec FILE --query_labelset FILE"
            << " --index_location FILE --scope_location FILE --space l2|ip"
            << " (--gt_containment FILE | --gt_equality FILE | --gt_overlap FILE) [options]\n\n"
            << "Options:\n"
            << "  --base_vec FILE      base vectors used for ground-truth validation\n"
            << "  --base_labelset FILE base labels used for ground-truth validation\n"
            << "  --k N                recall target (default: " << lssg::config::kDefaultK << ")\n"
            << "  --efs LIST           comma-separated EF values (default: 10,20,50,100,200,500,1000)\n"
            << "  --max_base_vectors N validate against the first N base vectors (0: all)\n"
            << "  --max_queries N      search only the first N queries (for smoke tests)\n";
}

} // namespace

int main(int argc, char **argv)
{
  std::string base_vec_file;
  std::string base_label_file;
  std::string query_vec_file;
  std::string query_label_file;
  std::string index_location;
  std::string scope_location;
  std::string space;
  std::string gt_containment;
  std::string gt_equality;
  std::string gt_overlap;
  size_t k = lssg::config::kDefaultK;
  size_t max_base_vectors = 0;
  size_t max_queries = 0;
  std::vector<size_t> efs = {10, 20, 50, 100, 200, 500, 1000};

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help" || option == "-h") {
        print_usage(argv[0]);
        return 0;
      } else if (option == "--base_vec") {
        base_vec_file = next_value(i, argc, argv, option);
      } else if (option == "--base_labelset") {
        base_label_file = next_value(i, argc, argv, option);
      } else if (option == "--query_vec") {
        query_vec_file = next_value(i, argc, argv, option);
      } else if (option == "--query_labelset") {
        query_label_file = next_value(i, argc, argv, option);
      } else if (option == "--k") {
        k = std::stoull(next_value(i, argc, argv, option));
      } else if (option == "--max_base_vectors") {
        max_base_vectors = std::stoull(next_value(i, argc, argv, option));
      } else if (option == "--efs") {
        efs = parse_efs(next_value(i, argc, argv, option));
      } else if (option == "--max_queries") {
        max_queries = std::stoull(next_value(i, argc, argv, option));
      } else if (option == "--index_location") {
        index_location = next_value(i, argc, argv, option);
      } else if (option == "--scope_location") {
        scope_location = next_value(i, argc, argv, option);
      } else if (option == "--space") {
        space = next_value(i, argc, argv, option);
      } else if (option == "--gt_containment") {
        gt_containment = next_value(i, argc, argv, option);
      } else if (option == "--gt_equality") {
        gt_equality = next_value(i, argc, argv, option);
      } else if (option == "--gt_overlap") {
        gt_overlap = next_value(i, argc, argv, option);
      } else {
        throw std::runtime_error("Unknown argument: " + option);
      }
    }

    if (query_vec_file.empty() || query_label_file.empty() || index_location.empty() || scope_location.empty() ||
        (space != "l2" && space != "ip") || k == 0 || efs.empty()) {
      throw std::runtime_error("Missing or invalid search arguments (use --help for usage)");
    }
    const size_t scenario_count = (!gt_containment.empty() ? 1 : 0) + (!gt_equality.empty() ? 1 : 0) +
        (!gt_overlap.empty() ? 1 : 0);
    if (scenario_count != 1) {
      throw std::runtime_error("Provide exactly one ground-truth file");
    }
    if (base_vec_file.empty() || base_label_file.empty()) {
      throw std::runtime_error("Base vectors and labels are required for ground-truth validation");
    }
    for (size_t ef : efs) {
      if (ef < k) {
        throw std::runtime_error("Every EF value must be at least k");
      }
    }

    size_t query_dim = 0;
    size_t query_count = 0;
    std::unique_ptr<float[]> query_vecs(benchmark::fvecs_read(query_vec_file, query_dim, query_count));
    auto query_labels = benchmark::LoadLabelSets(query_label_file);
    if (query_count != query_labels.size()) {
      throw std::runtime_error("Query vector count and query label set count do not match");
    }
    if (max_queries != 0) {
      if (max_queries > query_count) {
        throw std::runtime_error("--max_queries exceeds the available queries");
      }
      query_count = max_queries;
      query_labels.resize(query_count);
    }
    if (query_count == 0) {
      throw std::runtime_error("No queries to search");
    }

    std::unique_ptr<float[]> base_vecs;
    std::vector<lssg::labelset_t> base_labels;
    size_t base_dim = 0;
    size_t base_count = 0;
    base_vecs.reset(benchmark::fvecs_read(base_vec_file, base_dim, base_count));
    if (base_dim != query_dim) {
      throw std::runtime_error("Base and query vector dimensions do not match");
    }
    base_labels = benchmark::LoadLabelSets(base_label_file);
    if (base_count != base_labels.size()) {
      throw std::runtime_error("Base vector count and base label set count do not match");
    }
    if (max_base_vectors != 0) {
      if (max_base_vectors > base_count) {
        throw std::runtime_error("--max_base_vectors exceeds the available base data");
      }
      base_count = max_base_vectors;
      base_labels.resize(base_count);
    }
    if (base_count == 0) {
      throw std::runtime_error("No base vectors available for ground-truth validation");
    }

    std::vector<QueryScenario> scenarios;
    if (!gt_containment.empty()) {
      scenarios.push_back({lssg::LabelFilterConfig::LABEL_CONTAINMENT, "containment", gt_containment, {}});
    }
    if (!gt_equality.empty()) {
      scenarios.push_back({lssg::LabelFilterConfig::LABEL_EQUALITY, "equality", gt_equality, {}});
    }
    if (!gt_overlap.empty()) {
      scenarios.push_back({lssg::LabelFilterConfig::LABEL_OVERLAP, "overlap", gt_overlap, {}});
    }
    if (scenarios.empty()) {
      throw std::runtime_error("At least one ground-truth scenario is required");
    }

    for (auto &scenario : scenarios) {
      scenario.gt_data = benchmark::LoadGroundTruth(scenario.gt_file);
      if (scenario.gt_data.size() != query_count) {
        throw std::runtime_error("Ground truth query count does not match the query data for " + scenario.name);
      }
      benchmark::ValidateLabelSetGTSample(base_count, query_count, query_dim, k, base_vecs.get(), query_vecs.get(),
          base_labels, query_labels, scenario.gt_data, kGroundTruthValidationSamples, scenario.type, space,
          scenario.gt_file);
    }

    lssg::PoIndex<lssg::labelset_t, float, lssg::LabelSetScopeKernel> index(
        index_location, scope_location, space);

    std::cout << "ef,recall,qps,dist_comp,hops\n";
    for (const auto &scenario : scenarios) {
      lssg::LabelFilterConfig filter_config;
      filter_config.type_ = scenario.type;
      std::cerr << "Searching " << scenario.name << " queries\n";

      for (size_t ef : efs) {
        std::vector<std::vector<lssg::label_t>> results(query_count);
        double total_time = 0.0;
        index.metric_dist_comps_ = 0;
        index.metric_hops_ = 0;

        for (size_t query_id = 0; query_id < query_count; ++query_id) {
          auto start = std::chrono::high_resolution_clock::now();
          auto result = index.searchKNN(query_vecs.get() + query_id * query_dim, ef, k, query_labels[query_id],
              filter_config);
          auto end = std::chrono::high_resolution_clock::now();
          total_time += std::chrono::duration<double>(end - start).count();
          results[query_id].reserve(result.size());
          for (const auto &[distance, id] : result) {
            (void)distance;
            results[query_id].push_back(id);
          }
        }

        const float recall = benchmark::CalculateRecall(scenario.gt_data, results);
        const double qps = static_cast<double>(query_count) / total_time;
        std::cout << ef << ',' << recall << ',' << qps << ','
                  << static_cast<double>(index.metric_dist_comps_) / query_count << ','
                  << static_cast<double>(index.metric_hops_) / query_count << '\n';
      }
    }

    return 0;
  } catch (const std::exception &error) {
    std::cerr << "search_lssg: " << error.what() << '\n';
    return 2;
  }
}
