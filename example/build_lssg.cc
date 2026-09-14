#include "../lssg/config.hh"
#include "../lssg/poindex.hh"
#include "bench_utils.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char *program)
{
  std::cout << "Usage: " << program << " --basevec FILE --labelset FILE --space l2|ip"
            << " --index_location FILE --scope_location FILE [options]\n\n"
            << "Options:\n"
            << "  --m N              graph degree (default: " << lssg::config::kDefaultM << ")\n"
            << "  --efc N            construction beam width (default: " << lssg::config::kDefaultEfConstruction << ")\n"
            << "  --threads N        OpenMP build threads (default: max available)\n"
            << "  --max_vectors N    build only the first N vectors (for smoke tests)\n";
}

std::string next_value(int &index, int argc, char **argv, const std::string &option)
{
  if (index + 1 >= argc) {
    throw std::runtime_error("Missing value for " + option);
  }
  return argv[++index];
}

} // namespace

int main(int argc, char **argv)
{
  size_t m = lssg::config::kDefaultM;
  size_t efc = lssg::config::kDefaultEfConstruction;
  size_t max_vectors = 0;
  int threads = omp_get_max_threads();
  std::string basevec;
  std::string baseset;
  std::string space;
  std::string index_location;
  std::string scope_location;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help" || option == "-h") {
        print_usage(argv[0]);
        return 0;
      } else if (option == "--m") {
        m = std::stoull(next_value(i, argc, argv, option));
      } else if (option == "--efc") {
        efc = std::stoull(next_value(i, argc, argv, option));
      } else if (option == "--basevec") {
        basevec = next_value(i, argc, argv, option);
      } else if (option == "--labelset") {
        baseset = next_value(i, argc, argv, option);
      } else if (option == "--space") {
        space = next_value(i, argc, argv, option);
      } else if (option == "--threads") {
        threads = std::stoi(next_value(i, argc, argv, option));
      } else if (option == "--max_vectors") {
        max_vectors = std::stoull(next_value(i, argc, argv, option));
      } else if (option == "--index_location") {
        index_location = next_value(i, argc, argv, option);
      } else if (option == "--scope_location") {
        scope_location = next_value(i, argc, argv, option);
      } else {
        throw std::runtime_error("Unknown argument: " + option);
      }
    }

    if (m == 0 || efc == 0 || threads <= 0 || basevec.empty() || baseset.empty() ||
        (space != "l2" && space != "ip") || index_location.empty() || scope_location.empty()) {
      throw std::runtime_error("Missing or invalid build arguments (use --help for usage)");
    }

    size_t dim = 0;
    size_t vector_count = 0;
    std::unique_ptr<float[]> basevecs(benchmark::fvecs_read(basevec, dim, vector_count));
    auto label_sets = benchmark::LoadLabelSets(baseset);

    if (max_vectors != 0) {
      if (max_vectors > vector_count || max_vectors > label_sets.size()) {
        throw std::runtime_error("--max_vectors exceeds the available vectors or label sets");
      }
      vector_count = max_vectors;
      label_sets.resize(vector_count);
    }
    if (vector_count == 0 || label_sets.size() != vector_count) {
      throw std::runtime_error("Base vector count and label set count do not match");
    }

    const size_t max_layer = lssg::config::kMaxLayers;
    auto label_kernel = std::make_unique<lssg::LabelSetScopeKernel>(max_layer, vector_count, baseset,
        lssg::ThresholdDivision::UNIFORM, vector_count);
    auto init_start = std::chrono::high_resolution_clock::now();
    label_kernel->Init();
    auto init_end = std::chrono::high_resolution_clock::now();
    std::cerr << "Scope kernel initialized in "
              << std::chrono::duration<double>(init_end - init_start).count() << " seconds\n";

    lssg::PoIndex<lssg::labelset_t, float, lssg::LabelSetScopeKernel> index(
        vector_count, dim, m, efc, space, label_kernel);

    auto start = std::chrono::high_resolution_clock::now();
    std::atomic<size_t> inserted_count{0};
#pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
    for (size_t i = 0; i < vector_count; ++i) {
      index.insert(static_cast<lssg::label_t>(i), basevecs.get() + i * dim, label_sets[i]);
      const size_t inserted = inserted_count.fetch_add(1, std::memory_order_relaxed) + 1;
      if (inserted % 10000 == 0 || inserted == vector_count) {
#pragma omp critical(lssg_build_progress)
        std::cerr << "Inserted " << inserted << " / " << vector_count << " vectors\n";
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cerr << "Index built in " << std::chrono::duration<double>(end - start).count() << " seconds\n";

    const auto index_parent = std::filesystem::path(index_location).parent_path();
    const auto scope_parent = std::filesystem::path(scope_location).parent_path();
    if (!index_parent.empty()) {
      std::filesystem::create_directories(index_parent);
    }
    if (!scope_parent.empty()) {
      std::filesystem::create_directories(scope_parent);
    }
    index.save(index_location, scope_location);
    std::cerr << "Index saved to: " << index_location << "\n";

    return 0;
  } catch (const std::exception &error) {
    std::cerr << "build_lssg: " << error.what() << '\n';
    return 2;
  }
}
