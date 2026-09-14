#include "bench_utils.hh"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

lssg::LabelFilterConfig::Type ParseLabelQueryType(const std::string &type_str)
{
  if (type_str == "containment") {
    return lssg::LabelFilterConfig::LABEL_CONTAINMENT;
  }
  if (type_str == "equality") {
    return lssg::LabelFilterConfig::LABEL_EQUALITY;
  }
  if (type_str == "overlap") {
    return lssg::LabelFilterConfig::LABEL_OVERLAP;
  }
  throw std::runtime_error("Unsupported label query type: " + type_str);
}

std::string NextValue(int &index, int argc, char **argv, const std::string &option)
{
  if (index + 1 >= argc) {
    throw std::runtime_error("Missing value for " + option);
  }
  return argv[++index];
}

void PrintUsage(const char *program)
{
  std::cout << "Usage: " << program
            << " --basevec FILE --queryvec FILE --base_labelset FILE --query_labelset FILE"
            << " --gt_file FILE --space l2|ip --type containment|equality|overlap [options]\n\n"
            << "Options:\n"
            << "  --k N                   ground-truth size (default: 10)\n"
            << "  --max_base_vectors N    generate from the first N base vectors\n"
            << "  --max_queries N         generate for the first N queries\n";
}

int main(int argc, char **argv)
{
  std::string base_vec_file, query_vec_file;
  std::string base_label_file, query_label_file;
  std::string gt_file, space, type_str;
  size_t      k = 10;
  size_t      max_base_vectors = 0;
  size_t      max_queries = 0;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help" || option == "-h") {
        PrintUsage(argv[0]);
        return 0;
      } else if (option == "--basevec") {
        base_vec_file = NextValue(i, argc, argv, option);
      } else if (option == "--queryvec") {
        query_vec_file = NextValue(i, argc, argv, option);
      } else if (option == "--base_labelset") {
        base_label_file = NextValue(i, argc, argv, option);
      } else if (option == "--query_labelset") {
        query_label_file = NextValue(i, argc, argv, option);
      } else if (option == "--gt_file") {
        gt_file = NextValue(i, argc, argv, option);
      } else if (option == "--space") {
        space = NextValue(i, argc, argv, option);
      } else if (option == "--type") {
        type_str = NextValue(i, argc, argv, option);
      } else if (option == "--k") {
        k = std::stoull(NextValue(i, argc, argv, option));
      } else if (option == "--max_base_vectors") {
        max_base_vectors = std::stoull(NextValue(i, argc, argv, option));
      } else if (option == "--max_queries") {
        max_queries = std::stoull(NextValue(i, argc, argv, option));
      } else {
        throw std::runtime_error("Unknown argument: " + option);
      }
    }

    if (base_vec_file.empty() || query_vec_file.empty() || base_label_file.empty() || query_label_file.empty() ||
        gt_file.empty() || space.empty() || type_str.empty() || k == 0) {
      throw std::runtime_error("Missing required arguments for gen_label_gt");
    }

  auto query_type = ParseLabelQueryType(type_str);

  size_t base_dim = 0, nb = 0;
  std::unique_ptr<float[]> base_vecs(benchmark::fvecs_read(base_vec_file, base_dim, nb));

  size_t query_dim = 0, nq = 0;
  std::unique_ptr<float[]> query_vecs(benchmark::fvecs_read(query_vec_file, query_dim, nq));

  if (base_dim != query_dim) {
    throw std::runtime_error("Base and query vector dimensions do not match");
  }

  auto base_labelsets  = benchmark::LoadLabelSets(base_label_file);
  auto query_labelsets = benchmark::LoadLabelSets(query_label_file);

  if (max_base_vectors != 0) {
    if (max_base_vectors > nb || max_base_vectors > base_labelsets.size()) {
      throw std::runtime_error("--max_base_vectors exceeds the available base data");
    }
    nb = max_base_vectors;
    base_labelsets.resize(nb);
  }
  if (max_queries != 0) {
    if (max_queries > nq || max_queries > query_labelsets.size()) {
      throw std::runtime_error("--max_queries exceeds the available query data");
    }
    nq = max_queries;
    query_labelsets.resize(nq);
  }

  if (base_labelsets.size() != nb) {
    throw std::runtime_error("Base label set count does not match base vector count");
  }
  if (query_labelsets.size() != nq) {
    throw std::runtime_error("Query vector count and query label set count do not match");
  }

  auto gt = benchmark::GenLabelSetGT(nb, nq, base_dim, k, base_vecs.get(), query_vecs.get(), base_labelsets, query_labelsets,
      query_type, space);

  const auto parent = std::filesystem::path(gt_file).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream ofs(gt_file, std::ios::binary);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot open gt file for writing: " + gt_file);
  }

  for (size_t iq = 0; iq < gt.size(); ++iq) {
    int count = static_cast<int>(gt[iq].size());
    ofs.write(reinterpret_cast<const char *>(&count), sizeof(int));
    for (const auto &id : gt[iq]) {
      unsigned int uid = static_cast<unsigned int>(id);
      ofs.write(reinterpret_cast<const char *>(&uid), sizeof(unsigned int));
    }
  }
  if (!ofs.good()) {
    throw std::runtime_error("Failed while writing ground-truth file: " + gt_file);
  }
  ofs.close();

  std::cout << "Ground truth generated: " << gt_file << std::endl;

    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gen_label_gt: " << error.what() << '\n';
    return 2;
  }
}
