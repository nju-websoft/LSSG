#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 9
// GCC 9 advertises class NTTPs but cannot instantiate pybind11 2.13's
// optional typing::Literal machinery under C++20. LSSG does not use that
// machinery; disabling only this optional section keeps GCC 9 supported.
#ifdef __cpp_nontype_template_parameter_class
#undef __cpp_nontype_template_parameter_class
#endif
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../lssg/config.hh"
#include "../lssg/index.hh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using LabelSet = lssg::labelset_t;
using Index = lssg::LSSGIndex<LabelSet, float, lssg::LabelSetScopeKernel>;

int thread_count(size_t requested)
{
  if (requested == 0) {
    return omp_get_max_threads();
  }
  if (requested > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("threads is too large");
  }
  return static_cast<int>(requested);
}

void require_vector(const FloatArray &vectors, size_t dimension, const char *name)
{
  const auto info = vectors.request();
  if (info.ndim != 1 || static_cast<size_t>(info.shape[0]) != dimension) {
    throw std::invalid_argument(std::string(name) + " must have shape (" + std::to_string(dimension) + ",)");
  }
}

size_t require_matrix(const FloatArray &vectors, size_t dimension, const char *name)
{
  const auto info = vectors.request();
  if (info.ndim != 2 || static_cast<size_t>(info.shape[1]) != dimension) {
    throw std::invalid_argument(std::string(name) + " must have shape (n, " + std::to_string(dimension) + ")");
  }
  return static_cast<size_t>(info.shape[0]);
}

LabelSet normalize_label_set(const LabelSet &labels)
{
  LabelSet normalized = labels;
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
  return normalized;
}

std::vector<LabelSet> normalize_label_sets(const std::vector<LabelSet> &label_sets)
{
  std::vector<LabelSet> normalized;
  normalized.reserve(label_sets.size());
  for (const auto &labels : label_sets) {
    normalized.emplace_back(normalize_label_set(labels));
  }
  return normalized;
}

lssg::LabelFilterConfig::Type parse_filter(const std::string &filter)
{
  if (filter == "containment") {
    return lssg::LabelFilterConfig::LABEL_CONTAINMENT;
  }
  if (filter == "equality") {
    return lssg::LabelFilterConfig::LABEL_EQUALITY;
  }
  if (filter == "overlap") {
    return lssg::LabelFilterConfig::LABEL_OVERLAP;
  }
  throw std::invalid_argument("filter must be containment, equality, or overlap");
}

class LSSGIndex
{
public:
  LSSGIndex(size_t max_elements, size_t dimension, size_t M = lssg::config::kDefaultM,
      size_t ef_construction = lssg::config::kDefaultEfConstruction, const std::string &space = "l2",
      const std::string &label_file = "")
      : dimension_(dimension)
  {
    if (max_elements == 0 || dimension == 0 || M == 0 || ef_construction == 0) {
      throw std::invalid_argument("max_elements, dimension, M, and ef_construction must be greater than zero");
    }
    if (space != "l2" && space != "ip") {
      throw std::invalid_argument("space must be l2 or ip");
    }

    auto scope = std::make_unique<lssg::LabelSetScopeKernel>(lssg::config::kMaxLayers, max_elements, label_file,
        lssg::ThresholdDivision::UNIFORM);
    scope->Init();
    index_ = std::make_unique<Index>(max_elements, dimension, M, ef_construction, space, scope);
  }

  static std::unique_ptr<LSSGIndex> load(
      const std::string &index_file, const std::string &scope_file, const std::string &space = "l2")
  {
    return std::unique_ptr<LSSGIndex>(new LSSGIndex(index_file, scope_file, space, LoadTag{}));
  }

  void insert(lssg::label_t label, const FloatArray &vector, const LabelSet &labels)
  {
    require_vector(vector, dimension_, "vector");
    const auto *data = static_cast<const float *>(vector.request().ptr);
    const auto normalized = normalize_label_set(labels);
    py::gil_scoped_release release;
    index_->insert(label, data, normalized);
  }

  void add_items(const FloatArray &vectors, const std::vector<LabelSet> &label_sets, size_t start_id = 0,
      size_t threads = 0)
  {
    const size_t count = require_matrix(vectors, dimension_, "vectors");
    if (count != label_sets.size()) {
      throw std::invalid_argument("vectors and label_sets must contain the same number of items");
    }
    if (count == 0) {
      return;
    }
    if (count > index_->GetMaxElements() - index_->GetCurrentCount()) {
      throw std::invalid_argument("the batch exceeds the remaining index capacity");
    }
    if (start_id > std::numeric_limits<lssg::label_t>::max() - (count - 1)) {
      throw std::invalid_argument("start_id and batch size overflow label_t");
    }

    const auto normalized = normalize_label_sets(label_sets);
    const auto *data = static_cast<const float *>(vectors.request().ptr);
    const int workers = thread_count(threads);
    py::gil_scoped_release release;
#pragma omp parallel for num_threads(workers) schedule(dynamic, 64)
    for (size_t i = 0; i < count; ++i) {
      index_->insert(static_cast<lssg::label_t>(start_id + i), data + i * dimension_, normalized[i]);
    }
  }

  py::tuple search(const FloatArray &queries, const std::vector<LabelSet> &query_labels, size_t k, size_t ef,
      const std::string &filter, size_t threads = 0)
  {
    const size_t count = require_matrix(queries, dimension_, "queries");
    if (count != query_labels.size()) {
      throw std::invalid_argument("queries and query_labels must contain the same number of items");
    }
    if (k == 0 || ef < k) {
      throw std::invalid_argument("k must be greater than zero and ef must be at least k");
    }

    const auto normalized = normalize_label_sets(query_labels);
    const auto *data = static_cast<const float *>(queries.request().ptr);
    const int workers = thread_count(threads);
    lssg::LabelFilterConfig filter_config;
    filter_config.type_ = parse_filter(filter);
    std::vector<std::vector<std::pair<lssg::dist_t, lssg::label_t>>> results(count);

    {
      py::gil_scoped_release release;
#pragma omp parallel for num_threads(workers) schedule(dynamic, 64)
      for (size_t i = 0; i < count; ++i) {
        results[i] = index_->searchKNN(data + i * dimension_, ef, k, normalized[i], filter_config);
      }
    }

    py::array_t<std::int64_t> ids({static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(k)});
    py::array_t<float> distances({static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(k)});
    auto ids_view = ids.mutable_unchecked<2>();
    auto distances_view = distances.mutable_unchecked<2>();
    for (size_t i = 0; i < count; ++i) {
      for (size_t j = 0; j < k; ++j) {
        ids_view(i, j) = -1;
        distances_view(i, j) = std::numeric_limits<float>::infinity();
      }
      for (size_t j = 0; j < results[i].size() && j < k; ++j) {
        if (results[i][j].second > static_cast<lssg::label_t>(std::numeric_limits<std::int64_t>::max())) {
          throw std::runtime_error("result ID does not fit in int64");
        }
        distances_view(i, j) = results[i][j].first;
        ids_view(i, j) = static_cast<std::int64_t>(results[i][j].second);
      }
    }
    return py::make_tuple(std::move(ids), std::move(distances));
  }

  void save(const std::string &index_file, const std::string &scope_file)
  {
    const auto index_parent = std::filesystem::path(index_file).parent_path();
    const auto scope_parent = std::filesystem::path(scope_file).parent_path();
    if (!index_parent.empty()) {
      std::filesystem::create_directories(index_parent);
    }
    if (!scope_parent.empty()) {
      std::filesystem::create_directories(scope_parent);
    }
    py::gil_scoped_release release;
    index_->save(index_file, scope_file);
  }

  size_t dimension() const { return index_->GetDimension(); }
  size_t size() const { return index_->GetCurrentCount(); }
  size_t max_elements() const { return index_->GetMaxElements(); }
  size_t distance_computations() const { return index_->metric_dist_comps_.load(std::memory_order_relaxed); }
  size_t hops() const { return index_->metric_hops_.load(std::memory_order_relaxed); }

private:
  struct LoadTag
  {};

  LSSGIndex(const std::string &index_file, const std::string &scope_file, const std::string &space, LoadTag)
  {
    if (space != "l2" && space != "ip") {
      throw std::invalid_argument("space must be l2 or ip");
    }
    index_ = std::make_unique<Index>(index_file, scope_file, space);
    dimension_ = index_->GetDimension();
  }

  size_t dimension_{0};
  std::unique_ptr<Index> index_;
};

} // namespace

PYBIND11_MODULE(_pylssg, module)
{
  module.doc() = "Python bindings for the LSSG label-filtering ANN index";

  py::class_<LSSGIndex>(module, "LSSGIndex")
      .def(py::init<size_t, size_t, size_t, size_t, const std::string &, const std::string &>(),
          py::arg("max_elements"), py::arg("dimension"), py::arg("M") = lssg::config::kDefaultM,
          py::arg("ef_construction") = lssg::config::kDefaultEfConstruction, py::arg("space") = "l2",
          py::arg("label_file") = "")
      .def_static("load", &LSSGIndex::load, py::arg("index_file"), py::arg("scope_file"), py::arg("space") = "l2")
      .def("insert", &LSSGIndex::insert, py::arg("label"), py::arg("vector"), py::arg("labels"))
      .def("add_items", &LSSGIndex::add_items, py::arg("vectors"), py::arg("label_sets"),
          py::arg("start_id") = 0, py::arg("threads") = 0)
      .def("bulk_insert", &LSSGIndex::add_items, py::arg("vectors"), py::arg("label_sets"),
          py::arg("start_id") = 0, py::arg("threads") = 0)
      .def("search", &LSSGIndex::search, py::arg("queries"), py::arg("query_labels"), py::arg("k"),
          py::arg("ef"), py::arg("filter") = "containment", py::arg("threads") = 0)
      .def("search_batch", &LSSGIndex::search, py::arg("queries"), py::arg("query_labels"), py::arg("k"),
          py::arg("ef"), py::arg("filter") = "containment", py::arg("threads") = 0)
      .def("save", &LSSGIndex::save, py::arg("index_file"), py::arg("scope_file"))
      .def_property_readonly("dimension", &LSSGIndex::dimension)
      .def_property_readonly("size", &LSSGIndex::size)
      .def_property_readonly("max_elements", &LSSGIndex::max_elements)
      .def_property_readonly("distance_computations", &LSSGIndex::distance_computations)
      .def_property_readonly("hops", &LSSGIndex::hops);
}
