#pragma once

#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <unordered_set>
#include <sys/mman.h>
#include <iostream>
#include <string>

#ifndef unlikely
#if defined(__GNUC__) || defined(__clang__)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define unlikely(x) (x)
#endif
#endif

#define PUSH_HEAP(vec, ...)      \
  vec.emplace_back(__VA_ARGS__); \
  std::push_heap(vec.begin(), vec.end())

#define POP_HEAP(vec)                    \
  std::pop_heap(vec.begin(), vec.end()); \
  vec.pop_back();

#define TOP_HEAP(vec) vec.front()

namespace lssg {
typedef unsigned int  tableint;
typedef int           layer_t;
typedef size_t        label_t;
typedef uint32_t      labelsetmember_t;
typedef uint32_t      labelset_id_t;  // remember to change to uint32_t if labelsets > 65536
typedef float         dist_t;
typedef unsigned char vl_type;
using labelset_t = std::vector<labelsetmember_t>;
typedef uint64_t labelset_bitmap_t;

const tableint INVALID_TABLEINT = static_cast<tableint>(-1);

template <typename att_t>
struct lssg_range
{
  att_t l_{};
  att_t u_{};

  lssg_range(const att_t &l, const att_t &u) : l_(l), u_(u) {}
  lssg_range() = default;

  inline __attribute__((always_inline)) auto Test(const att_t &att) const -> bool { return att >= l_ && att <= u_; }
};

template <typename att_t>
struct lssg_set
{
  std::unordered_set<att_t>                  set_;
  void                                       Set(att_t i) { set_.insert(i); }
  inline __attribute__((always_inline)) bool Test(att_t i) const { return set_.find(i) != set_.end(); }
};

struct dist_id_pair
{
  dist_t   dist_;
  tableint id_;

  dist_id_pair() = default;

  dist_id_pair(dist_t dist, tableint id) : dist_(dist), id_(id) {}

  bool operator<(const dist_id_pair &rhs) const { return dist_ < rhs.dist_; }

  bool operator>(const dist_id_pair &rhs) const { return dist_ > rhs.dist_; }
};

struct FilterConfig
{
  virtual ~FilterConfig() = default;
};

struct LabelFilterConfig : public FilterConfig
{
  enum Type
  {
    LABEL_EQUALITY,     // exact match
    LABEL_CONTAINMENT,  // base label set contains query label set
    LABEL_OVERLAP,      // at least one common label
  } type_{LABEL_CONTAINMENT};
};

// Per-query tier-usage statistics collected during search (optional, compile-time gated)
struct TierUsageStats
{
  layer_t             top_layer{0};
  size_t              total_visited{0};
  std::vector<size_t> visited_per_layer;
  std::vector<size_t> candidates_per_layer;
  double              selectivity{0.0};

  // Filter-valid expansion ratio counters (per tier/layer)
  std::vector<size_t> total_scanned_per_layer;  // all scanned neighbors
  std::vector<size_t> valid_scanned_per_layer;  // subset passing filter

  void Reset(layer_t max_layer)
  {
    top_layer = max_layer;
    visited_per_layer.assign(max_layer + 1, 0);
    candidates_per_layer.assign(max_layer + 1, 0);
    total_scanned_per_layer.assign(max_layer + 1, 0);
    valid_scanned_per_layer.assign(max_layer + 1, 0);
    total_visited = 0;
    selectivity   = 0.0;
  }
};

}  // namespace lssg

namespace std {
template <>
struct hash<lssg::labelset_t>
{
  std::size_t operator()(const lssg::labelset_t &v) const
  {
    std::size_t seed = v.size();
    for (auto &i : v) {
      seed ^= std::hash<uint32_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
}  // namespace std