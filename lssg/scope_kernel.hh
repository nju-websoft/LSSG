#pragma once
#include "utils.hh"
#include "disk.hh"

namespace lssg {
// filter kernel should define the following functions:
// init: how to calculate the scope of the lowest layer
// get: what is the scope of the given layer
// test: given a node's attribute and the layer, whether it is in the scope
// insert: given a node's attribute, insert it into the scope kernel to maintain the scope structure
// landing: given a query, return the top layer to start the search
template <typename att_t, typename scope_t>
class ScopeKernel
{
public:
  ScopeKernel() = default;
  explicit ScopeKernel(size_t top_layer, size_t max_elements)
      : cur_max_layer_(0), top_layer_(top_layer), max_elements_(max_elements)
  {}
  virtual ~ScopeKernel() = default;
  virtual void Init()    = 0;
  // Insert a node attribute into the kernel so that the kernel can determine the scope of each attribute
  virtual void Insert(const att_t &att, tableint inid) = 0;
  // Optimizer: given a query scope, which layer to land
  virtual auto Landing(scope_t &scope, std::vector<tableint> &entry_points, size_t ep_num) -> layer_t = 0;
  // Optimizer: when to raise the layer for all nodes, if true the index should copy all edges to top+1 layer
  virtual auto TryRaiseLayer() -> bool = 0;
  // Given a vertex's internal id and the scope, whether it is in the scope
  virtual auto TestInScope(tableint inid, const scope_t &scope) -> bool = 0;
  // Optimizer for insertion: given candidates, and the center node's internal id, which is the center of the scope,
  // which candidates are in the scope in this layer
  virtual auto FilterCandidates(const std::vector<dist_id_pair> &candidates, tableint center_inid, layer_t layer)
      -> std::vector<dist_id_pair> = 0;

  inline virtual auto GetEmptyScope() -> scope_t{
    return scope_t{};
  }

  inline virtual void MarkInnerIDinScope(scope_t &scope, tableint inid){
    (void)scope;
    (void)inid;
    return;
  }

  virtual auto NeedResearch(scope_t &marked_scope, scope_t &query_scope, std::vector<tableint> &entry_points, const FilterConfig &filter_config, size_t ep_num) -> bool{
    (void)marked_scope;
    (void)query_scope;
    (void)entry_points;
    (void)filter_config;
    (void)ep_num;
    return false;
  }

  virtual void Serialize(std::ostream &os) {
    (void)os;
    throw std::runtime_error("Serialize is not implemented");
  };
  virtual void Deserialize(std::istream &is) {
    (void)is;
    throw std::runtime_error("Deserialize is not implemented");
  };

public:
  size_t cur_max_layer_;
  size_t top_layer_;
  size_t max_elements_;
};

}  // namespace lssg
