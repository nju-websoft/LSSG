#pragma once
#include <atomic>
#include <cstddef>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace lssg {

/**
 * @brief Thread-safe LRU cache for storing Jaccard distance arrays
 * 
 * This cache uses simple LRU (Least Recently Used) eviction policy with O(1) operations.
 * Much faster and simpler than LRU-K, and sufficient for our workload where recently
 * inserted labelsets are likely to be queried repeatedly.
 * 
 * Thread-safety is achieved using:
 * - Independent sharded cache maps, each with its own lock
 * - Read-write locks (shared_mutex) for high read concurrency
 * - O(1) insertion, lookup, and eviction using hash map + doubly-linked list
 * 
 * KEY DESIGN: Each shard is completely independent with its own map, list, and lock.
 * This prevents cross-shard data races and maximizes concurrency.
 * 
 * @tparam KeyType Type of the cache key (labelset_id_t)
 * @tparam ValueType Type of the cached value (std::vector<double>)
 */
template <typename KeyType, typename ValueType>
class LRUCache
{
public:
  /**
   * @brief Construct a new LRUCache with specified capacity
   * 
   * @param capacity Maximum number of entries in the cache (total across all shards)
   * @param num_shards Number of independent cache shards (default: 64, must be power of 2)
   * @param name Optional name for this cache (for logging)
   */
  explicit LRUCache(size_t capacity, size_t num_shards = 64, const std::string &name = "LRUCache")
      : capacity_(capacity), num_shards_(num_shards), cache_name_(name), 
        total_hits_(0), total_misses_(0)
  {
    if (capacity == 0) {
      throw std::invalid_argument("Cache capacity must be greater than 0");
    }
    if (num_shards == 0 || (num_shards & (num_shards - 1)) != 0) {
      throw std::invalid_argument("Number of shards must be a power of 2");
    }

    // Initialize independent shards
    shards_ = std::make_unique<CacheShard[]>(num_shards_);
    
    // Distribute capacity evenly across shards (at least 1 per shard)
    size_t per_shard_capacity = std::max(size_t(1), capacity_ / num_shards_);
    for (size_t i = 0; i < num_shards_; ++i) {
      shards_[i].capacity = per_shard_capacity;
    }
  }

  /**
   * @brief Destructor - prints cache statistics
   */
  ~LRUCache()
  {
    size_t hits = total_hits_.load(std::memory_order_relaxed);
    size_t misses = total_misses_.load(std::memory_order_relaxed);
    size_t total_accesses = hits + misses;
    
    if (total_accesses > 0) {
      double hit_rate = (double)hits / total_accesses * 100.0;
      std::cerr << "[" << cache_name_ << "] Cache Statistics:" << std::endl;
      std::cerr << "  Total accesses: " << total_accesses << std::endl;
      std::cerr << "  Hits: " << hits << " (" << hit_rate << "%)" << std::endl;
      std::cerr << "  Misses: " << misses << " (" << (100.0 - hit_rate) << "%)" << std::endl;
      std::cerr << "  Final size: " << size() << " / " << capacity_ << std::endl;
    }
  }

  /**
   * @brief Get value from cache (thread-safe, O(1))
   * 
   * @param key The key to lookup
   * @param out_value Output parameter for the value if found
   * @return true if key found in cache, false otherwise
   */
  bool get(const KeyType &key, ValueType &out_value)
  {
    size_t shard_idx = get_shard_index(key);
    auto &shard = shards_[shard_idx];

    std::unique_lock<std::shared_mutex> lock(shard.mutex);

    auto it = shard.cache_map.find(key);
    if (it == shard.cache_map.end()) {
      // total_misses_.fetch_add(1, std::memory_order_relaxed);
      return false;  // Cache miss
    }

    // Cache hit!
    // total_hits_.fetch_add(1, std::memory_order_relaxed);

    // Move accessed item to front of LRU list (most recently used)
    shard.lru_list.splice(shard.lru_list.begin(), shard.lru_list, it->second);

    out_value = it->second->value;  // Copy the value
    return true;                     // Cache hit
  }

  /**
   * @brief Put value into cache (thread-safe, O(1))
   * 
   * @param key The key to store
   * @param value The value to store
   */
  void put(const KeyType &key, const ValueType &value)
  {
    size_t shard_idx = get_shard_index(key);
    auto &shard = shards_[shard_idx];

    std::unique_lock<std::shared_mutex> lock(shard.mutex);

    auto it = shard.cache_map.find(key);
    if (it != shard.cache_map.end()) {
      // Key already exists, update value and move to front
      it->second->value = value;
      shard.lru_list.splice(shard.lru_list.begin(), shard.lru_list, it->second);
      return;
    }

    // Need to insert new entry
    // Check if eviction is needed (O(1) eviction!)
    if (shard.cache_map.size() >= shard.capacity) {
      // Evict least recently used (back of list)
      auto lru_key = shard.lru_list.back().key;
      shard.cache_map.erase(lru_key);
      shard.lru_list.pop_back();
    }

    // Insert new entry at front of list (most recently used)
    shard.lru_list.push_front({key, value});
    shard.cache_map[key] = shard.lru_list.begin();
  }

  /**
   * @brief Check if key exists in cache (thread-safe, O(1))
   * 
   * @param key The key to check
   * @return true if key exists in cache, false otherwise
   */
  bool contains(const KeyType &key) const
  {
    size_t shard_idx = get_shard_index(key);
    const auto &shard = shards_[shard_idx];

    std::shared_lock<std::shared_mutex> lock(shard.mutex);
    return shard.cache_map.find(key) != shard.cache_map.end();
  }

  /**
   * @brief Clear all entries from cache (thread-safe)
   */
  void clear()
  {
    // Clear each shard independently
    for (size_t i = 0; i < num_shards_; ++i) {
      auto &shard = shards_[i];
      std::unique_lock<std::shared_mutex> lock(shard.mutex);
      shard.cache_map.clear();
      shard.lru_list.clear();
    }
  }

  /**
   * @brief Get current cache size (thread-safe)
   * 
   * @return size_t Current number of entries in cache
   */
  size_t size() const
  {
    size_t total = 0;
    for (size_t i = 0; i < num_shards_; ++i) {
      const auto &shard = shards_[i];
      std::shared_lock<std::shared_mutex> lock(shard.mutex);
      total += shard.cache_map.size();
    }
    return total;
  }

  /**
   * @brief Get cache capacity
   * 
   * @return size_t Maximum number of entries the cache can hold
   */
  size_t capacity() const { return capacity_; }

  /**
   * @brief Get cache statistics (thread-safe)
   * 
   * @return Cache statistics
   */
  struct CacheStats
  {
    size_t current_size;
    size_t capacity;
    size_t hits;
    size_t misses;
    double hit_rate;  // Percentage (0-100)
  };

  CacheStats get_stats() const
  {
    CacheStats stats;
    stats.current_size = size();
    stats.capacity     = capacity_;
    stats.hits         = total_hits_.load(std::memory_order_relaxed);
    stats.misses       = total_misses_.load(std::memory_order_relaxed);
    size_t total       = stats.hits + stats.misses;
    stats.hit_rate     = (total > 0) ? ((double)stats.hits / total * 100.0) : 0.0;
    return stats;
  }

private:
  // LRU list node - stores key and value
  struct ListNode
  {
    KeyType   key;
    ValueType value;
  };

  // Independent cache shard - each has its own list, map, and lock
  // This prevents cross-shard data races and maximizes concurrency
  struct CacheShard
  {
    // Doubly-linked list for LRU ordering (front = most recent, back = least recent)
    std::list<ListNode> lru_list;
    
    // Hash map for O(1) lookup: key -> iterator to position in lru_list
    std::unordered_map<KeyType, typename std::list<ListNode>::iterator> cache_map;
    
    // Lock for this shard
    mutable std::shared_mutex mutex;
    
    // Capacity for this shard
    size_t capacity{0};
  };

  /**
   * @brief Get shard index for a key (for lock striping)
   * 
   * @param key The key to hash
   * @return size_t Shard index in [0, num_shards_)
   */
  size_t get_shard_index(const KeyType &key) const
  {
    size_t hash = std::hash<KeyType>{}(key);
    // Use bitwise AND for fast modulo (works because num_shards is power of 2)
    return hash & (num_shards_ - 1);
  }

  // Cache configuration
  const size_t capacity_;    // Maximum total number of entries across all shards
  const size_t num_shards_;  // Number of independent cache shards
  const std::string cache_name_;  // Name for logging

  // Independent cache shards - each shard is completely isolated
  // This is the KEY to thread-safety: no cross-shard data races possible
  std::unique_ptr<CacheShard[]> shards_;

  // Cache metrics (atomic for thread-safe updates)
  mutable std::atomic<size_t> total_hits_;    // Total cache hits
  mutable std::atomic<size_t> total_misses_;  // Total cache misses
};

}  // namespace lssg
