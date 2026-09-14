#pragma once
#include <mutex>
#include "utils.hh"

// SIMD intrinsics headers
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace lssg {
template <typename id_t = tableint>
class VisitedBaseClass
{
public:
  virtual void Clear()            = 0;
  virtual void Fill()             = 0;
  virtual void Set(id_t i)        = 0;
  virtual bool Test(id_t i) const = 0;
  virtual void Reset(id_t i)      = 0;
  virtual ~VisitedBaseClass()     = default;
};

// memory optimized Bitset, aligned to 64 bits and use aligned alloc for cache line alignment
template <typename id_t = tableint>
class lssg_bitset : public VisitedBaseClass<id_t>
{
public:
  static uint64_t *Allocate(size_t n_uint64)
  {
    if (n_uint64 == 0) {
      return nullptr;
    }
    const size_t n_bytes = n_uint64 * sizeof(uint64_t);
    const size_t alloc_bytes = (n_bytes + 63) / 64 * 64;
    auto *data = static_cast<uint64_t *>(aligned_alloc(64, alloc_bytes));
    if (data == nullptr) {
      throw std::runtime_error("fail to alloc for bitset");
    }
    return data;
  }

  lssg_bitset() = default;
  lssg_bitset(size_t n) : n_(n)
  {
    // Calculate number of uint64_t needed and allocate in uint64_t units to ensure proper alignment.
    size_t n_uint64 = n / 64 + (n % 64 != 0);
    size_t n_bytes  = n_uint64 * sizeof(uint64_t);
    data_           = Allocate(n_uint64);
    if (data_) {
      memset(data_, 0, n_bytes);
    }
  }

  void Resize(size_t n)
  {
    if (data_) {
      free(data_);
    }
    n_              = n;
    size_t n_uint64 = n / 64 + (n % 64 != 0);
    size_t n_bytes  = n_uint64 * sizeof(uint64_t);
    data_           = Allocate(n_uint64);
    num_ones_       = 0;
    last_1_         = id_t{};
    if (n_bytes != 0 && data_ == nullptr) {
      throw std::runtime_error("fail to alloc for bitset");
    }
    if (data_) {
      memset(data_, 0, n_bytes);
    }
  }

  // Copy constructor
  lssg_bitset(const lssg_bitset &other) : n_(other.n_), num_ones_(other.num_ones_), last_1_(other.last_1_)
  {
    size_t n_uint64 = n_ / 64 + (n_ % 64 != 0);
    size_t n_bytes  = n_uint64 * sizeof(uint64_t);
    data_           = Allocate(n_uint64);
    if (n_bytes != 0) {
      memcpy(data_, other.data_, n_bytes);
    }
  }

  // Copy assignment operator
  lssg_bitset &operator=(const lssg_bitset &other)
  {
    if (this != &other) {
      // Free old data
      if (data_) {
        free(data_);
      }

      // Copy members
      n_        = other.n_;
      num_ones_ = other.num_ones_;
      last_1_   = other.last_1_;

      // Allocate and copy data
      size_t n_uint64 = n_ / 64 + (n_ % 64 != 0);
      size_t n_bytes  = n_uint64 * sizeof(uint64_t);
      data_           = Allocate(n_uint64);
      if (n_bytes != 0) {
        memcpy(data_, other.data_, n_bytes);
      }
    }
    return *this;
  }

  // Move constructor
  lssg_bitset(lssg_bitset &&other) noexcept
      : n_(other.n_), num_ones_(other.num_ones_), data_(other.data_), last_1_(other.last_1_)
  {
    other.n_ = 0;
    other.num_ones_ = 0;
    other.data_ = nullptr;
    other.last_1_ = id_t{};
  }

  // Move assignment operator
  lssg_bitset &operator=(lssg_bitset &&other) noexcept
  {
    if (this != &other) {
      // Free old data
      if (data_) {
        free(data_);
      }

      // Move members
      n_          = other.n_;
      data_       = other.data_;
      other.n_    = 0;
      other.data_ = nullptr;
      num_ones_   = other.num_ones_;
      other.num_ones_ = 0;
      last_1_     = other.last_1_;
      other.last_1_ = id_t{};
    }
    return *this;
  }

  ~lssg_bitset() override
  {
    if (data_)
      free(data_);
  }

  inline __attribute__((always_inline)) void Set(id_t i) override
  {
    const uint64_t mask = 1ULL << (i % 64);
    if ((data_[i / 64] & mask) == 0) {
      data_[i / 64] |= mask;
      ++num_ones_;
    }
    last_1_ = i;
  }

  inline __attribute__((always_inline)) bool Test(id_t i) const override { return data_[i / 64] & (1ULL << (i % 64)); }

  inline __attribute__((always_inline)) void Reset(id_t i) override
  {
    const uint64_t mask = 1ULL << (i % 64);
    if (data_[i / 64] & mask) {
      data_[i / 64] &= ~mask;
      --num_ones_;
    }
  }

  inline __attribute__((always_inline)) auto GetData(id_t i) -> uint64_t * { return &data_[i / 64]; }

  inline __attribute__((always_inline)) void Clear() override
  {
    size_t n_uint64 = n_ / 64 + (n_ % 64 != 0);
    size_t n_bytes  = n_uint64 * sizeof(uint64_t);
    if (n_bytes != 0) {
      memset(data_, 0, n_bytes);
    }
    num_ones_ = 0;
  }

  inline __attribute__((always_inline)) void Fill() override
  {
    size_t n_uint64 = n_ / 64 + (n_ % 64 != 0);
    size_t n_bytes  = n_uint64 * sizeof(uint64_t);
    if (n_bytes != 0) {
      memset(data_, 0xFF, n_bytes);
    }
    num_ones_ = n_;
  }

  inline void bitintersect(const lssg_bitset<id_t> &other)
  {
    size_t n_uint64 = n_ / 64 + (n_ % 64 != 0);
    num_ones_       = 0;
    size_t i        = 0;

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    // AVX-512 path: process 8 uint64_t (512 bits) at a time
    const size_t simd_width = 8;
    const size_t simd_end   = (n_uint64 / simd_width) * simd_width;

    for (; i < simd_end; i += simd_width) {
      // Load 8 uint64_t values from both bitsets using unaligned load
      __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&data_[i]));
      __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&other.data_[i]));

      // Perform bitwise AND
      __m512i result = _mm512_and_si512(a, b);

      // Store result using unaligned store
      _mm512_storeu_si512(reinterpret_cast<__m512i *>(&data_[i]), result);

      // Count population using AVX-512 VPOPCNTDQ instruction
      __m512i popcnt = _mm512_popcnt_epi64(result);

      // Sum the population counts
      num_ones_ += _mm512_reduce_add_epi64(popcnt);
    }
#elif defined(__AVX2__)
    // AVX2 path: process 4 uint64_t (256 bits) at a time
    const size_t simd_width = 4;
    const size_t simd_end   = (n_uint64 / simd_width) * simd_width;

    for (; i < simd_end; i += simd_width) {
      // Load 4 uint64_t values from both bitsets using unaligned load
      __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&data_[i]));
      __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&other.data_[i]));

      // Perform bitwise AND
      __m256i result = _mm256_and_si256(a, b);

      // Store result using unaligned store
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(&data_[i]), result);

      // Count population - extract and use scalar popcnt
      // We need to access the data we just stored
      for (size_t j = 0; j < 4; ++j) {
        num_ones_ += __builtin_popcountll(data_[i + j]);
      }
    }
#endif

    // Scalar fallback for remaining elements
    for (; i < n_uint64; ++i) {
      data_[i] &= other.data_[i];
      num_ones_ += __builtin_popcountll(data_[i]);
    }
  }

  inline void bitunion(const lssg_bitset<id_t> &other)
  {
    size_t n_uint64 = n_ / 64 + (n_ % 64 != 0);
    num_ones_       = 0;
    size_t i        = 0;

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    // AVX-512 path: process 8 uint64_t (512 bits) at a time
    const size_t simd_width = 8;
    const size_t simd_end   = (n_uint64 / simd_width) * simd_width;

    for (; i < simd_end; i += simd_width) {
      // Load 8 uint64_t values from both bitsets using unaligned load
      __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&data_[i]));
      __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&other.data_[i]));

      // Perform bitwise OR
      __m512i result = _mm512_or_si512(a, b);

      // Store result using unaligned store
      _mm512_storeu_si512(reinterpret_cast<__m512i *>(&data_[i]), result);

      // Count population using AVX-512 VPOPCNTDQ instruction
      __m512i popcnt = _mm512_popcnt_epi64(result);

      // Sum the population counts
      num_ones_ += _mm512_reduce_add_epi64(popcnt);
    }
#elif defined(__AVX2__)
    // AVX2 path: process 4 uint64_t (256 bits) at a time
    const size_t simd_width = 4;
    const size_t simd_end   = (n_uint64 / simd_width) * simd_width;

    for (; i < simd_end; i += simd_width) {
      // Load 4 uint64_t values from both bitsets using unaligned load
      __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&data_[i]));
      __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&other.data_[i]));

      // Perform bitwise OR
      __m256i result = _mm256_or_si256(a, b);

      // Store result using unaligned store
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(&data_[i]), result);

      // Count population - extract and use scalar popcnt
      // We need to access the data we just stored
      for (size_t j = 0; j < 4; ++j) {
        num_ones_ += __builtin_popcountll(data_[i + j]);
      }
    }
#endif

    // Scalar fallback for remaining elements
    for (; i < n_uint64; ++i) {
      data_[i] |= other.data_[i];
      num_ones_ += __builtin_popcountll(data_[i]);
    }
  }

  // Find the first (lowest-index) set bit. Returns n_ if no bit is set.
  inline id_t FindFirst1() const
  {
    size_t n_uint64 = n_ / 64 + (n_ % 64 != 0);
    size_t i        = 0;

#if defined(__AVX512F__)
    const size_t simd_width = 8;
    const size_t simd_end   = (n_uint64 / simd_width) * simd_width;
    const __m512i zero      = _mm512_setzero_si512();
    for (; i < simd_end; i += simd_width) {
      __m512i chunk = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&data_[i]));
      // mask bit k=1 when lane k != 0
      __mmask8 mask = ~_mm512_cmpeq_epi64_mask(chunk, zero);
      if (mask) {
        size_t w = i + __builtin_ctz(mask);
        return static_cast<id_t>(w * 64 + __builtin_ctzll(data_[w]));
      }
    }
#elif defined(__AVX2__)
    const size_t simd_width = 4;
    const size_t simd_end   = (n_uint64 / simd_width) * simd_width;
    const __m256i zero      = _mm256_setzero_si256();
    for (; i < simd_end; i += simd_width) {
      __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&data_[i]));
      // movemask_epi8: 8 bits per 64-bit lane, all-1 if lane==0, all-0 if lane!=0
      int mask = ~_mm256_movemask_epi8(_mm256_cmpeq_epi64(chunk, zero));
      if (mask) {
        size_t w = i + __builtin_ctz(static_cast<unsigned>(mask)) / 8;
        return static_cast<id_t>(w * 64 + __builtin_ctzll(data_[w]));
      }
    }
#endif

    for (; i < n_uint64; ++i) {
      if (data_[i]) {
        return static_cast<id_t>(i * 64 + __builtin_ctzll(data_[i]));
      }
    }
    return static_cast<id_t>(n_);
  }

public:
  size_t    n_{};
  size_t    num_ones_{0};
  uint64_t *data_{nullptr};
  id_t      last_1_{};
};

template <typename id_t = tableint>
class VisitedList : public VisitedBaseClass<id_t>
{
public:
  VisitedList() = default;

  VisitedList(int numelements1)
  {
    curV_        = -1;
    numelements_ = numelements1;
    mass_        = Allocate(numelements_);
  }

  void Resize(int numelements1)
  {
    free(mass_);
    curV_        = -1;
    numelements_ = numelements1;
    mass_        = Allocate(numelements_);
  }

  inline void Clear() override
  {
    curV_++;
    if (curV_ == 0) {
      memset(mass_, 0, sizeof(vl_type) * numelements_);
      curV_++;
    }
  }

  inline void Fill() override
  {
    for (size_t i = 0; i < numelements_; ++i)
      mass_[i] = curV_;
    num_ones_ = numelements_;
    last_1_   = numelements_ - 1;
  }

  inline __attribute__((always_inline)) void Set(id_t i) override
  {
    mass_[i] = curV_;
    ++num_ones_;
    last_1_ = i;
  }

  inline __attribute__((always_inline)) bool Test(id_t i) const override { return (mass_[i] == curV_); }

  inline __attribute__((always_inline)) void Reset(id_t i) override { mass_[i] = -1; }

  inline __attribute__((always_inline)) auto GetData(id_t i) -> vl_type * { return &mass_[i]; }

  ~VisitedList() override { free(mass_); }

private:
  static vl_type *Allocate(int numelements)
  {
    if (numelements <= 0) {
      return nullptr;
    }
    const size_t bytes = (static_cast<size_t>(numelements) * sizeof(vl_type) + 63) & ~size_t(63);
    return static_cast<vl_type *>(aligned_alloc(64, bytes));
  }

public:
  vl_type      curV_{0};
  vl_type     *mass_{nullptr};
  unsigned int numelements_{0};
  size_t       num_ones_{0};
  id_t         last_1_{};
};

template <typename VisitedType = lssg_bitset<tableint>>
class VisitedPool
{
public:
  /**
   * @brief Construct a new Visited Bit Set Pool object
   *
   * @param n number of elements for each bitset to store
   * @param pool_size number of bitset to store
   */
  VisitedPool() = default;

  ~VisitedPool()
  {
    for (auto bs : pool_) {
      delete bs;
    }
  }

  void Init(size_t n) { n_ = n; }

  inline __attribute__((always_inline)) auto Get() -> VisitedType *
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pool_.empty()) {
      return new VisitedType(n_);
    }
    auto bs = pool_.back();
    pool_.pop_back();
    return bs;
  }

  inline void Return(VisitedType *bs)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    pool_.push_back(bs);
  }

private:
  size_t                     n_{};
  std::vector<VisitedType *> pool_;
  std::mutex                 mtx_;
};
}  // namespace lssg
