#pragma once

#include <cstddef>
#include <cstdint>

// The CMake build defines LSSG_USE_MINHASH explicitly.  For a direct compiler
// invocation, MinHash remains the default so the public example is usable
// without any extra flags.  Set this to 0 (or pass -DLSSG_USE_MINHASH=0) for
// the IVF variant.
#ifndef LSSG_USE_MINHASH
#define LSSG_USE_MINHASH 1
#endif

#if LSSG_USE_MINHASH
#ifndef USE_MINHASH
#define USE_MINHASH
#endif
#endif

namespace lssg::config {

// Index layout and graph defaults.
inline constexpr std::size_t kMaxLayers = 8;
inline constexpr std::size_t kDefaultM = 16;
inline constexpr std::size_t kDefaultEfConstruction = 128;

// Label-index defaults.
inline constexpr std::size_t kDenseIndexMemoryBudgetBytes = 100ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kLabelsetEntryCapacity = 32;
inline constexpr std::size_t kQueryEntryPointCapacity = 5;
inline constexpr std::size_t kDefaultLandingEntryPoints = 32;
inline constexpr std::size_t kSearchMinEntryPoints = 16;
inline constexpr std::size_t kSearchMaxEntryPoints = 256;
inline constexpr std::size_t kEntryMutexCount = 64;
inline constexpr std::size_t kJaccardCacheCapacity = 10000;
inline constexpr std::size_t kJaccardCacheShards = 64;
inline constexpr double kSimilarLabelsetMargin = 0.05;
inline constexpr std::size_t kIvfCandidateBudget = 5000;
inline constexpr std::size_t kDenseLabelFrequencyFallback = 100;

// MinHash/LSH defaults.  num_minhash_funcs must be divisible by num_bands.
inline constexpr std::size_t kNumMinhashFunctions = 64;
inline constexpr std::size_t kNumBands = 16;
inline constexpr std::size_t kMinhashBitsPerHash = 4;
inline constexpr std::uint64_t kMinhashPrime = 4294967291ULL;

// Search defaults used by the benchmark executable.
inline constexpr std::size_t kDefaultK = 10;

// Index files are intentionally versioned.  The old files were produced with
// incompatible hard-coded configurations and must not be loaded silently.
inline constexpr std::uint64_t kIndexMagic = 0x4C53534749445831ULL; // "LSSGIDX1"
inline constexpr std::uint64_t kScopeMagic = 0x4C53534753434F31ULL; // "LSSGSCO1"
inline constexpr std::uint32_t kFileFormatVersion = 1;

static_assert(kNumMinhashFunctions > 0, "MinHash requires at least one hash function");
static_assert(kNumBands > 0, "MinHash requires at least one band");
static_assert(kNumMinhashFunctions % kNumBands == 0,
    "kNumMinhashFunctions must be divisible by kNumBands");
static_assert(kMinhashBitsPerHash == 1 || kMinhashBitsPerHash == 2 || kMinhashBitsPerHash == 4 ||
                  kMinhashBitsPerHash == 8,
    "kMinhashBitsPerHash must be 1, 2, 4, or 8");

} // namespace lssg::config
