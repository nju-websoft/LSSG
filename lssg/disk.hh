#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <type_traits>
#include <cstring>
#include <cstdint>

namespace lssg{

template<typename T>
static void WriteBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void ReadBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

// High-performance buffered serialization
class BufferedSerializer {
private:
  // Optimized buffer size for modern systems (256KB for better disk I/O)
  static constexpr size_t BUFFER_SIZE = 256 * 1024;
  static constexpr size_t ALIGNMENT = 64; // Cache line alignment
  
  alignas(ALIGNMENT) char buffer_[BUFFER_SIZE]; // Stack-allocated aligned buffer
  size_t buffer_pos_ = 0;
  std::ostream* os_;

  __attribute__((always_inline)) void flush() {
    if (buffer_pos_ > 0) {
      os_->write(buffer_, buffer_pos_);
      buffer_pos_ = 0;
    }
  }

  // Fast aligned memory copy for large data
  __attribute__((always_inline)) void fast_copy(void* dst, const void* src, size_t size) {
    if (size >= 64) {
      // Use system's optimized memcpy for large copies
      std::memcpy(dst, src, size);
    } else {
      // Unrolled loop for small copies
      const char* s = static_cast<const char*>(src);
      char* d = static_cast<char*>(dst);
      switch (size) {
        case 8: *reinterpret_cast<uint64_t*>(d) = *reinterpret_cast<const uint64_t*>(s); break;
        case 4: *reinterpret_cast<uint32_t*>(d) = *reinterpret_cast<const uint32_t*>(s); break;
        case 2: *reinterpret_cast<uint16_t*>(d) = *reinterpret_cast<const uint16_t*>(s); break;
        case 1: *d = *s; break;
        default: std::memcpy(d, s, size); break;
      }
    }
  }

public:
  explicit BufferedSerializer(std::ostream& os) : os_(&os) {}

  ~BufferedSerializer() {
    flush();
  }

  template<typename T>
  __attribute__((always_inline)) void write(const T& data) {
    constexpr size_t data_size = sizeof(T);
    
    // Fast path: data fits in buffer
    if (__builtin_expect(buffer_pos_ + data_size <= BUFFER_SIZE, 1)) {
      fast_copy(buffer_ + buffer_pos_, &data, data_size);
      buffer_pos_ += data_size;
    } else {
      // Slow path: need to flush
      flush();
      if constexpr (data_size >= BUFFER_SIZE / 2) {
        // Very large data - write directly
        os_->write(reinterpret_cast<const char*>(&data), data_size);
      } else {
        fast_copy(buffer_, &data, data_size);
        buffer_pos_ = data_size;
      }
    }
  }

  template<typename T>
  void write_array(const T* data, size_t count) {
    size_t total_size = count * sizeof(T);
    
    if (__builtin_expect(total_size > BUFFER_SIZE / 2, 0)) {
      // Large array - flush and write directly
      flush();
      os_->write(reinterpret_cast<const char*>(data), total_size);
    } else if (buffer_pos_ + total_size <= BUFFER_SIZE) {
      // Fast path: fits in current buffer
      fast_copy(buffer_ + buffer_pos_, data, total_size);
      buffer_pos_ += total_size;
    } else {
      // Slow path: flush and buffer
      flush();
      fast_copy(buffer_, data, total_size);
      buffer_pos_ = total_size;
    }
  }

  // Manual flush for critical sections
  void force_flush() { flush(); }

  // High-level serialization methods
  template<typename T>
  void serialize_attribute(const T& attr) {
    if constexpr (std::is_same_v<T, std::string>) {
      // String specialization
      size_t len = attr.length();
      write(len);
      if (len > 0) {
        write_raw_data(attr.data(), len);
      }
    } else if constexpr (std::is_same_v<T, std::vector<typename T::value_type>>) {
      // Vector specialization
      size_t size = attr.size();
      write(size);
      for (const auto& item : attr) {
        serialize_attribute(item);
      }
    } else {
      // POD types
      static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable or have specialization");
      write(attr);
    }
  }

  template<typename T>
  void serialize_attribute_vector(const std::vector<T>& attrs) {
    size_t size = attrs.size();
    write(size);
    
    if constexpr (std::is_trivially_copyable_v<T>) {
      // For POD types - batch write all at once
      if (!attrs.empty()) {
        write_raw_data(attrs.data(), size * sizeof(T));
      }
    } else {
      // For complex types - individual serialization
      for (const auto& attr : attrs) {
        serialize_attribute(attr);
      }
    }
  }

private:
  // Raw data writing for non-templated data
  void write_raw_data(const void* data, size_t size) {
    const char* char_data = static_cast<const char*>(data);
    
    if (__builtin_expect(size > BUFFER_SIZE / 2, 0)) {
      // Large data - flush and write directly
      flush();
      os_->write(char_data, size);
    } else if (buffer_pos_ + size <= BUFFER_SIZE) {
      // Fast path: fits in current buffer
      fast_copy(buffer_ + buffer_pos_, char_data, size);
      buffer_pos_ += size;
    } else {
      // Slow path: flush and buffer
      flush();
      fast_copy(buffer_, char_data, size);
      buffer_pos_ = size;
    }
  }
};

class BufferedDeserializer {
private:
  static constexpr size_t BUFFER_SIZE = 256 * 1024;
  static constexpr size_t ALIGNMENT = 64;
  static constexpr size_t PREFETCH_DISTANCE = 64;
  
  alignas(ALIGNMENT) char buffer_[BUFFER_SIZE]; // Stack-allocated aligned buffer
  size_t buffer_pos_ = 0;
  size_t buffer_end_ = 0;
  std::istream* is_;

  __attribute__((always_inline)) void fill_buffer() {
    buffer_pos_ = 0;
    is_->read(buffer_, BUFFER_SIZE);
    buffer_end_ = is_->gcount();
    
    // Prefetch first cache line of new data
    if (buffer_end_ > 0) {
      __builtin_prefetch(buffer_, 0, 3);
    }
  }

  // Fast aligned memory copy
  __attribute__((always_inline)) void fast_copy(void* dst, const void* src, size_t size) {
    // Prefetch source data if it's in our buffer and we're reading ahead
    if (size <= 64 && buffer_pos_ + size + PREFETCH_DISTANCE < buffer_end_) {
      __builtin_prefetch(buffer_ + buffer_pos_ + PREFETCH_DISTANCE, 0, 1);
    }
    
    if (size >= 64) {
      std::memcpy(dst, src, size);
    } else {
      const char* s = static_cast<const char*>(src);
      char* d = static_cast<char*>(dst);
      switch (size) {
        case 8: *reinterpret_cast<uint64_t*>(d) = *reinterpret_cast<const uint64_t*>(s); break;
        case 4: *reinterpret_cast<uint32_t*>(d) = *reinterpret_cast<const uint32_t*>(s); break;
        case 2: *reinterpret_cast<uint16_t*>(d) = *reinterpret_cast<const uint16_t*>(s); break;
        case 1: *d = *s; break;
        default: std::memcpy(d, s, size); break;
      }
    }
  }

public:
  explicit BufferedDeserializer(std::istream& is) : is_(&is) {
    fill_buffer();
  }

  template<typename T>
  __attribute__((always_inline)) void read(T& data) {
    constexpr size_t data_size = sizeof(T);
    
    // Fast path: data available in buffer
    if (__builtin_expect(buffer_pos_ + data_size <= buffer_end_, 1)) {
      fast_copy(&data, buffer_ + buffer_pos_, data_size);
      buffer_pos_ += data_size;
    } else {
      // Slow path: need more data
      if (__builtin_expect(buffer_end_ < BUFFER_SIZE, 0)) {
        // EOF or partial read - read directly from stream
        is_->read(reinterpret_cast<char*>(&data), data_size);
      } else {
        // Refill buffer and try again
        fill_buffer();
        if (buffer_pos_ + data_size <= buffer_end_) {
          fast_copy(&data, buffer_ + buffer_pos_, data_size);
          buffer_pos_ += data_size;
        } else {
          // Still not enough - read directly
          is_->read(reinterpret_cast<char*>(&data), data_size);
        }
      }
    }
  }

  template<typename T>
  void read_array(T* data, size_t count) {
    size_t total_size = count * sizeof(T);
    
    if (__builtin_expect(total_size > BUFFER_SIZE / 2, 0)) {
      // Large array - read directly from stream
      is_->read(reinterpret_cast<char*>(data), total_size);
    } else if (buffer_pos_ + total_size <= buffer_end_) {
      // Fast path: all data available in buffer
      fast_copy(data, buffer_ + buffer_pos_, total_size);
      buffer_pos_ += total_size;
    } else {
      // Slow path: need to refill buffer
      if (buffer_end_ < BUFFER_SIZE) {
        // EOF situation - read directly
        is_->read(reinterpret_cast<char*>(data), total_size);
      } else {
        fill_buffer();
        if (buffer_pos_ + total_size <= buffer_end_) {
          fast_copy(data, buffer_ + buffer_pos_, total_size);
          buffer_pos_ += total_size;
        } else {
          // Still not enough - read directly
          is_->read(reinterpret_cast<char*>(data), total_size);
        }
      }
    }
  }

  // Check if more data is available
  __attribute__((always_inline)) bool has_data() const {
    return buffer_pos_ < buffer_end_ || !is_->eof();
  }

  // High-level deserialization methods
  template<typename T>
  void deserialize_attribute(T& attr) {
    if constexpr (std::is_same_v<T, std::string>) {
      // String specialization
      size_t len;
      read(len);
      attr.resize(len);
      if (len > 0) {
        read_raw_data(&attr[0], len);
      }
    } else if constexpr (std::is_same_v<T, std::vector<typename T::value_type>>) {
      // Vector specialization  
      size_t size;
      read(size);
      attr.resize(size);
      for (auto& item : attr) {
        deserialize_attribute(item);
      }
    } else {
      // POD types
      static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable or have specialization");
      read(attr);
    }
  }

  template<typename T>
  void deserialize_attribute_vector(std::vector<T>& attrs) {
    size_t size;
    read(size);
    attrs.resize(size);
    
    if constexpr (std::is_trivially_copyable_v<T>) {
      // For POD types - batch read all at once
      if (size > 0) {
        read_raw_data(attrs.data(), size * sizeof(T));
      }
    } else {
      // For complex types - individual deserialization
      for (auto& attr : attrs) {
        deserialize_attribute(attr);
      }
    }
  }

private:
  // Raw data reading for non-templated data
  void read_raw_data(void* data, size_t size) {
    char* char_data = static_cast<char*>(data);
    
    if (__builtin_expect(size > BUFFER_SIZE / 2, 0)) {
      // Large data - read directly from stream
      is_->read(char_data, size);
    } else if (buffer_pos_ + size <= buffer_end_) {
      // Fast path: all data available in buffer
      fast_copy(char_data, buffer_ + buffer_pos_, size);
      buffer_pos_ += size;
    } else {
      // Slow path: need to refill buffer
      if (buffer_end_ < BUFFER_SIZE) {
        // EOF situation - read directly
        is_->read(char_data, size);
      } else {
        fill_buffer();
        if (buffer_pos_ + size <= buffer_end_) {
          fast_copy(char_data, buffer_ + buffer_pos_, size);
          buffer_pos_ += size;
        } else {
          // Still not enough - read directly
          is_->read(char_data, size);
        }
      }
    }
  }
};

} // namespace lssg