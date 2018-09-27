// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/bluebridge/rmem_pool.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>  // IWYU pragma: keep

#include "arrow/status.h"
#include "arrow/util/logging.h"
extern "C" {
  #include "utils.h"
}



#ifdef ARROW_JEMALLOC
// Needed to support jemalloc 3 and 4
#define JEMALLOC_MANGLE
// Explicitly link to our version of jemalloc
#include "jemalloc_ep/dist/include/jemalloc/jemalloc.h"
#endif

namespace arrow {

constexpr size_t kAlignment = 64;

namespace {
// Allocate rmem according to the alignment requirements for Arrow
// (as of May 2016 64 bytes)
Status AllocateAligned(int64_t size, uint8_t** out) {
// TODO(emkornfield) find something compatible with windows
#ifdef _WIN32
  // Special code path for Windows
  *out =
      reinterpret_cast<uint8_t*>(_aligned_malloc(static_cast<size_t>(size), kAlignment));
  if (!*out) {
    std::stringstream ss;
    ss << "malloc of size " << size << " failed";
    return Status::OutOfRMem(ss.str());
  }
#elif defined(ARROW_JEMALLOC)
  *out = reinterpret_cast<uint8_t*>(mallocx(
      std::max(static_cast<size_t>(size), kAlignment), MALLOCX_ALIGN(kAlignment)));
  if (*out == NULL) {
    std::stringstream ss;
    ss << "malloc of size " << size << " failed";
    return Status::OutOfRMem(ss.str());
  }
#else
  const int result = posix_memalign(reinterpret_cast<void**>(out), kAlignment,
                                    static_cast<size_t>(size));
  if (result == ENOMEM) {
    std::stringstream ss;
    ss << "malloc of size " << size << " failed";
    return Status::OutOfRMem(ss.str());
  }

  if (result == EINVAL) {
    std::stringstream ss;
    ss << "invalid alignment parameter: " << kAlignment;
    return Status::Invalid(ss.str());
  }
#endif
  return Status::OK();
}
}  // namespace

RMemPool::RMemPool() {}

RMemPool::~RMemPool() {}

int64_t RMemPool::max_rmem() const { return -1; }

///////////////////////////////////////////////////////////////////////
// Helper tracking rmem statistics

class RMemPoolStats {
 public:
  RMemPoolStats() : bytes_allocated_(0), max_rmem_(0) {}

  int64_t max_rmem() const { return max_rmem_.load(); }

  int64_t bytes_allocated() const { return bytes_allocated_.load(); }

  inline void UpdateAllocatedBytes(int64_t diff) {
    auto allocated = bytes_allocated_.fetch_add(diff) + diff;
    DCHECK_GE(allocated, 0) << "allocation counter became negative";
    // "maximum" allocated rmem is ill-defined in multi-threaded code,
    // so don't try to be too rigorous here
    if (diff > 0 && allocated > max_rmem_) {
      max_rmem_ = allocated;
    }
  }

 protected:
  std::atomic<int64_t> bytes_allocated_;
  std::atomic<int64_t> max_rmem_;
};

///////////////////////////////////////////////////////////////////////
// Default RMemPool implementation

class DefaultRMemPool : public RMemPool {
 public:
  ~DefaultRMemPool() override {}

  Status Allocate(int64_t size, uint8_t** out) override {
    RETURN_NOT_OK(AllocateAligned(size, out));

    stats_.UpdateAllocatedBytes(size);
    return Status::OK();
  }

  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) override {
#ifdef ARROW_JEMALLOC
    uint8_t* previous_ptr = *ptr;
    *ptr = reinterpret_cast<uint8_t*>(rallocx(*ptr, new_size, MALLOCX_ALIGN(kAlignment)));
    if (*ptr == NULL) {
      std::stringstream ss;
      ss << "realloc of size " << new_size << " failed";
      *ptr = previous_ptr;
      return Status::OutOfRMem(ss.str());
    }
#else
    // Note: We cannot use realloc() here as it doesn't guarantee alignment.

    // Allocate new chunk
    uint8_t* out = nullptr;
    RETURN_NOT_OK(AllocateAligned(new_size, &out));
    DCHECK(out);
    // Copy contents and release old rmem chunk
    memcpy(out, *ptr, static_cast<size_t>(std::min(new_size, old_size)));
#ifdef _MSC_VER
    _aligned_free(*ptr);
#else
    std::free(*ptr);
#endif  // defined(_MSC_VER)
    *ptr = out;
#endif  // defined(ARROW_JEMALLOC)

    stats_.UpdateAllocatedBytes(new_size - old_size);
    return Status::OK();
  }

  int64_t bytes_allocated() const override { return stats_.bytes_allocated(); }

  void Free(uint8_t* buffer, int64_t size) override {
#ifdef _MSC_VER
    _aligned_free(buffer);
#elif defined(ARROW_JEMALLOC)
    dallocx(buffer, MALLOCX_ALIGN(kAlignment));
#else
    std::free(buffer);
#endif
    stats_.UpdateAllocatedBytes(-size);
  }

  int64_t max_rmem() const override { return stats_.max_rmem(); }

 private:
  RMemPoolStats stats_;
};

RMemPool* default_rmem_pool() {
  static DefaultRMemPool default_rmem_pool_;
  return &default_rmem_pool_;
}

///////////////////////////////////////////////////////////////////////
// LoggingRMemPool implementation

LoggingRMemPool::LoggingRMemPool(RMemPool* pool) : pool_(pool) {}

Status LoggingRMemPool::Allocate(int64_t size, uint8_t** out) {
  Status s = pool_->Allocate(size, out);
  std::cout << "Allocate: size = " << size << std::endl;
  return s;
}

Status LoggingRMemPool::Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) {
  Status s = pool_->Reallocate(old_size, new_size, ptr);
  std::cout << "Reallocate: old_size = " << old_size << " - new_size = " << new_size
            << std::endl;
  return s;
}

void LoggingRMemPool::Free(uint8_t* buffer, int64_t size) {
  pool_->Free(buffer, size);
  std::cout << "Free: size = " << size << std::endl;
}

int64_t LoggingRMemPool::bytes_allocated() const {
  int64_t nb_bytes = pool_->bytes_allocated();
  std::cout << "bytes_allocated: " << nb_bytes << std::endl;
  return nb_bytes;
}

int64_t LoggingRMemPool::max_rmem() const {
  int64_t mem = pool_->max_rmem();
  std::cout << "max_rmem: " << mem << std::endl;
  return mem;
}

///////////////////////////////////////////////////////////////////////
// ProxyRMemPool implementation

class ProxyRMemPool::ProxyRMemPoolImpl {
 public:
  explicit ProxyRMemPoolImpl(RMemPool* pool) : pool_(pool) {}

  Status Allocate(int64_t size, uint8_t** out) {
    RETURN_NOT_OK(pool_->Allocate(size, out));
    stats_.UpdateAllocatedBytes(size);
    return Status::OK();
  }

  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) {
    RETURN_NOT_OK(pool_->Reallocate(old_size, new_size, ptr));
    stats_.UpdateAllocatedBytes(new_size - old_size);
    return Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size) {
    pool_->Free(buffer, size);
    stats_.UpdateAllocatedBytes(-size);
  }

  int64_t bytes_allocated() const { return stats_.bytes_allocated(); }

  int64_t max_rmem() const { return stats_.max_rmem(); }

 private:
  RMemPool* pool_;
  RMemPoolStats stats_;
};

ProxyRMemPool::ProxyRMemPool(RMemPool* pool) {
  impl_.reset(new ProxyRMemPoolImpl(pool));
}

ProxyRMemPool::~ProxyRMemPool() {}

Status ProxyRMemPool::Allocate(int64_t size, uint8_t** out) {
  return impl_->Allocate(size, out);
}

Status ProxyRMemPool::Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) {
  return impl_->Reallocate(old_size, new_size, ptr);
}

void ProxyRMemPool::Free(uint8_t* buffer, int64_t size) {
  return impl_->Free(buffer, size);
}

int64_t ProxyRMemPool::bytes_allocated() const { return impl_->bytes_allocated(); }

int64_t ProxyRMemPool::max_rmem() const { return impl_->max_rmem(); }

}  // namespace arrow
