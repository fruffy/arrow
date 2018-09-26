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

#ifndef ARROW_RMEM_POOL_H
#define ARROW_RMEM_POOL_H

#include <cstdint>
#include <memory>

#include "arrow/util/visibility.h"

namespace arrow {

class Status;

/// Base class for rmem allocation.
///
/// Besides tracking the number of allocated bytes, the allocator also should
/// take care of the required 64-byte alignment.
class ARROW_EXPORT RMemPool {
 public:
  virtual ~RMemPool();

  /// Allocate a new rmem region of at least size bytes.
  ///
  /// The allocated region shall be 64-byte aligned.
  virtual Status Allocate(int64_t size, uint8_t** out) = 0;

  /// Resize an already allocated rmem section.
  ///
  /// As by default most default allocators on a platform don't support aligned
  /// reallocation, this function can involve a copy of the underlying data.
  virtual Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) = 0;

  /// Free an allocated region.
  ///
  /// @param buffer Pointer to the start of the allocated rmem region
  /// @param size Allocated size located at buffer. An allocator implementation
  ///   may use this for tracking the amount of allocated bytes as well as for
  ///   faster deallocation if supported by its backend.
  virtual void Free(uint8_t* buffer, int64_t size) = 0;

  /// The number of bytes that were allocated and not yet free'd through
  /// this allocator.
  virtual int64_t bytes_allocated() const = 0;

  /// Return peak rmem allocation in this rmem pool
  ///
  /// \return Maximum bytes allocated. If not known (or not implemented),
  /// returns -1
  virtual int64_t max_rmem() const;

 protected:
  RMemPool();
};

class ARROW_EXPORT LoggingRMemPool : public RMemPool {
 public:
  explicit LoggingRMemPool(RMemPool* pool);
  ~LoggingRMemPool() override = default;

  Status Allocate(int64_t size, uint8_t** out) override;
  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) override;

  void Free(uint8_t* buffer, int64_t size) override;

  int64_t bytes_allocated() const override;

  int64_t max_rmem() const override;

 private:
  RMemPool* pool_;
};

/// Derived class for rmem allocation.
///
/// Tracks the number of bytes and maximum rmem allocated through its direct
/// calls. Actual allocation is delegated to RMemPool class.
class ARROW_EXPORT ProxyRMemPool : public RMemPool {
 public:
  explicit ProxyRMemPool(RMemPool* pool);
  ~ProxyRMemPool() override;

  Status Allocate(int64_t size, uint8_t** out) override;
  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) override;

  void Free(uint8_t* buffer, int64_t size) override;

  int64_t bytes_allocated() const override;

  int64_t max_rmem() const override;

 private:
  class ProxyRMemPoolImpl;
  std::unique_ptr<ProxyRMemPoolImpl> impl_;
};

ARROW_EXPORT RMemPool* default_rmem_pool();

#ifdef ARROW_NO_DEFAULT_RMEM_POOL
#define ARROW_RMEM_POOL_DEFAULT
#else
#define ARROW_RMEM_POOL_DEFAULT = default_rmem_pool()
#endif

}  // namespace arrow

#endif  // ARROW_RMEM_POOL_H
