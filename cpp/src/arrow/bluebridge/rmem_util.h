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

#ifndef ARROW_UTIL_MEMORY_H
#define ARROW_UTIL_MEMORY_H

#include <thread>
#include <vector>

#include "arrow/util/thread-pool.h"

namespace arrow {
namespace internal {

uint8_t* ptr_logical_and(const uint8_t* address, uintptr_t bits) {
  uintptr_t value = reinterpret_cast<uintptr_t>(address);
  return reinterpret_cast<uint8_t*>(value & bits);
}

// A helper function for doing memcpy with multiple threads. This is required
// to saturate the memory bandwidth of modern cpus.
void parallel_rmemcopy(uint8_t* dst, const uint8_t* src, int64_t nbytes,
                      uintptr_t block_size, int num_threads) {
  // XXX This function is really using `num_threads + 1` threads.
  auto pool = GetCpuThreadPool();

  uint8_t* left = ptr_logical_and(src + block_size - 1, ~(block_size - 1));
  uint8_t* right = ptr_logical_and(src + nbytes, ~(block_size - 1));
  int64_t num_blocks = (right - left) / block_size;

  // Update right address
  right = right - (num_blocks % num_threads) * block_size;

  // Now we divide these blocks between available threads. The remainder is
  // handled separately.
  int64_t chunk_size = (right - left) / num_threads;
  int64_t prefix = left - src;
  int64_t suffix = src + nbytes - right;
  // Now the data layout is | prefix | k * num_threads * block_size | suffix |.
  // We have chunk_size = k * block_size, therefore the data layout is
  // | prefix | num_threads * chunk_size | suffix |.
  // Each thread gets a "chunk" of k blocks.

  // Start all parallel memcpy tasks and handle leftovers while threads run.
  std::vector<std::future<void*>> futures;

  for (int i = 0; i < num_threads; i++) {
    futures.emplace_back(pool->Submit(memcpy, dst + prefix + i * chunk_size,
                                      left + i * chunk_size, chunk_size));
  }
  memcpy(dst, src, prefix);
  memcpy(dst + prefix + num_threads * chunk_size, right, suffix);

  for (auto& fut : futures) {
    fut.get();
  }
}

}  // namespace internal
}  // namespace arrow

#endif  // ARROW_UTIL_MEMORY_H
