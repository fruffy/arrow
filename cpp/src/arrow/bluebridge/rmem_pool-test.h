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

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "arrow/bluebridge/rmem_pool.h"
#include "arrow/status.h"
#include "arrow/test-util.h"

namespace arrow {

class TestRMemPoolBase : public ::testing::Test {
 public:
  virtual ::arrow::RMemPool* rmem_pool() = 0;

  void TestRMemTracking() {
    auto pool = rmem_pool();

    uint8_t* data;
    uint64_t test1 = -1;
    uint64_t test2 = -1;
    ASSERT_OK(pool->Allocate(100, &data));
    pool->Read(data, 0, (uint8_t *) &test1, 1);
    EXPECT_EQ(static_cast<uint64_t>(0), reinterpret_cast<uint64_t>(test1) % 64);
    ASSERT_EQ(100, pool->bytes_allocated());

    uint8_t* data2;
    ASSERT_OK(pool->Allocate(27, &data2));
    pool->Read(data2, 0, (uint8_t *) &test2, 1);
    EXPECT_EQ(static_cast<uint64_t>(0), reinterpret_cast<uint64_t>(test2) % 64);
    ASSERT_EQ(127, pool->bytes_allocated());

    pool->Free(data, 100);
    ASSERT_EQ(27, pool->bytes_allocated());
    pool->Free(data2, 27);
    ASSERT_EQ(0, pool->bytes_allocated());
  }

  void TestOOM() {
    auto pool = rmem_pool();

    uint8_t* data;
    int64_t to_alloc = std::numeric_limits<int64_t>::max();
    ASSERT_RAISES(OutOfRMem, pool->Allocate(to_alloc, &data));
  }

  void TestReallocate() {
    auto pool = rmem_pool();

    uint8_t* data;
    ASSERT_OK(pool->Allocate(10, &data));
    ASSERT_EQ(10, pool->bytes_allocated());
    uint8_t test1 = 35;
    uint8_t test2 = 12;
    pool->Write(data, 0, (uint8_t *) &test1, 1);
    pool->Write(data, 9, (uint8_t *) &test2, 1);
    ASSERT_OK(pool->Reallocate(10, 20, &data));
    test1 = 0;
    test2 = 0;
    pool->Read(data, 9, (uint8_t *) &test2, 1);
    // Expand
    ASSERT_EQ(test2, 12);
    ASSERT_EQ(20, pool->bytes_allocated());
    // Shrink
    ASSERT_OK(pool->Reallocate(20, 5, &data));
    pool->Read(data, 0, (uint8_t *) &test1, 1);
    ASSERT_EQ(test1, 35);
    ASSERT_EQ(5, pool->bytes_allocated());

    // Free
    pool->Free(data, 5);
    ASSERT_EQ(0, pool->bytes_allocated());
  }
};

}  // namespace arrow
