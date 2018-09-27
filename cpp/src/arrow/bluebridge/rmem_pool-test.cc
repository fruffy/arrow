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

#include <gtest/gtest.h>

#include "arrow/bluebridge/rmem_pool-test.h"
#include "arrow/bluebridge/rmem_pool.h"
#include "arrow/status.h"
extern "C" {
#include "client_lib.h"
}
namespace arrow {

class TestDefaultRMemPool : public ::arrow::TestRMemPoolBase {
 public:
  ::arrow::RMemPool* rmem_pool() override { return ::arrow::default_rmem_pool(); }
};

TEST_F(TestDefaultRMemPool, RMemTracking) { this->TestRMemTracking(); }

TEST_F(TestDefaultRMemPool, OOM) {
#ifndef ADDRESS_SANITIZER
  this->TestOOM();
#endif
}

TEST_F(TestDefaultRMemPool, Reallocate) { this->TestReallocate(); }

// Death tests and valgrind are known to not play well 100% of the time. See
// googletest documentation
#if !(defined(ARROW_VALGRIND) || defined(ADDRESS_SANITIZER))

TEST(DefaultRMemPoolDeathTest, FreeLargeRMem) {
  RMemPool* pool = default_rmem_pool();

  uint8_t* data;
  ASSERT_OK(pool->Allocate(100, &data));

#ifndef NDEBUG
  EXPECT_DEATH(pool->Free(data, 120),
               ".*Check failed:.* allocation counter became negative");
#endif

  pool->Free(data, 100);
}

TEST(DefaultRMemPoolDeathTest, MaxRMem) {
  RMemPool* pool = default_rmem_pool();
  uint8_t* data1;
  uint8_t* data2;

  ASSERT_OK(pool->Allocate(100, &data1));
  ASSERT_OK(pool->Allocate(50, &data2));
  pool->Free(data2, 50);
  ASSERT_OK(pool->Allocate(100, &data2));
  pool->Free(data1, 100);
  pool->Free(data2, 100);

  ASSERT_EQ(200, pool->max_rmem());
}

#endif  // ARROW_VALGRIND

TEST(LoggingRMemPool, Logging) {
  RMemPool* pool = default_rmem_pool();

  LoggingRMemPool lp(pool);

  uint8_t* data;
  ASSERT_OK(pool->Allocate(100, &data));

  uint8_t* data2;
  ASSERT_OK(pool->Allocate(100, &data2));

  pool->Free(data, 100);
  pool->Free(data2, 100);

  ASSERT_EQ(200, pool->max_rmem());
}

TEST(ProxyRMemPool, Logging) {
  RMemPool* pool = default_rmem_pool();

  ProxyRMemPool pp(pool);

  uint8_t* data;
  ASSERT_OK(pool->Allocate(100, &data));

  uint8_t* data2;
  ASSERT_OK(pp.Allocate(300, &data2));

  ASSERT_EQ(400, pool->bytes_allocated());
  ASSERT_EQ(300, pp.bytes_allocated());

  pool->Free(data, 100);
  pp.Free(data2, 300);

  ASSERT_EQ(0, pool->bytes_allocated());
  ASSERT_EQ(0, pp.bytes_allocated());
}
}  // namespace arrow
