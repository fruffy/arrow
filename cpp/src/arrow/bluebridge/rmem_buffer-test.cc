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
#include <cstring>
#include <limits>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "arrow/bluebridge/rmem_buffer.h"
#include "arrow/bluebridge/rmem_pool.h"
#include "arrow/status.h"
#include "arrow/buffer.h"

using std::string;

namespace arrow {

#define STRINGIFY(x) #x

#define ASSERT_RAISES(ENUM, expr)                                         \
  do {                                                                    \
    ::arrow::Status s = (expr);                                           \
    if (!s.Is##ENUM()) {                                                  \
      FAIL() << "Expected '" STRINGIFY(expr) "' to fail with " STRINGIFY( \
                    ENUM) ", but got "                                    \
             << s.ToString();                                             \
    }                                                                     \
  } while (false)

#define ASSERT_OK(expr)                                               \
  do {                                                                \
    ::arrow::Status s = (expr);                                       \
    if (!s.ok()) {                                                    \
      FAIL() << "'" STRINGIFY(expr) "' failed with " << s.ToString(); \
    }                                                                 \
  } while (false)

#define ASSERT_OK_NO_THROW(expr) ASSERT_NO_THROW(ASSERT_OK(expr))

#define EXPECT_OK(expr)         \
  do {                          \
    ::arrow::Status s = (expr); \
    EXPECT_TRUE(s.ok());        \
  } while (false)

#define ABORT_NOT_OK(s)                  \
  do {                                   \
    ::arrow::Status _s = (s);            \
    if (ARROW_PREDICT_FALSE(!_s.ok())) { \
      std::cerr << s.ToString() << "\n"; \
      std::abort();                      \
    }                                    \
  } while (false);


TEST(TestRMemBuffer, FromStdString) {
  std::string val = "hello, world";

  RMemBuffer buf(val);
  ASSERT_EQ(0, memcmp(buf.data(), val.c_str(), val.size()));
  ASSERT_EQ(static_cast<int64_t>(val.size()), buf.size());
}

TEST(TestRMemBuffer, FromStdStringWithRMem) {
  std::string expected = "hello, world";
  std::shared_ptr<RMemBuffer> buf;

  {
    std::string temp = "hello, world";
    ASSERT_OK(RMemBuffer::FromString(temp, &buf));
    ASSERT_EQ(0, memcmp(buf->data(), temp.c_str(), temp.size()));
    ASSERT_EQ(static_cast<int64_t>(temp.size()), buf->size());
  }

  // Now temp goes out of scope and we check if created buffer
  // is still valid to make sure it actually owns its space
  ASSERT_EQ(0, memcmp(buf->data(), expected.c_str(), expected.size()));
  ASSERT_EQ(static_cast<int64_t>(expected.size()), buf->size());
}

TEST(TestRMemBuffer, EqualsWithSameContent) {
  RMemPool* pool = default_rmem_pool();
  const int32_t bufferSize = 128 * 1024;
  uint8_t* rawRMemBuffer1;
  ASSERT_OK(pool->Allocate(bufferSize, &rawRMemBuffer1));
  memset(rawRMemBuffer1, 12, bufferSize);
  uint8_t* rawRMemBuffer2;
  ASSERT_OK(pool->Allocate(bufferSize, &rawRMemBuffer2));
  memset(rawRMemBuffer2, 12, bufferSize);
  uint8_t* rawRMemBuffer3;
  ASSERT_OK(pool->Allocate(bufferSize, &rawRMemBuffer3));
  memset(rawRMemBuffer3, 3, bufferSize);

  RMemBuffer buffer1(rawRMemBuffer1, bufferSize);
  RMemBuffer buffer2(rawRMemBuffer2, bufferSize);
  RMemBuffer buffer3(rawRMemBuffer3, bufferSize);
  ASSERT_TRUE(buffer1.Equals(buffer2));
  ASSERT_FALSE(buffer1.Equals(buffer3));

  pool->Free(rawRMemBuffer1, bufferSize);
  pool->Free(rawRMemBuffer2, bufferSize);
  pool->Free(rawRMemBuffer3, bufferSize);
}

TEST(TestRMemBuffer, EqualsWithSameRMemBuffer) {
  RMemPool* pool = default_rmem_pool();
  const int32_t bufferSize = 128 * 1024;
  uint8_t* rawRMemBuffer;
  ASSERT_OK(pool->Allocate(bufferSize, &rawRMemBuffer));
  memset(rawRMemBuffer, 111, bufferSize);

  RMemBuffer buffer1(rawRMemBuffer, bufferSize);
  RMemBuffer buffer2(rawRMemBuffer, bufferSize);
  ASSERT_TRUE(buffer1.Equals(buffer2));

  const int64_t nbytes = bufferSize / 2;
  RMemBuffer buffer3(rawRMemBuffer, nbytes);
  ASSERT_TRUE(buffer1.Equals(buffer3, nbytes));
  ASSERT_FALSE(buffer1.Equals(buffer3, nbytes + 1));

  pool->Free(rawRMemBuffer, bufferSize);
}

TEST(TestRMemBuffer, Copy) {
  std::string data_str = "some data to copy";

  auto data = reinterpret_cast<const uint8_t*>(data_str.c_str());

  RMemBuffer buf(data, data_str.size());

  std::shared_ptr<RMemBuffer> out;

  ASSERT_OK(buf.Copy(5, 4, &out));

  RMemBuffer expected(data + 5, 4);
  ASSERT_TRUE(out->Equals(expected));
  // assert the padding is zeroed
  std::vector<uint8_t> zeros(out->capacity() - out->size());
  ASSERT_EQ(0, memcmp(out->data() + out->size(), zeros.data(), zeros.size()));
}

TEST(TestRMemBuffer, SliceRMemBuffer) {
  std::string data_str = "some data to slice";

  auto data = reinterpret_cast<const uint8_t*>(data_str.c_str());

  auto buf = std::make_shared<RMemBuffer>(data, data_str.size());

  std::shared_ptr<RMemBuffer> out = SliceRMemBuffer(buf, 5, 4);
  RMemBuffer expected(data + 5, 4);
  ASSERT_TRUE(out->Equals(expected));

  ASSERT_EQ(2, buf.use_count());
}

TEST(TestMutableRMemBuffer, Wrap) {
  std::vector<int32_t> values = {1, 2, 3};

  auto buf = MutableRMemBuffer::Wrap(values.data(), values.size());
  reinterpret_cast<int32_t*>(buf->mutable_data())[1] = 4;

  ASSERT_EQ(4, values[1]);
}

TEST(TestRMemBuffer, SliceMutableRMemBuffer) {
  std::string data_str = "some data to slice";
  auto data = reinterpret_cast<const uint8_t*>(data_str.c_str());

  std::shared_ptr<RMemBuffer> buffer;
  ASSERT_OK(AllocateRMemBuffer(50, &buffer));

  memcpy(buffer->mutable_data(), data, data_str.size());

  std::shared_ptr<RMemBuffer> slice = SliceMutableRMemBuffer(buffer, 5, 10);
  ASSERT_TRUE(slice->is_mutable());
  ASSERT_EQ(10, slice->size());

  RMemBuffer expected(data + 5, 10);
  ASSERT_TRUE(slice->Equals(expected));
}

TEST(TestRMemBufferBuilder, ResizeReserve) {
  const std::string data = "some data";
  auto data_ptr = data.c_str();

  RMemBufferBuilder builder;

  ASSERT_OK(builder.Append(data_ptr, 9));
  ASSERT_EQ(9, builder.length());

  ASSERT_OK(builder.Resize(128));
  ASSERT_EQ(128, builder.capacity());

  // Do not shrink to fit
  ASSERT_OK(builder.Resize(64, false));
  ASSERT_EQ(128, builder.capacity());

  // Shrink to fit
  ASSERT_OK(builder.Resize(64));
  ASSERT_EQ(64, builder.capacity());

  // Reserve elements
  ASSERT_OK(builder.Reserve(60));
  ASSERT_EQ(128, builder.capacity());
}

template <typename T>
class TypedTestRMemBuffer : public ::testing::Test {};

using RMemBufferPtrs =
    ::testing::Types<std::shared_ptr<ResizableRMemBuffer>, std::unique_ptr<ResizableRMemBuffer>>;

TYPED_TEST_CASE(TypedTestRMemBuffer, RMemBufferPtrs);

TYPED_TEST(TypedTestRMemBuffer, IsMutableFlag) {
  RMemBuffer buf(nullptr, 0);

  ASSERT_FALSE(buf.is_mutable());

  MutableRMemBuffer mbuf(nullptr, 0);
  ASSERT_TRUE(mbuf.is_mutable());

  TypeParam pool_buf;
  ASSERT_OK(AllocateResizableRMemBuffer(0, &pool_buf));
  ASSERT_TRUE(pool_buf->is_mutable());
}

TYPED_TEST(TypedTestRMemBuffer, Resize) {
  TypeParam buf;
  ASSERT_OK(AllocateResizableRMemBuffer(0, &buf));

  ASSERT_EQ(0, buf->size());
  ASSERT_OK(buf->Resize(100));
  ASSERT_EQ(100, buf->size());
  ASSERT_OK(buf->Resize(200));
  ASSERT_EQ(200, buf->size());

  // Make it smaller, too
  ASSERT_OK(buf->Resize(50, true));
  ASSERT_EQ(50, buf->size());
  // We have actually shrunken in size
  // The spec requires that capacity is a multiple of 64
  ASSERT_EQ(64, buf->capacity());

  // Resize to a larger capacity again to test shrink_to_fit = false
  ASSERT_OK(buf->Resize(100));
  ASSERT_EQ(128, buf->capacity());
  ASSERT_OK(buf->Resize(50, false));
  ASSERT_EQ(128, buf->capacity());
}

TYPED_TEST(TypedTestRMemBuffer, TypedResize) {
  TypeParam buf;
  ASSERT_OK(AllocateResizableRMemBuffer(0, &buf));

  ASSERT_EQ(0, buf->size());
  ASSERT_OK(buf->template TypedResize<double>(100));
  ASSERT_EQ(800, buf->size());
  ASSERT_OK(buf->template TypedResize<double>(200));
  ASSERT_EQ(1600, buf->size());

  ASSERT_OK(buf->template TypedResize<double>(50, true));
  ASSERT_EQ(400, buf->size());
  ASSERT_EQ(448, buf->capacity());

  ASSERT_OK(buf->template TypedResize<double>(100));
  ASSERT_EQ(832, buf->capacity());
  ASSERT_OK(buf->template TypedResize<double>(50, false));
  ASSERT_EQ(832, buf->capacity());
}

TYPED_TEST(TypedTestRMemBuffer, ResizeOOM) {
// This test doesn't play nice with AddressSanitizer
#ifndef ADDRESS_SANITIZER
  // realloc fails, even though there may be no explicit limit
  TypeParam buf;
  ASSERT_OK(AllocateResizableRMemBuffer(0, &buf));
  ASSERT_OK(buf->Resize(100));
  int64_t to_alloc = std::numeric_limits<int64_t>::max();
  ASSERT_RAISES(OutOfRMem, buf->Resize(to_alloc));
#endif
}

}  // namespace arrow
