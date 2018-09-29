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

#include "arrow/bluebridge/rmem_buffer.h"

#include <cstdint>
#include <utility>

#include "arrow/bluebridge/rmem_pool.h"
#include "arrow/status.h"
#include "arrow/util/bit-util.h"
#include "arrow/util/logging.h"

namespace arrow {

Status RMemBuffer::Copy(const int64_t start, const int64_t nbytes, RMemPool* pool,
                    std::shared_ptr<RMemBuffer>* out) const {
  // Sanity checks
  DCHECK_LT(start, size_);
  DCHECK_LE(nbytes, size_ - start);

  std::shared_ptr<ResizableRMemBuffer> new_buffer;
  RETURN_NOT_OK(AllocateResizableRMemBuffer(pool, nbytes, &new_buffer));

  std::memcpy(new_buffer->mutable_data(), data() + start, static_cast<size_t>(nbytes));

  *out = new_buffer;
  return Status::OK();
}

Status RMemBuffer::Copy(const int64_t start, const int64_t nbytes,
                    std::shared_ptr<RMemBuffer>* out) const {
  return Copy(start, nbytes, default_rmem_pool(), out);
}

bool RMemBuffer::Equals(const RMemBuffer& other, const int64_t nbytes) const {
  return this == &other || (size_ >= nbytes && other.size_ >= nbytes &&
                            (data_ == other.data_ ||
                             !memcmp(data_, other.data_, static_cast<size_t>(nbytes))));
}

bool RMemBuffer::Equals(const RMemBuffer& other) const {
  return this == &other || (size_ == other.size_ &&
                            (data_ == other.data_ ||
                             !memcmp(data_, other.data_, static_cast<size_t>(size_))));
}

Status RMemBuffer::FromString(const std::string& data, RMemPool* pool,
                          std::shared_ptr<RMemBuffer>* out) {
  auto size = static_cast<int64_t>(data.size());
  RETURN_NOT_OK(AllocateRMemBuffer(pool, size, out));
  std::copy(data.c_str(), data.c_str() + size, (*out)->mutable_data());
  return Status::OK();
}

Status RMemBuffer::FromString(const std::string& data, std::shared_ptr<RMemBuffer>* out) {
  return FromString(data, default_rmem_pool(), out);
}

std::string RMemBuffer::ToString() const {
  return std::string(reinterpret_cast<const char*>(data_), static_cast<size_t>(size_));
}

void RMemBuffer::CheckMutable() const { DCHECK(is_mutable()) << "buffer not mutable"; }

/// A RMemBuffer whose lifetime is tied to a particular RMemPool
class PoolRMemBuffer : public ResizableRMemBuffer {
 public:
  explicit PoolRMemBuffer(RMemPool* pool) : ResizableRMemBuffer(nullptr, 0) {
    if (pool == nullptr) {
      pool = default_rmem_pool();
    }
    pool_ = pool;
  }

  ~PoolRMemBuffer() override {
    if (mutable_data_ != nullptr) {
      pool_->Free(mutable_data_, capacity_);
    }
  }

  Status Reserve(const int64_t capacity) override {
    if (!mutable_data_ || capacity > capacity_) {
      uint8_t* new_data;
      int64_t new_capacity = BitUtil::RoundUpToMultipleOf64(capacity);
      if (mutable_data_) {
        RETURN_NOT_OK(pool_->Reallocate(capacity_, new_capacity, &mutable_data_));
      } else {
        RETURN_NOT_OK(pool_->Allocate(new_capacity, &new_data));
        mutable_data_ = new_data;
      }
      data_ = mutable_data_;
      capacity_ = new_capacity;
    }
    return Status::OK();
  }

  Status Resize(const int64_t new_size, bool shrink_to_fit = true) override {
    if (!shrink_to_fit || (new_size > size_)) {
      RETURN_NOT_OK(Reserve(new_size));
    } else {
      // RMemBuffer is not growing, so shrink to the requested size without
      // excess space.
      int64_t new_capacity = BitUtil::RoundUpToMultipleOf64(new_size);
      if (capacity_ != new_capacity) {
        // RMemBuffer hasn't got yet the requested size.
        if (new_size == 0) {
          pool_->Free(mutable_data_, capacity_);
          capacity_ = 0;
          mutable_data_ = nullptr;
          data_ = nullptr;
        } else {
          RETURN_NOT_OK(pool_->Reallocate(capacity_, new_capacity, &mutable_data_));
          data_ = mutable_data_;
          capacity_ = new_capacity;
        }
      }
    }
    size_ = new_size;

    return Status::OK();
  }

 private:
  RMemPool* pool_;
};

std::shared_ptr<RMemBuffer> SliceMutableRMemBuffer(const std::shared_ptr<RMemBuffer>& buffer,
                                           const int64_t offset, const int64_t length) {
  return std::make_shared<MutableRMemBuffer>(buffer, offset, length);
}

MutableRMemBuffer::MutableRMemBuffer(const std::shared_ptr<RMemBuffer>& parent, const int64_t offset,
                             const int64_t size)
    : MutableRMemBuffer(parent->mutable_data() + offset, size) {
  DCHECK(parent->is_mutable()) << "Must pass mutable buffer";
  parent_ = parent;
}

namespace {
// A utility that does most of the work of the `AllocateRMemBuffer` and
// `AllocateResizableRMemBuffer` methods. The argument `buffer` should be a smart pointer to a
// PoolRMemBuffer.
template <typename PoolRMemBufferPtr, typename RMemBufferPtr>
inline Status ResizePoolRMemBuffer(PoolRMemBufferPtr&& buffer, const int64_t size,
                               RMemBufferPtr* out) {
  RETURN_NOT_OK(buffer->Resize(size));
  buffer->ZeroPadding();
  *out = std::move(buffer);
  return Status::OK();
}
}  // namespace

Status AllocateRMemBuffer(RMemPool* pool, const int64_t size,
                      std::shared_ptr<RMemBuffer>* out) {
  return ResizePoolRMemBuffer(std::make_shared<PoolRMemBuffer>(pool), size, out);
}

Status AllocateRMemBuffer(RMemPool* pool, const int64_t size,
                      std::unique_ptr<RMemBuffer>* out) {
  return ResizePoolRMemBuffer(std::unique_ptr<PoolRMemBuffer>(new PoolRMemBuffer(pool)), size, out);
}

Status AllocateRMemBuffer(const int64_t size, std::shared_ptr<RMemBuffer>* out) {
  return AllocateRMemBuffer(default_rmem_pool(), size, out);
}

Status AllocateRMemBuffer(const int64_t size, std::unique_ptr<RMemBuffer>* out) {
  return AllocateRMemBuffer(default_rmem_pool(), size, out);
}

Status AllocateResizableRMemBuffer(RMemPool* pool, const int64_t size,
                               std::shared_ptr<ResizableRMemBuffer>* out) {
  return ResizePoolRMemBuffer(std::make_shared<PoolRMemBuffer>(pool), size, out);
}

Status AllocateResizableRMemBuffer(RMemPool* pool, const int64_t size,
                               std::unique_ptr<ResizableRMemBuffer>* out) {
  return ResizePoolRMemBuffer(std::unique_ptr<PoolRMemBuffer>(new PoolRMemBuffer(pool)), size, out);
}

Status AllocateResizableRMemBuffer(const int64_t size,
                               std::shared_ptr<ResizableRMemBuffer>* out) {
  return AllocateResizableRMemBuffer(default_rmem_pool(), size, out);
}

Status AllocateResizableRMemBuffer(const int64_t size,
                               std::unique_ptr<ResizableRMemBuffer>* out) {
  return AllocateResizableRMemBuffer(default_rmem_pool(), size, out);
}

Status AllocateEmptyBitmap(RMemPool* pool, int64_t length,
                           std::shared_ptr<RMemBuffer>* out) {
  RETURN_NOT_OK(AllocateRMemBuffer(pool, BitUtil::BytesForBits(length), out));
  memset((*out)->mutable_data(), 0, static_cast<size_t>((*out)->size()));
  return Status::OK();
}

Status AllocateEmptyBitmap(int64_t length, std::shared_ptr<RMemBuffer>* out) {
  return AllocateEmptyBitmap(default_rmem_pool(), length, out);
}

}  // namespace arrow
