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
#include <arpa/inet.h>  // IWYU pragma: keep
#include "arrow/status.h"
#include "arrow/util/logging.h"
extern "C" {
  #include "utils.h"
  #include "client_lib.h"
}
#include "khash.h"
KHASH_MAP_INIT_INT64(ptr_ip6_map, ip6_memaddr_block)
#define BB_PORT 5000

namespace arrow {

constexpr size_t kAlignment = 64;

namespace {
// Allocate rmem according to the alignment requirements for Arrow
// (as of May 2016 64 bytes)
Status AllocateRMem(int64_t size, sockaddr_in6 *target_ip, ip6_memaddr_block *ip6_block) {
  target_ip->sin6_addr = gen_rdm_ip6_target();
  ip6_memaddr_block tmp_block = allocate_uniform_rmem(target_ip, static_cast<size_t>(size));
  memcpy(ip6_block, &tmp_block, sizeof(ip6_memaddr_block));
//  if (ip6_block) {
//    std::stringstream ss;
//    ss << "malloc of size " << size << " failed";
//    return Status::OutOfRMem(ss.str());
//  }
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
  DefaultRMemPool() {
    h = kh_init(ptr_ip6_map);
    target_ip.sin6_port = ntohs(BB_PORT);
  }

  uint8_t *storeIP6(ip6_memaddr_block *ip6_block) {
    uint8_t* out;
    int ret;
    k = kh_put(ptr_ip6_map, h, (uint64_t) ip6_block, &ret);
    kh_value(h, k) = *ip6_block;
    out = (uint8_t *) ip6_block;
    return out;
  }

  ip6_memaddr_block PtrToIp6(void *ptr) {
      k = kh_get(ptr_ip6_map, h, (uint64_t) ptr);
      if (k == kh_end(h)) {  // k will be equal to kh_end if key not present
         printf("Key not found!\n");
      }
      return kh_value(h, k);
  }

  void Write(void *ptr, uint64_t offset,  uint8_t *data, uint64_t size) override {
    ip6_memaddr_block ip6_block = PtrToIp6(ptr);
    ip6_block.offset = offset;
    write_uniform_rmem(&target_ip, ip6_block, data, size);
  }

  void Read(void *ptr, uint64_t offset, uint8_t *buffer, uint64_t size) override {
    ip6_memaddr_block ip6_block = PtrToIp6(ptr);
    ip6_block.offset = offset;
    read_uniform_rmem(&target_ip, ip6_block, buffer, size);
  }

  Status Allocate(int64_t size, uint8_t** out) override {
    ip6_memaddr_block *ip6_block = (ip6_memaddr_block *)malloc(sizeof(ip6_memaddr_block));
    RETURN_NOT_OK(AllocateRMem(size, &target_ip, ip6_block));
    *out = storeIP6(ip6_block);
    stats_.UpdateAllocatedBytes(size);
    return Status::OK();
  }

  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) override {
    // Allocate new chunk
    uint8_t* out = nullptr;
    RETURN_NOT_OK(Allocate(new_size, &out));
    DCHECK(out);
    // Should be an explicit migrate call, for now we do it the slow, simple way
    ip6_memaddr_block src_block = PtrToIp6(*ptr);
    ip6_memaddr_block dst_block = PtrToIp6(out);
    uint8_t tmp_data[old_size];
    read_uniform_rmem(&target_ip, src_block, tmp_data, std::min(new_size, old_size));
    write_uniform_rmem(&target_ip, dst_block, tmp_data, std::min(new_size, old_size));
    this->Free(*ptr, old_size);
    *ptr = out;

    return Status::OK();
  }

  int64_t bytes_allocated() const override { return stats_.bytes_allocated(); }

  void Free(uint8_t* buffer, int64_t size) override {
    ip6_memaddr_block ip6_block = PtrToIp6(buffer);
    int status = free_rmem(&target_ip, &ip6_block.memaddr);
    stats_.UpdateAllocatedBytes(-size);
  }

  int64_t max_rmem() const override { return stats_.max_rmem(); }

 private:
  RMemPoolStats stats_;
  khiter_t k;
  khash_t(ptr_ip6_map) *h;
  struct sockaddr_in6 target_ip;

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

void LoggingRMemPool::Write(void *ptr, uint64_t offset, uint8_t* data, uint64_t size) {
  pool_->Write(ptr, offset, data, size);
}

void LoggingRMemPool::Read(void *ptr, uint64_t offset, uint8_t* buffer, uint64_t size){
  pool_->Read(ptr, offset, buffer, size);
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

  void Write(void *ptr, uint64_t offset, uint8_t* data, uint64_t size) {pool_->Write(ptr, offset, data, size);}

  void Read(void *ptr, uint64_t offset, uint8_t* buffer, uint64_t size) {pool_->Read(ptr, offset, buffer, size);}

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

void ProxyRMemPool::Write(void *ptr, uint64_t offset, uint8_t* data, uint64_t size) {
  impl_->Write(ptr, offset, data, size);
}

void ProxyRMemPool::Read(void *ptr, uint64_t offset, uint8_t* buffer, uint64_t size){
  impl_->Read(ptr, offset, buffer, size);
}

int64_t ProxyRMemPool::bytes_allocated() const { return impl_->bytes_allocated(); }

int64_t ProxyRMemPool::max_rmem() const { return impl_->max_rmem(); }


}  // namespace arrow
