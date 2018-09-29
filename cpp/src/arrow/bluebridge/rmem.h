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

// Public API for different memory sharing / IO mechanisms

#ifndef ARROW_IO_RMEM_H
#define ARROW_IO_RMEM_H

#include <cstdint>
#include <memory>

#include "arrow/io/interfaces.h"
#include "arrow/util/visibility.h"

namespace arrow {

class RMemBuffer;
class RMemPool;
class ResizableRMemBuffer;
class Status;

namespace io {

// \brief An output stream that writes to a resizable buffer
class ARROW_EXPORT RMemBufferOutputStream : public OutputStream {
 public:
  explicit RMemBufferOutputStream(const std::shared_ptr<ResizableRMemBuffer>& buffer);

  /// \brief Create in-memory output stream with indicated capacity using a
  /// memory pool
  /// \param[in] initial_capacity the initial allocated internal capacity of
  /// the OutputStream
  /// \param[in,out] pool a RMemPool to use for allocations
  /// \param[out] out the created stream
  static Status Create(int64_t initial_capacity, RMemPool* pool,
                       std::shared_ptr<RMemBufferOutputStream>* out);

  ~RMemBufferOutputStream() override;

  // Implement the OutputStream interface
  Status Close() override;
  Status Tell(int64_t* position) const override;
  Status Write(const void* data, int64_t nbytes) override;

  /// Close the stream and return the buffer
  Status Finish(std::shared_ptr<RMemBuffer>* result);

 private:
  // Ensures there is sufficient space available to write nbytes
  Status Reserve(int64_t nbytes);

  std::shared_ptr<ResizableRMemBuffer> buffer_;
  bool is_open_;
  int64_t capacity_;
  int64_t position_;
  uint8_t* mutable_data_;
};

// \brief A helper class to tracks the size of allocations
class ARROW_EXPORT RMemMockOutputStream : public OutputStream {
 public:
  RMemMockOutputStream() : extent_bytes_written_(0) {}

  // Implement the OutputStream interface
  Status Close() override;
  Status Tell(int64_t* position) const override;
  Status Write(const void* data, int64_t nbytes) override;

  int64_t GetExtentBytesWritten() const { return extent_bytes_written_; }

 private:
  int64_t extent_bytes_written_;
};

/// \brief Enables random writes into a fixed-size mutable buffer
class ARROW_EXPORT FixedSizeRMemBufferWriter : public WritableFile {
 public:
  /// Input buffer must be mutable, will abort if not
  explicit FixedSizeRMemBufferWriter(const std::shared_ptr<RMemBuffer>& buffer);
  ~FixedSizeRMemBufferWriter() override;

  Status Close() override;
  Status Seek(int64_t position) override;
  Status Tell(int64_t* position) const override;
  Status Write(const void* data, int64_t nbytes) override;
  Status WriteAt(int64_t position, const void* data, int64_t nbytes) override;

  void set_memcopy_threads(int num_threads);
  void set_memcopy_blocksize(int64_t blocksize);
  void set_memcopy_threshold(int64_t threshold);

 protected:
  class FixedSizeRMemBufferWriterImpl;
  std::unique_ptr<FixedSizeRMemBufferWriterImpl> impl_;
};

/// \class RMemBufferReader
/// \brief Random access zero-copy reads on an arrow::RMemBuffer
class ARROW_EXPORT RMemBufferReader : public RandomAccessFile {
 public:
  explicit RMemBufferReader(const std::shared_ptr<RMemBuffer>& buffer);
  explicit RMemBufferReader(const RMemBuffer& buffer);
  RMemBufferReader(const uint8_t* data, int64_t size);

  Status Close() override;
  Status Tell(int64_t* position) const override;
  Status Read(int64_t nbytes, int64_t* bytes_read, void* buffer) override;
  // Zero copy read
  Status Read(int64_t nbytes, std::shared_ptr<RMemBuffer>* out);

  Status ReadAt(int64_t position, int64_t nbytes, int64_t* bytes_read,
                void* out) override;
  Status ReadAt(int64_t position, int64_t nbytes, std::shared_ptr<RMemBuffer>* out);

  Status GetSize(int64_t* size) override;
  Status Seek(int64_t position) override;

  bool supports_zero_copy() const override;

  std::shared_ptr<RMemBuffer> buffer() const { return buffer_; }

 protected:
  std::shared_ptr<RMemBuffer> buffer_;
  const uint8_t* data_;
  int64_t size_;
  int64_t position_;
};

}  // namespace io
}  // namespace arrow

#endif  // ARROW_IO_MEMORY_H
