# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#######################################
# bluebridge
#######################################
set (BLUEBRIDGE_DIR ${CMAKE_SOURCE_DIR}/../../../ip6)
include_directories(${BLUEBRIDGE_DIR}/lib)
link_directories(${BLUEBRIDGE_DIR})

set(BLUEBRIDGE_TEST_LINK_LIBS bluebridge ${ARROW_TEST_LINK_LIBS})

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/src/arrow/bluebridge)

install(FILES
  rmem_pool.h
  rmem_buffer.h
  rmem.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/arrow/bluebridge")

ADD_ARROW_TEST(rmem_pool-test STATIC_LINK_LIBS ${BLUEBRIDGE_TEST_LINK_LIBS})
ADD_ARROW_TEST(rmem_buffer-test STATIC_LINK_LIBS ${BLUEBRIDGE_TEST_LINK_LIBS})
