/*
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename intel_gpu.hpp
 *
 */

#ifndef SYCL_BLAS_BLAS1_INTEL_GPU_HPP
#define SYCL_BLAS_BLAS1_INTEL_GPU_HPP

namespace blas {
namespace backend {

class HardwareSpec {
 private:
  static constexpr int vector_size = 4;

 public:
  static constexpr int get_vector_size() { return vector_size; }
};

}  // namespace backend
}  // namespace blas

#endif  // SYCL_BLAS_BLAS1_INTEL_GPU_HPP
