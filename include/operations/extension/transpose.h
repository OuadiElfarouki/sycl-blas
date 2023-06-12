/***************************************************************************
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
 *  @filename transpose.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_TRANSPOSE_H
#define SYCL_BLAS_EXTENSION_TRANSPOSE_H

#include <CL/sycl.hpp>

#include "container/sycl_iterator.h"

namespace blas {

/*!
 * @brief This class holds the kernel for the matrix transpose operator. It can
 * also be used to perform matrix scaling and other pre/post-transpose
 * operations.
 *
 * This templated kernel is designed to use or not local memory, while remaining
 * customizable Tiling-size wise for performance purposes.
 *
 * @tparam in_place Whether the transpose is in or out of place
 * @tparam Tile_size Tiling size used explicitly in the local memory kernel, and
 * used to compute work-group size in the non-local memory case.
 * @tparam wg_size work group size
 * @tparam cl_size cache line size
 * @tparam local_memory Whether to use local memory
 * @tparam in_t The input matrix type
 * @tparam out_t The output matrix type
 * @tparam element_t The scaling factor type
 *
 */
template <bool in_place, int Tile_size, int wg_size, int cl_size,
          bool local_memory, typename in_t, typename out_t, typename element_t>
class Transpose {
 public:
  using index_t = typename in_t::index_t;
  using value_t = element_t;
  in_t A_;
  out_t At_;
  index_t N_;
  index_t M_;
  value_t alpha_;
  // Leading dimensions
  index_t lda_;
  index_t ldat_;
  // Increment value (denoted stride in oneMKL specification)
  index_t inc_a_;
  index_t inc_at_;
  // Stride values (denoted stride in oneMKL specification for batched
  // operations)
  index_t stride_a_;
  index_t stride_at_;
  // Minimum number of tiles used to cover matrices rows & columns
  index_t tile_count_m_;
  index_t tile_count_n_;
  // Total number of tiles used to cover the matrix
  index_t tile_count_total_;
  // Number of Inner WG Tiles
  static constexpr const index_t inner_tile_size_ = wg_size / Tile_size;
  static constexpr const index_t inner_tile_count_ =
      Tile_size / inner_tile_size_;
  // Minimum number of Tile-mutliple rows & columns to cover the matrices
  index_t M_pad_;
  index_t N_pad_;
  // Total size of Tile-mutliple covering matrix
  index_t size_pad_;
  // Batch size when using batched transpose
  index_t batch_size_;

  Transpose(in_t &A, index_t &inc_a, index_t &stride_a, out_t &At,
            index_t &inc_at, index_t &stride_at, value_t &alpha,
            index_t &batch_size)
      : A_(A),
        At_(At),
        lda_(A_.getSizeL()),
        ldat_(At_.getSizeL()),
        M_(A_.get_size_row()),
        N_(A_.get_size_col()),
        alpha_(alpha),
        tile_count_m_((M_ - 1) / Tile_size + 1),
        tile_count_n_((N_ - 1) / Tile_size + 1),
        tile_count_total_(tile_count_m_ * tile_count_n_),
        inc_a_(inc_a),
        stride_a_(stride_a),
        stride_at_(stride_at),
        inc_at_(inc_at),
        M_pad_(tile_count_m_ * Tile_size),
        N_pad_(tile_count_n_ * Tile_size),
        size_pad_(M_pad_ * N_pad_),
        batch_size_(batch_size) {}

  index_t get_size() const;

  bool valid_thread(cl::sycl::nd_item<1> item) const;
  void bind(cl::sycl::handler &cgh);
  void adjust_access_displacement();
  void eval(cl::sycl::nd_item<1> item);
  template <typename local_memory_t>
  void eval(local_memory_t local_mem, cl::sycl::nd_item<1> id);
  void get_indices(cl::sycl::nd_item<1> id, index_t &in_idx,
                   index_t &in_local_idx, index_t &out_idx,
                   index_t &out_local_idx, bool &valid_index_in,
                   bool &valid_index_out);
};

/*!
 @brief Generator/factory for Transpose trees.
 */
template <bool in_place, int Tile_size, int wg_size, int cl_size,
          bool local_memory, typename in_t, typename out_t, typename element_t,
          typename index_t>
Transpose<in_place, Tile_size, wg_size, cl_size, local_memory, in_t, out_t,
          element_t>
make_transpose(in_t &A, index_t inc_a, index_t &stride_a, out_t &At,
               index_t inc_a_t, index_t &stride_at, element_t &alpha,
               index_t &batch_size) {
  return Transpose<in_place, Tile_size, wg_size, cl_size, local_memory, in_t,
                   out_t, element_t>(A, inc_a, stride_a, At, inc_a_t, stride_at,
                                     alpha, batch_size);
}

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TRANSPOSE_H
