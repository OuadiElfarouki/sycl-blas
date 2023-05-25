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

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
class Transpose {
 public:
  using index_t = typename in_t::index_t;
  using value_t = element_t;
  in_t A_;
  out_t At_;
  index_t lda_;
  index_t ldat_;
  index_t stridea_;
  index_t strideat_;
  index_t N_;
  index_t M_;
  value_t alpha_;
  index_t tile_count_m_;
  index_t tile_count_n_;
  index_t M_pad_;
  index_t N_pad_;

  Transpose(in_t &A, index_t &stridea, out_t &At, index_t &strideat,
            value_t &alpha)
      : A_(A),
        At_(At),
        lda_(A_.getSizeL()),
        ldat_(At_.getSizeL()),
        M_(A_.get_size_row()),
        N_(A_.get_size_col()),
        alpha_(alpha),
        tile_count_m_((M_ - 1) / Tile_size + 1),
        tile_count_n_((N_ - 1) / Tile_size + 1),
        stridea_(stridea),
        strideat_(strideat),
        M_pad_(tile_count_m_ * Tile_size),
        N_pad_(tile_count_n_ * Tile_size) {}

  index_t get_size() const;

  bool valid_thread(cl::sycl::nd_item<1> item) const;
  void bind(cl::sycl::handler &cgh);
  void adjust_access_displacement();
  void eval(cl::sycl::nd_item<1> item);
  template <typename local_memory_t>
  void eval(local_memory_t local_mem, cl::sycl::nd_item<1> id);

  template <typename index_t>
  void compute_trans_indices(cl::sycl::nd_item<1> id, index_t &in_idx,
                             index_t &in_idc, index_t &out_idx,
                             index_t &out_idc, bool &valid_index_in,
                             bool &valid_index_out);
};

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t, typename index_t>
Transpose<in_place, Tile_size, local_memory, in_t, out_t, element_t>
make_transpose(in_t &A, index_t stridea, out_t &At, index_t stridea_t,
               element_t &alpha) {
  return Transpose<in_place, Tile_size, local_memory, in_t, out_t, element_t>(
      A, stridea, At, stridea_t, alpha);
}

// Transpose-Add
template <bool both_trans, int Tile_size, bool local_memory, typename in1_t,
          typename in2_t, typename out_t, typename element_t>
class TransposeAdd {
 public:
  using index_t = typename in1_t::index_t;
  using value_t = element_t;
  in1_t A_;
  in2_t B_;
  out_t C_;

  index_t lda_;
  index_t ldb_;
  index_t ldc_;

  index_t N_;
  index_t M_;
  value_t alpha_;
  value_t beta_;

  index_t tile_count_m_;
  index_t tile_count_n_;

  index_t M_pad_;
  index_t N_pad_;

  TransposeAdd(in1_t &A, in2_t &B, out_t &C, value_t &alpha, value_t &beta)
      : A_(A),
        B_(B),
        C_(C),
        lda_(A_.getSizeL()),
        ldb_(B_.getSizeL()),
        ldc_(C_.getSizeL()),
        M_(C_.get_size_row()),
        N_(C_.get_size_col()),
        alpha_(alpha),
        beta_(beta),
        tile_count_m_((M_ - 1) / Tile_size + 1),
        tile_count_n_((N_ - 1) / Tile_size + 1),
        M_pad_(tile_count_m_ * Tile_size),
        N_pad_(tile_count_n_ * Tile_size) {}

  index_t get_size() const;

  bool valid_thread(cl::sycl::nd_item<1> item) const;
  void bind(cl::sycl::handler &cgh);
  void adjust_access_displacement();
  void eval(cl::sycl::nd_item<1> item);
  template <typename local_memory_t>
  void eval(local_memory_t local_mem, cl::sycl::nd_item<1> id);

  template <typename index_t>
  void compute_trans_add_indices(cl::sycl::nd_item<1> id, index_t &in_a_idx,
                                 index_t &in_b_idx, index_t &in_local_idx,
                                 index_t &out_idx, index_t &out_local_idx,
                                 bool &valid_index_in, bool &valid_index_out);
};

template <bool both_trans, int Tile_size, bool local_memory, typename in1_t,
          typename in2_t, typename out_t, typename element_t>
TransposeAdd<both_trans, Tile_size, local_memory, in1_t, in2_t, out_t,
             element_t>
make_transpose_add(in1_t &A, in2_t &B, out_t &C, element_t &alpha,
                   element_t &beta) {
  return TransposeAdd<both_trans, Tile_size, local_memory, in1_t, in2_t, out_t,
                      element_t>(A, B, C, alpha, beta);
}

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TRANSPOSE_H