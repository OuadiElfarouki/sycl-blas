/***************************************************************************
 *
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
 *  @filename default_cpu.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_TRANSPOSE_DEFAULT_CPU_BACKEND_HPP
#define SYCL_BLAS_TRANSPOSE_DEFAULT_CPU_BACKEND_HPP
#include "interface/transpose_launcher.h"

namespace blas {
namespace extension {
namespace backend {

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _transpose_outplace(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t in_, index_t _ld_in, index_t _inc_in, index_t _stride_in,
    container_1_t out_, index_t _ld_out, index_t _inc_out, index_t _stride_out,
    index_t _batch_size) {
  return Transpose_Launcher<
      16, 64, 64, false>::template _select_transpose_outplace(sb_handle, _M, _N,
                                                              _alpha, in_,
                                                              _ld_in, _inc_in,
                                                              _stride_in, out_,
                                                              _ld_out, _inc_out,
                                                              _stride_out,
                                                              _batch_size);
}

template <bool both_trans, typename sb_handle_t, typename container_0_t,
          typename container_1_t, typename container_2_t, typename element_t,
          typename index_t>
typename sb_handle_t::event_t _transpose_add(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t a_, index_t _ld_a, index_t _a_rows, index_t _a_cols,
    index_t _stride_a, element_t _beta, container_1_t b_, index_t _ld_b,
    index_t _b_rows, index_t _b_cols, index_t _stride_b, container_2_t c_,
    index_t _ld_c, index_t _stride_c, index_t _batch_size) {
  return TransposeAdd_Launcher<both_trans, 16, 64, 64, false>::
      template _select_transpose_add(sb_handle, _M, _N, _alpha, a_, _ld_a,
                                     _a_rows, _a_cols, _stride_a, _beta, b_,
                                     _ld_b, _b_rows, _b_cols, _stride_b, c_,
                                     _ld_c, _stride_c, _batch_size);
}

}  // namespace backend
}  // namespace extension
}  // namespace blas

#endif