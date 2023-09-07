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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename gemm_load_store.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_GEMM_LOAD_STORE_HPP
#define PORTBLAS_BLAS3_GEMM_LOAD_STORE_HPP

namespace blas {

/*! @brief Contains static methods for loading and storing vector packets
from/to non-vectorized memory as well as some constants for the vector type and
packet size. SFINAE is used to select the appropriate method when called.
* @tparam vector_size The desired vector size to be used. If
GEMM_VECTORIZATION_SUPPORT is not enabled in CMake a vector_size of 1 will be
used no matter what value is passed here.
* @tparam value_t The type of the matrix data (typically float or double, if
supported).
*/
template <int vector_size, typename value_t, typename index_t>
struct Packetize {
#ifdef GEMM_VECTORIZATION_SUPPORT
  using PacketType = cl::sycl::vec<value_t, vector_size>;
  static constexpr int packet_size = vector_size;
  template <index_t dimension>
  PORTBLAS_INLINE static constexpr bool check_size() {
    return packet_size == 1 || dimension == packet_size;
  }
#else
  // In the case where vectorization is not enabled, always set to 1
  using PacketType = cl::sycl::vec<value_t, 1>;
  static constexpr int packet_size = 1;
  template <index_t dimension>
  PORTBLAS_INLINE static constexpr bool check_size() {
    return true;
  }
#endif

  /*! @brief Performs a coalesced non-vectorized load when the current block is
   * not internal.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory.
   */

  template <bool trans, bool internal, int ld, typename SrcPointerType,
            typename DestPointerType, typename EdgePredicate>
  static PORTBLAS_INLINE typename std::enable_if<!internal>::type load(
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate) {
#ifdef SB_ENABLE_JOINT_MATRIX
    value_t val = in_range ? *(src) : value_t{0};
    using address_t = cl::sycl::access::address_space;
    if constexpr (std::is_same<cl::sycl::multi_ptr<cl::sycl::half,
                                                   address_t::local_space>,
                               DestPointerType>::value) {
      using dtype = cl::sycl::half;
      *dest = static_cast<dtype>(val);
    } else if constexpr (std::is_same<cl::sycl::multi_ptr<
                                          cl::sycl::ext::oneapi::bfloat16,
                                          address_t::local_space>,
                                      DestPointerType>::value) {
      using dtype = cl::sycl::ext::oneapi::bfloat16;
      *dest = static_cast<dtype>(val);
    } else {
      using namespace cl::sycl::ext::oneapi::experimental::matrix;
      *dest = round_to_tf32(val);
    }
#else
    *(dest) = in_range ? *(src) : value_t{0};
#endif
  }
  /*! @brief Performs a vectorised load using sycl::vec::load when the current
   * block is internal. In the case where k < the
   * number of elements being loaded then edge loads will be element wise with
   * additional bounds checking.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory. */
  template <bool trans, bool internal, index_t ld, typename SrcPointerType,
            typename DestPointerType, typename EdgePredicate>
  static PORTBLAS_INLINE typename std::enable_if<internal>::type load(
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate edge_in_range) {
    PacketType packet{};

    if (in_range) {
      using address_t = cl::sycl::access::address_space;
      packet.template load<address_t::global_space>(
          0, cl::sycl::multi_ptr<const value_t, address_t::global_space>(src));
    } else {
#pragma unroll
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<value_t *>(&packet)[i] =
            edge_in_range(i) ? *(src + i) : value_t{0};
      }
    }
    store<trans, ld>(packet, dest);
  }
  /*! @brief Store a vector packet into local memory when the source is
   * transposed. This will untranspose the elements individually when storing so
   * the data in local memory is always consistent.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam ld The leading dimension of the destination memory.*/
  template <bool trans, index_t ld, typename DestPointerType>
  static PORTBLAS_INLINE typename std::enable_if<trans>::type store(
      PacketType &packet, DestPointerType dest) {
#ifdef SB_ENABLE_JOINT_MATRIX
    using address_t = cl::sycl::access::address_space;
#pragma unroll
    for (index_t i = 0; i < packet_size; i++) {
      value_t val = reinterpret_cast<value_t *>(&packet)[i];
      if constexpr (std::is_same<cl::sycl::multi_ptr<cl::sycl::half,
                                                     address_t::local_space>,
                                 DestPointerType>::value) {
        using dtype = cl::sycl::half;
        *(dest + ld * i) = static_cast<dtype>(val);
      } else if constexpr (std::is_same<cl::sycl::multi_ptr<
                                            cl::sycl::ext::oneapi::bfloat16,
                                            address_t::local_space>,
                                        DestPointerType>::value) {
        using dtype = cl::sycl::ext::oneapi::bfloat16;
        *(dest + ld * i) = static_cast<dtype>(val);
      } else {
        using namespace cl::sycl::ext::oneapi::experimental::matrix;
        *(dest + ld * i) = round_to_tf32(val);
      }
    }
#else
#pragma unroll
    for (index_t i = 0; i < packet_size; i++) {
      *(dest + ld * i) = reinterpret_cast<value_t *>(&packet)[i];
    }
#endif
  }

  /*! @brief Store a vector packet into local memory when the source is not
   * transposed. This will use sycl::vec::store function.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam ld The leading dimension of the destination memory.*/
  template <bool trans, int ld, typename DestPointerType>
  static PORTBLAS_INLINE typename std::enable_if<!trans>::type store(
      PacketType &packet, DestPointerType dest) {
    using address_t = cl::sycl::access::address_space;
#ifdef SB_ENABLE_JOINT_MATRIX
    if constexpr (std::is_same<cl::sycl::multi_ptr<cl::sycl::half,
                                                   address_t::local_space>,
                               DestPointerType>::value) {
      using dtype = cl::sycl::half;
      *dest = static_cast<dtype>(packet[0]);
    } else if constexpr (std::is_same<cl::sycl::multi_ptr<
                                          cl::sycl::ext::oneapi::bfloat16,
                                          address_t::local_space>,
                                      DestPointerType>::value) {
      using dtype = cl::sycl::ext::oneapi::bfloat16;
      *dest = static_cast<dtype>(packet[0]);
    } else {
      using namespace cl::sycl::ext::oneapi::experimental::matrix;
      *dest = round_to_tf32(packet[0]);
    }
#else
    packet.template store<address_t::local_space>(
        0, cl::sycl::multi_ptr<value_t, address_t::local_space>(dest));
#endif
  }
};

#ifdef BLAS_ENABLE_COMPLEX
/*! @brief vec_complex is an intermediate wrapper of sycl::complex used in
 * Packetize. It serves as a temporary workaround to the upcoming
 * sycl::vec<syc::complex> container
 * github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_complex.asciidoc
 * and only supports size = 1.
 * @tparam DataT Complex type of the vector's data
 * @tparam NumElements Elements count of the vector (only 1 is supported)
 */
template <typename DataT, int NumElements = 1>
class vec_complex {
  static_assert(NumElements == 1,
                "Vector wrapper arround sycl::complex of size>1 unsupported.");
  using address_t = cl::sycl::access::address_space;
  using decorated_t = cl::sycl::access::decorated;
  using DataType = DataT;
  static constexpr int getNumElements() { return NumElements; }
  size_t size() const noexcept { return NumElements; }

 private:
  DataType m_Data;

 public:
  vec_complex() = default;

  constexpr vec_complex(const vec_complex &rhs) = default;
  constexpr vec_complex(vec_complex &&rhs) = default;
  constexpr vec_complex &operator=(const vec_complex &rhs) = default;

  vec_complex(const DataType &rhs_data) : m_Data{rhs_data} {}

  // Conversion operator (valid with NumElements==1)
  operator DataT() const { return m_Data; }

  // Subscript operators
  DataT &operator[](int i) {
    assert(i < NumElements);
    return (m_Data);
  }
  const DataT &operator[](int i) const {
    assert(i < NumElements);
    return (m_Data);
  }

  // Binary Ops
  // Multiply
  vec_complex operator*(const vec_complex &rhs) {
    return (vec_complex{m_Data * static_cast<DataT>(rhs)});
  }

  vec_complex operator*(const DataType &rhs) {
    return (vec_complex{m_Data * rhs});
  }

  // Compound Multiply
  vec_complex &operator*=(const DataType &rhs) {
    this->m_Data = this->m_Data * rhs;
    return (*this);
  }

  vec_complex &operator*=(const vec_complex &rhs) {
    this->m_Data = this->m_Data * static_cast<DataT>(rhs);
    return (*this);
  }

  // Add
  vec_complex operator+(const vec_complex &rhs) {
    return (vec_complex{m_Data + static_cast<DataT>(rhs)});
  }

  vec_complex operator+(const DataType &rhs) {
    return (vec_complex{m_Data + rhs});
  }

  // Compound Add
  vec_complex &operator+=(const DataType &rhs) {
    this->m_Data = this->m_Data * rhs;
    return (*this);
  }

  vec_complex &operator+=(const vec_complex &rhs) {
    this->m_Data = this->m_Data + static_cast<DataT>(rhs);
    return (*this);
  }

  // Load
  template <address_t Space, decorated_t DecorateAddress>
  void load(size_t Offset,
            cl::sycl::multi_ptr<const DataT, Space, DecorateAddress> Ptr) {
    m_Data = *(Ptr + Offset * NumElements);
  }

  // Store
  template <address_t Space, decorated_t DecorateAddress>
  void store(size_t Offset,
             cl::sycl::multi_ptr<DataT, Space, DecorateAddress> Ptr) const {
    *(Ptr + Offset * NumElements) = m_Data;
  }
};

/*! @brief Partial specialization of the Packetize class dedicated to
sycl::complex types. It contains static methods for loading and storing size=1
complex packets from/to memory . SFINAE is used to select the appropriate method
when called.
* @tparam vector_size The desired vector size to be used. Only size = 1 is
supported so far.
* @tparam value_t The complex type of the matrix data.
*/
template <int vector_size, typename T, typename index_t>
struct Packetize<vector_size, complex_sycl<T>, index_t> {
  // Vectorization is not enabled for complex, always set to 1
  using value_t = complex_sycl<T>;
  using PacketType = vec_complex<value_t, 1>;
  static constexpr int packet_size = 1;
  template <index_t dimension>
  static PORTBLAS_INLINE constexpr bool check_size() {
    return true;
  }

  /*! @brief Performs a coalesced non-vectorized load when the current block is
   * not internal.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory.
   */

  template <bool trans, bool internal, int ld, typename SrcPointerType,
            typename DestPointerType, typename EdgePredicate>
  static PORTBLAS_INLINE typename std::enable_if<!internal>::type load(
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate) {
    *(dest) = in_range ? *(src) : value_t{(T)0, (T)0};
  }
  /*! @brief Performs a non vectorised load of sycl::complex data element while
   * checking if current block is internal.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory. */
  template <bool trans, bool internal, index_t ld, typename SrcPointerType,
            typename DestPointerType, typename EdgePredicate>
  static PORTBLAS_INLINE typename std::enable_if<internal>::type load(
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate edge_in_range) {
    *(dest) = in_range ? *(src) : value_t{(T)0, (T)0};
  }

  /*! @brief Store a size = 1 vector packet of sycl::complex data into local
   * memory (whether source is transposed or not since it's only 1 element).
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam ld The leading dimension of the destination memory.*/
  template <bool trans, index_t ld, typename DestPointerType>
  static PORTBLAS_INLINE void store(PacketType &packet, DestPointerType dest) {
    *dest = packet[0];
  }
};
#endif

}  // namespace blas
#endif  // PORTBLAS_BLAS3_GEMM_LOAD_STORE_HPP
