#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  portBLAS: BLAS implementation using SYCL
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/
set(PORTBLAS_EXPRTEST ${CMAKE_CURRENT_SOURCE_DIR})

# compiling tests
set(SYCL_EXPRTEST_SRCS
  ${PORTBLAS_EXPRTEST}/blas1_scal_asum_test.cpp
  ${PORTBLAS_EXPRTEST}/blas1_axpy_copy_test.cpp
  ${PORTBLAS_EXPRTEST}/collapse_nested_tuple.cpp
  )

foreach(blas_test ${SYCL_EXPRTEST_SRCS})
  get_filename_component(test_exec ${blas_test} NAME_WE)
  add_executable(${test_exec} main.cpp ${blas_test})
  
  # -DTUNING_TARGET is needed when using portBLAS in header only mode.
  target_compile_definitions(${test_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_TEST_INDEX_TYPE} -D${TUNING_TARGET})
  target_link_libraries(${test_exec} PRIVATE gtest_main blas::blas portblas)
  target_include_directories(${test_exec} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." ${CBLAS_INCLUDE} ${PORTBLAS_SRC} ${PORTBLAS_COMMON_INCLUDE_DIR})
  add_sycl_to_target(
    TARGET ${test_exec}
    SOURCES ${blas_test}
  )
  if(TEST_DEVICE)
    add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec} --device ${TEST_DEVICE} --gtest_output=xml:output/)
  else()
    add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec} --gtest_output=xml:output/)
  endif()
  message(STATUS "Created google test ${test_exec}")
  install(TARGETS ${test_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT tests
  )
endforeach()
