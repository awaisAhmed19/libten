cmake_minimum_required(VERSION 3.21)

project(
  libten
  VERSION 0.1.0
  DESCRIPTION "Standalone C11 tensor library with optional CUDA support"
  LANGUAGES C)

include(GNUInstallDirs)
include(CTest)

#
# Language standards
#

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type" FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
                                             RelWithDebInfo MinSizeRel)

#
# Build options
#

option(TEN_ENABLE_CUDA "Build libten with the NVIDIA CUDA backend" OFF)
option(ENABLE_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(TEN_CUDA_ARCHITECTURES
    "75"
    CACHE STRING "CUDA architectures to compile for (for example: 75 or 75;86)")

#
# Sanitizers
#

if(ENABLE_SANITIZERS)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "ENABLE_SANITIZERS requires GCC or Clang.")
  endif()
endif()

function(ten_enable_sanitizers target)
  if(ENABLE_SANITIZERS)
    target_compile_options(
      ${target}
      PRIVATE $<$<COMPILE_LANGUAGE:C>:-fsanitize=address,undefined>
              $<$<COMPILE_LANGUAGE:C>:-fno-sanitize-recover=all>
              $<$<COMPILE_LANGUAGE:C>:-fno-omit-frame-pointer>)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined
                        -fno-sanitize-recover=all)
  endif()
endfunction()

#
# Explicit CUDA configuration
#
# CUDA is OFF by default. When ON, CUDA is required: configuration fails rather
# than silently producing a CPU-only build.
#

set(TEN_CUDA_AVAILABLE OFF)

if(TEN_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(NOT CMAKE_CUDA_COMPILER)
    message(
      FATAL_ERROR
        "TEN_ENABLE_CUDA=ON, but no CUDA compiler (nvcc) was found. "
        "Install the CUDA Toolkit or configure with -DTEN_ENABLE_CUDA=OFF.")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_ARCHITECTURES "${TEN_CUDA_ARCHITECTURES}")

  set(TEN_CUDA_AVAILABLE ON)

  message(STATUS "CUDA compiler : ${CMAKE_CUDA_COMPILER}")
  message(STATUS "CUDA version  : ${CMAKE_CUDA_COMPILER_VERSION}")
  message(STATUS "CUDA arch     : ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CUDA backend  : explicitly disabled")
endif()

#
# Sources
#

set(TEN_SOURCES
    src/core/ten_dtype.c
    src/core/ten_tensor.c
    src/core/ten_type_checked.c
    src/core/ten_view.c
    src/internal/ten_backend.c
    src/backend/cpu/ten_memory.c
    src/backend/cpu/ten_ops.c)

if(TEN_CUDA_AVAILABLE)
  list(APPEND TEN_SOURCES src/backend/cuda/ten_memory.cu)
endif()

#
# Library
#

add_library(ten ${TEN_SOURCES})
add_library(libten::ten ALIAS ten)

target_compile_features(ten PUBLIC c_std_11)

target_include_directories(
  ten
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(
  ten
  PROPERTIES OUTPUT_NAME ten
             C_STANDARD 11
             C_STANDARD_REQUIRED ON
             C_EXTENSIONS OFF
             POSITION_INDEPENDENT_CODE ON)

ten_enable_sanitizers(ten)

#
# CUDA target configuration
#

if(TEN_CUDA_AVAILABLE)
  target_compile_definitions(ten PRIVATE USE_CUDA=1)
  target_link_libraries(ten PRIVATE CUDA::cudart)

  set_target_properties(
    ten
    PROPERTIES CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               CUDA_EXTENSIONS OFF
               CUDA_ARCHITECTURES "${TEN_CUDA_ARCHITECTURES}")
endif()

#
# Tests
#

if(BUILD_TESTING)

  # * Common test configuration -

  function(ten_configure_test target_name)
    target_link_libraries(${target_name} PRIVATE ten)

    target_include_directories(
      ${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                             ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    target_compile_features(${target_name} PRIVATE c_std_11)

    ten_enable_sanitizers(${target_name})
  endfunction()

  # * CPU / general tests -

  function(ten_add_test test_name source_file)
    set(target_name "ten_${test_name}_tests")

    add_executable(${target_name} "tests/${source_file}")

    ten_configure_test(${target_name})

    if(TEN_CUDA_AVAILABLE)
      target_compile_definitions(${target_name} PRIVATE TEN_TEST_HAS_CUDA=1)
    endif()

    add_test(NAME ${test_name} COMMAND ${target_name})
  endfunction()

  ten_add_test(helpers tensor_helpers_test.c)
  ten_add_test(constructor tensor_constructor_test.c)
  ten_add_test(share_storage tensor_share_storage_test.c)
  ten_add_test(shape tensor_shape_test.c)
  ten_add_test(tensor tensor_test.c)
  ten_add_test(checked tensor_checked_test.c)
  ten_add_test(views tensor_views_test.c)

  # * CUDA memory and copy tests -

  if(TEN_CUDA_AVAILABLE)

    function(ten_add_cuda_test test_name source_file)
      set(target_name "ten_${test_name}_tests")

      add_executable(${target_name} "tests/cuda/${source_file}")

      ten_configure_test(${target_name})

      target_include_directories(
        ${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda
                               ${CUDAToolkit_INCLUDE_DIRS})

      target_link_libraries(${target_name} PRIVATE CUDA::cudart)

      add_test(NAME ${test_name} COMMAND ${target_name})

      set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    ten_add_cuda_test(cuda_memory cuda_memory_test.c)
    ten_add_cuda_test(cuda_copy cuda_copy_test.c)

  endif()

endif()

#
# Installation
#

install(
  TARGETS ten
  EXPORT libtenTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/ten DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT libtenTargets
  FILE libtenTargets.cmake
  NAMESPACE libten::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ten")

#
# Build summary
#

if(TEN_CUDA_AVAILABLE)
  set(TEN_BACKEND_NAME "CPU + CUDA")
else()
  set(TEN_BACKEND_NAME "CPU only")
endif()

message(STATUS "")
message(STATUS "libten ${PROJECT_VERSION}")
message(
  STATUS "C compiler : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "C standard : C11")
message(STATUS "Backend    : ${TEN_BACKEND_NAME}")
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Build dir  : ${CMAKE_BINARY_DIR}")
message(STATUS "")
