cmake_minimum_required(VERSION 3.21)

project(
  libten
  VERSION 0.1.0
  DESCRIPTION "Standalone C11 tensor library with NVIDIA CUDA support"
  LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_SANITIZERS "Enable runtime sanitizers" OFF)

if(ENABLE_SANITIZERS)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(TEN_SANITIZER_FLAGS -fsanitize=address,undefined
                            -fno-sanitize-recover=all -fno-omit-frame-pointer)
  else()
    message(FATAL_ERROR "ENABLE_SANITIZERS requires GCC or Clang")
  endif()
endif()

function(ten_enable_sanitizers target)
  if(ENABLE_SANITIZERS)
    target_compile_options(${target} PRIVATE ${TEN_SANITIZER_FLAGS})
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type" FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
                                             RelWithDebInfo MinSizeRel)

# ---------------------------------------------------------------------------
# CUDA detection — optional backend, automatic CPU fallback
# ---------------------------------------------------------------------------

include(CheckLanguage)
check_language(CUDA)

set(TEN_CUDA_AVAILABLE OFF)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  set(CMAKE_CUDA_ARCHITECTURES
      75
      CACHE STRING "NVIDIA CUDA architectures")

  find_package(CUDAToolkit QUIET)

  if(CUDAToolkit_FOUND)
    set(TEN_CUDA_AVAILABLE ON)

    message(STATUS "CUDA compiler : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA version  : ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA arch     : ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(
      WARNING "CUDA compiler found, but the CUDA Toolkit could not be found. "
              "Building CPU backend only.")
  endif()
else()
  message(STATUS "CUDA compiler not found. Building CPU backend only.")
endif()

# ---------------------------------------------------------------------------
# Library sources
# ---------------------------------------------------------------------------

file(GLOB_RECURSE TEN_C_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.c"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.c")

if(NOT TEN_C_SOURCES)
  message(FATAL_ERROR "No C sources found under src/core or src/cpu.")
endif()

add_library(ten ${TEN_C_SOURCES})
add_library(libten::ten ALIAS ten)

ten_enable_sanitizers(ten)
target_compile_features(ten PUBLIC c_std_11)

target_include_directories(
  ten
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

set_target_properties(
  ten
  PROPERTIES OUTPUT_NAME ten
             C_STANDARD 11
             C_STANDARD_REQUIRED ON
             C_EXTENSIONS OFF
             POSITION_INDEPENDENT_CODE ON)

if(TEN_CUDA_AVAILABLE)
  # Enable CUDA runtime calls from C source files.
  target_compile_definitions(ten PRIVATE USE_CUDA)

  # Required by cudaMalloc, cudaMemcpy and cudaFree.
  target_link_libraries(ten PRIVATE CUDA::cudart)

  # CUDA kernels are optional; runtime allocation does not require .cu files.
  file(GLOB_RECURSE TEN_CUDA_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")

  if(TEN_CUDA_SOURCES)
    target_sources(ten PRIVATE ${TEN_CUDA_SOURCES})
    target_link_libraries(ten PRIVATE CUDA::cublas)
    set_target_properties(
      ten
      PROPERTIES CUDA_STANDARD 17
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_SEPARABLE_COMPILATION ON
                 CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  endif()

  message(STATUS "libten backend: CPU + CUDA runtime")
else()
  message(STATUS "libten backend: CPU only")
endif()

if(BUILD_TESTING)
  # Each white-box test includes tensor.c directly. Build each test in its own
  # executable to avoid duplicate symbols and main().
  function(ten_add_whitebox_test test_name source_file)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${source_file}")
      set(target_name "ten_${test_name}_tests")

      add_executable(${target_name} "tests/${source_file}" src/core/ten_dtype.c)

      ten_enable_sanitizers(${target_name})

      target_compile_features(${target_name} PRIVATE c_std_11)

      target_include_directories(
        ${target_name}
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests")

      add_test(NAME "${test_name}" COMMAND ${target_name})
    endif()
  endfunction()

  ten_add_whitebox_test(helpers tensor_helpers_test.c)
  ten_add_whitebox_test(constructor tensor_constructor_test.c)
  ten_add_whitebox_test(share_storage tensor_share_storage_test.c)
  ten_add_whitebox_test(shape tensor_share_test.c)
  ten_add_whitebox_test(tensor tensor_test.c)
  ten_add_whitebox_test(checked tensor_checked_test.c)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_tensor.c")
  add_executable(ten_basic_example examples/basic_tensor.c)
  if(TARGET ten_basic_example)
    ten_enable_sanitizers(ten_basic_example)
  endif()

  target_compile_features(ten_basic_example PRIVATE c_std_11)
  target_link_libraries(ten_basic_example PRIVATE libten::ten)
endif()

install(
  TARGETS ten
  EXPORT libtenTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(DIRECTORY include/ten DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tenConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/tenConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ten")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/tenConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(
  EXPORT libtenTargets
  FILE libtenTargets.cmake
  NAMESPACE libten::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ten")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tenConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/tenConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ten")

message(STATUS "")
message(STATUS "libten ${PROJECT_VERSION}")
message(
  STATUS "C compiler   : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "C standard   : C11")
if(TEN_CUDA_AVAILABLE)
  message(STATUS "Backend      : CPU + CUDA")
  message(STATUS "CUDA version : ${CMAKE_CUDA_COMPILER_VERSION}")
  message(STATUS "CUDA arch    : ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "Backend      : CPU only")
endif()
message(STATUS "Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "Build dir    : ${CMAKE_BINARY_DIR}")
message(STATUS "")
