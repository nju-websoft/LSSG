cmake_minimum_required(VERSION 3.20)

project(lssg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LSSG_USE_MINHASH "Build the MinHash LSH variant (OFF builds IVF)" ON)
option(LSSG_ENABLE_NATIVE_SIMD "Enable CPU-native SSE/AVX2/AVX-512 optimizations" ON)
option(LSSG_BUILD_PYTHON "Build the optional pybind11 Python module" OFF)
find_package(OpenMP REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(LSSG_USE_MINHASH)
    add_compile_definitions(LSSG_USE_MINHASH=1)
else()
    add_compile_definitions(LSSG_USE_MINHASH=0)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -Wpedantic -fPIC)

    if(LSSG_ENABLE_NATIVE_SIMD)
        include(CheckCXXCompilerFlag)
        set(LSSG_SIMD_FLAGS -march=native -finline-functions)

        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64" AND EXISTS "/proc/cpuinfo")
            file(READ "/proc/cpuinfo" LSSG_CPUINFO)

            if(LSSG_CPUINFO MATCHES " avx2 ")
                check_cxx_compiler_flag("-mavx2" LSSG_COMPILER_SUPPORTS_AVX2)
                if(LSSG_COMPILER_SUPPORTS_AVX2)
                    list(APPEND LSSG_SIMD_FLAGS -mavx2)
                endif()
            endif()

            if(LSSG_CPUINFO MATCHES " avx ")
                check_cxx_compiler_flag("-mavx" LSSG_COMPILER_SUPPORTS_AVX)
                if(LSSG_COMPILER_SUPPORTS_AVX)
                    list(APPEND LSSG_SIMD_FLAGS -mavx)
                endif()
            endif()

            if(LSSG_CPUINFO MATCHES " sse ")
                check_cxx_compiler_flag("-msse" LSSG_COMPILER_SUPPORTS_SSE)
                if(LSSG_COMPILER_SUPPORTS_SSE)
                    list(APPEND LSSG_SIMD_FLAGS -msse)
                endif()
            endif()

            if(LSSG_CPUINFO MATCHES " avx512f ")
                check_cxx_compiler_flag("-mavx512f" LSSG_COMPILER_SUPPORTS_AVX512F)
                if(LSSG_COMPILER_SUPPORTS_AVX512F)
                    list(APPEND LSSG_SIMD_FLAGS -mavx512f)
                endif()
            endif()

            if(LSSG_CPUINFO MATCHES " avx512dq ")
                check_cxx_compiler_flag("-mavx512dq" LSSG_COMPILER_SUPPORTS_AVX512DQ)
                if(LSSG_COMPILER_SUPPORTS_AVX512DQ)
                    list(APPEND LSSG_SIMD_FLAGS -mavx512dq)
                endif()
            endif()

            if(LSSG_CPUINFO MATCHES " avx512_vpopcntdq " OR LSSG_CPUINFO MATCHES " avx512vpopcntdq ")
                check_cxx_compiler_flag("-mavx512vpopcntdq" LSSG_COMPILER_SUPPORTS_AVX512VPOPCNTDQ)
                if(LSSG_COMPILER_SUPPORTS_AVX512VPOPCNTDQ)
                    list(APPEND LSSG_SIMD_FLAGS -mavx512vpopcntdq)
                endif()
            endif()

            if(LSSG_SIMD_FLAGS MATCHES "mavx512f")
                message(STATUS "LSSG SIMD: AVX-512F extensions enabled")
            endif()
        endif()

        add_compile_options(${LSSG_SIMD_FLAGS})
        message(STATUS "LSSG SIMD flags: ${LSSG_SIMD_FLAGS}")
    else()
        message(STATUS "LSSG SIMD: disabled; using portable compiler defaults")
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lssg)

if(LSSG_USE_MINHASH)
    message(STATUS "LSSG variant: MinHash")
else()
    message(STATUS "LSSG variant: IVF")
endif()

add_subdirectory(example)

if(LSSG_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        RESULT_VARIABLE LSSG_PYBIND11_RESULT
        OUTPUT_VARIABLE LSSG_PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(LSSG_PYBIND11_RESULT EQUAL 0 AND EXISTS "${LSSG_PYBIND11_CMAKE_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${LSSG_PYBIND11_CMAKE_DIR}")
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(python)
endif()
