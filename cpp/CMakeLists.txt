cmake_minimum_required(VERSION 3.15)
project(TOP3D_XL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option to force MKL preference
option(USE_MKL "Prefer Intel MKL for BLAS" ON)

# Performance optimization flags
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Find BLAS/CBLAS
if(USE_MKL)
    set(BLA_VENDOR "Intel10_64lp")
    message(STATUS "Searching for Intel MKL (BLA_VENDOR=${BLA_VENDOR})...")
endif()

find_package(BLAS)

if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    add_compile_definitions(HAVE_CBLAS)

    # Check for cblas.h - Add MKL include paths to hints
    find_path(CBLAS_INCLUDE_DIR cblas.h
        HINTS
            $ENV{MKLROOT}/include
            /opt/intel/mkl/include
            /usr/include
            /usr/local/include
            /usr/include/openblas
            /usr/local/opt/openblas/include
            /opt/homebrew/opt/openblas/include
    )
    if(CBLAS_INCLUDE_DIR)
        message(STATUS "CBLAS header found: ${CBLAS_INCLUDE_DIR}")
    endif()
else()
    message(WARNING "BLAS not found - performance will be degraded (using manual implementation)")
endif()

add_library(top3d_xl
    src/core.cpp
    src/fea.cpp
    src/solver.cpp
    src/filter.cpp
    src/multigrid/hierarchy.cpp
    src/multigrid/transfers.cpp
    src/multigrid/masks.cpp
    src/multigrid/diagonal.cpp
    src/multigrid/coarsest.cpp
    src/multigrid/preconditioner.cpp
    src/io.cpp
    src/driver.cpp
)

# Important: Add CBLAS_INCLUDE_DIR *before* local includes to prefer system/MKL headers
if(CBLAS_INCLUDE_DIR)
    target_include_directories(top3d_xl PUBLIC ${CBLAS_INCLUDE_DIR})
endif()

target_include_directories(top3d_xl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_sources(top3d_xl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fea.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/solver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/filter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/driver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/multigrid/multigrid.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/multigrid/padding.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/geometry_out/export_stl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/geometry_out/voxel_surface.hpp
)

add_executable(top3d_xl_cli
    apps/top3d_cli.cpp
)

target_link_libraries(top3d_xl_cli PRIVATE top3d_xl)

# OpenMP support
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(top3d_xl PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link BLAS if found
if(BLAS_FOUND)
    target_link_libraries(top3d_xl PUBLIC ${BLAS_LIBRARIES})
endif()
