cmake_minimum_required(VERSION 3.15)
project(TOP3D_XL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_DEBUG   "-g -O0")

option(ENABLE_SANITIZERS "Enable Address/UB/Thread sanitizers" ON)

if(ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Adding sanitizers to Debug build")

    set(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SANITIZER_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} ${SANITIZER_FLAGS}")

    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${SANITIZER_FLAGS}")
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

# OpenMP support: prefer CMake's FindOpenMP; on macOS fall back to Homebrew libomp
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(top3d_xl PUBLIC OpenMP::OpenMP_CXX)
endif()
