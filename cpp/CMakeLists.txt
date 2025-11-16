cmake_minimum_required(VERSION 3.15)
project(TOP3D_XL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(top3d_xl
    src/top3d_xl.cpp
    src/isosurface.cpp
)

target_include_directories(top3d_xl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(top3d_xl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/multigrid_padding.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/export.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/geometry/voxel_surface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/geometry/isosurface.hpp
)

add_executable(top3d_xl_cli
    src/main.cpp
)

target_link_libraries(top3d_xl_cli PRIVATE top3d_xl)