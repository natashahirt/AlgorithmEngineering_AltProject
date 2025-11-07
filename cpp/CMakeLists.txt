cmake_minimum_required(VERSION 3.15)
project(TOP3D_XL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(top3d_xl
    src/core/top3d_xl.cpp
    src/optimization/MMAseq.cpp
    src/io/topvoxel.cpp
    src/geometry/isosurface.cpp 
)
target_include_directories(top3d_xl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(top3d_xl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/top3d_xl/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/top3d_xl/mesh/multigrid_padding.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/top3d_xl/io/topvoxel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/top3d_xl/io/export.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/top3d_xl/geometry/voxel_surface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/top3d_xl/geometry/isosurface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/top3d_xl/optimization/MMAseq.hpp
)

add_executable(top3d_xl_cli
    src/apps/top3d_xl_cli/main.cpp
)
target_link_libraries(top3d_xl_cli PRIVATE top3d_xl)