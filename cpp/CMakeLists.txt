cmake_minimum_required(VERSION 3.15)
project(TOP3D_XL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(top3d_xl PUBLIC OpenMP::OpenMP_CXX)
else()
    if (APPLE)
        find_library(LIBOMP_LIB NAMES omp libomp PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib NO_DEFAULT_PATH)
        if (LIBOMP_LIB)
            # AppleClang needs -Xpreprocessor -fopenmp
            target_compile_options(top3d_xl PUBLIC "-Xpreprocessor" "-fopenmp")
            target_link_libraries(top3d_xl PUBLIC ${LIBOMP_LIB})
            target_include_directories(top3d_xl PUBLIC /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
        else()
            message(WARNING "OpenMP not found: building without OpenMP (install libomp via Homebrew to enable)")
        endif()
    else()
        message(WARNING "OpenMP not found: building without OpenMP")
    endif()
endif()