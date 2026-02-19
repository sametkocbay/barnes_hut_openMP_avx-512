cmake_minimum_required(VERSION 3.25.1)
project(barnesHut LANGUAGES CXX)

# Enable OpenMP
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to build tests (default OFF)
option(HPCLab_BUILD_TESTS "Build the unit tests" OFF)

include_directories(.)
include_directories(include)
include_directories(src)

# Define source files (excluding main.cpp for reuse in tests)
set(LIB_SOURCES
    src/Algorithm.cpp
    src/GravitationalForce.cpp
    src/Cube.cpp
    src/cubeNode.cpp
    src/initialization.cpp
    src/MeshTree.cpp
    src/Node.cpp
    src/Particle.cpp
    src/bruteForce.cpp
    src/particleNode.cpp
        config/JsonParser.cpp
)

add_executable(barnesHut
    include/Cube.h
    include/cubeNode.h
    include/initialization.h
    include/bruteForce.h
    include/SpaceDividing.h
    include/MeshTree.h
    include/Node.h
    include/Particle.h
    include/particleNode.h
    include/Algorithm.h
    include/InteractionForce.h
    include/GravitationalForce.h
        ${LIB_SOURCES}
    src/main.cpp
    config/JsonParser.h
)

target_compile_features(barnesHut PRIVATE cxx_std_17)

# AVX-512 SIMD support (auto-detected, can be disabled with -DUSE_AVX512=OFF)
option(USE_AVX512 "Enable AVX-512 SIMD optimizations if supported" ON)

if(USE_AVX512)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    if(COMPILER_SUPPORTS_AVX512F)
        set(AVX512_FLAG "-mavx512f")
        message(STATUS "AVX-512F support enabled")
    else()
        set(AVX512_FLAG "")
        message(STATUS "AVX-512F not supported by compiler, using scalar fallback")
    endif()
else()
    set(AVX512_FLAG "")
    message(STATUS "AVX-512 disabled by user, using scalar fallback")
endif()

if(AVX512_FLAG)
    target_compile_options(barnesHut PRIVATE ${AVX512_FLAG})
endif()

target_link_libraries(barnesHut PRIVATE OpenMP::OpenMP_CXX)

# Specify the configuration file
set(CONFIG_FILE config.json)

# Add a custom command to copy the JSON file to the build directory
add_custom_command(
        TARGET barnesHut POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/${CONFIG_FILE}
        ${CMAKE_BINARY_DIR}/
)

# Copy the config directory (including scenarios.json) to the build directory
add_custom_command(
        TARGET barnesHut POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config
        ${CMAKE_BINARY_DIR}/config
)

# Display a message indicating the copy action
message(STATUS "The config.json file will be copied to the build directory.")
message(STATUS "The config directory (with scenarios.json) will be copied to the build directory.")

#####################################################################################################
# Google Test Integration for Unit Testing (only if HPCLab_BUILD_TESTS is ON)
#####################################################################################################
if(HPCLab_BUILD_TESTS)
    message(STATUS "Building with unit tests enabled")

    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Create test executable
    add_executable(testBarnesHut
        tests/BarnesHutTest.cpp
        ${LIB_SOURCES}
    )

    target_link_libraries(testBarnesHut
        GTest::gtest_main
        OpenMP::OpenMP_CXX
    )

    if(AVX512_FLAG)
        target_compile_options(testBarnesHut PRIVATE ${AVX512_FLAG})
    endif()

    include(GoogleTest)
    gtest_discover_tests(testBarnesHut)
else()
    message(STATUS "Unit tests disabled. Use -DHPCLab_BUILD_TESTS=ON to enable.")
endif()
