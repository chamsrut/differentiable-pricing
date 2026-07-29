cmake_minimum_required(VERSION 3.20)

project(
    differentiable_pricing
    VERSION 0.1.0
    DESCRIPTION "Classical and neural derivative-pricing research"
    LANGUAGES CXX
)

option(DP_BUILD_PYTHON_BINDINGS "Build the Python extension" ON)
option(DP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(
    dp_core
    cpp/src/black_scholes.cpp
    cpp/src/smooth_mlp.cpp
)
add_library(dp::core ALIAS dp_core)
set_target_properties(dp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(dp_core PUBLIC cxx_std_20)
target_include_directories(
    dp_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(dp_core PRIVATE /W4 /permissive-)
    if(DP_WARNINGS_AS_ERRORS)
        target_compile_options(dp_core PRIVATE /WX)
    endif()
else()
    target_compile_options(
        dp_core
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
    )
    if(DP_WARNINGS_AS_ERRORS)
        target_compile_options(dp_core PRIVATE -Werror)
    endif()
endif()

add_executable(dp_pricer cpp/apps/pricer_cli.cpp)
target_link_libraries(dp_pricer PRIVATE dp::core)

include(CTest)
if(BUILD_TESTING)
    add_executable(dp_cpp_tests cpp/tests/test_main.cpp)
    target_link_libraries(dp_cpp_tests PRIVATE dp::core)
    add_test(NAME dp_cpp_tests COMMAND dp_cpp_tests)
endif()

if(DP_BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core MODULE bindings/python/module.cpp)
    target_link_libraries(_core PRIVATE dp::core)
    target_compile_features(_core PRIVATE cxx_std_20)
    install(TARGETS _core LIBRARY DESTINATION differentiable_pricing)
endif()

install(TARGETS dp_core EXPORT dpTargets)
install(DIRECTORY cpp/include/ DESTINATION include)
