cmake_minimum_required(VERSION 3.20)

project(
    differentiable_pricing
    VERSION 0.1.0
    DESCRIPTION "Classical and neural derivative-pricing research"
    LANGUAGES CXX
)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE
        RelWithDebInfo
        CACHE STRING "Choose Debug, Release, RelWithDebInfo, or MinSizeRel"
        FORCE
    )
    set_property(
        CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
    )
endif()

option(DP_BUILD_PYTHON_BINDINGS "Build the Python extension" ON)
option(DP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(Threads REQUIRED)

add_library(
    dp_core
    cpp/src/binomial_tree.cpp
    cpp/src/black_scholes.cpp
    cpp/src/least_squares_monte_carlo.cpp
    cpp/src/option.cpp
    cpp/src/smooth_mlp.cpp
)
add_library(dp::core ALIAS dp_core)
set_target_properties(dp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(dp_core PUBLIC Threads::Threads)

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

add_executable(dp_american_pricer cpp/apps/american_pricer_cli.cpp)
target_link_libraries(dp_american_pricer PRIVATE dp::core)

include(CTest)
if(BUILD_TESTING)
    add_executable(dp_cpp_tests cpp/tests/test_main.cpp)
    target_link_libraries(dp_cpp_tests PRIVATE dp::core)
    add_test(NAME dp_cpp_tests COMMAND dp_cpp_tests)
    add_test(
        NAME dp_american_cli_contract
        COMMAND
            ${CMAKE_COMMAND}
            -DPRICER=$<TARGET_FILE:dp_american_pricer>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_american_cli.cmake
    )
endif()

if(DP_BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG REQUIRED)
    set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/module.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/dp/binomial_tree.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/dp/black_scholes.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/dp/least_squares_monte_carlo.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/dp/option.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/binomial_tree.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/black_scholes.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/least_squares_monte_carlo.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/option.cpp
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/module.cpp
        DP_BINDING_SOURCE_SHA256
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/dp/binomial_tree.hpp
        DP_CRR_HEADER_SHA256
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/dp/black_scholes.hpp
        DP_BLACK_SCHOLES_HEADER_SHA256
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/dp/least_squares_monte_carlo.hpp
        DP_LSM_HEADER_SHA256
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/dp/option.hpp
        DP_OPTION_HEADER_SHA256
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/binomial_tree.cpp
        DP_CRR_SOURCE_SHA256
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/black_scholes.cpp
        DP_BLACK_SCHOLES_SOURCE_SHA256
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/least_squares_monte_carlo.cpp
        DP_LSM_SOURCE_SHA256
    )
    file(
        SHA256
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/option.cpp
        DP_OPTION_SOURCE_SHA256
    )
    string(
        CONCAT
        DP_CRR_IDENTITY_INPUT
        "${DP_BINDING_SOURCE_SHA256}:${DP_CRR_HEADER_SHA256}:"
        "${DP_BLACK_SCHOLES_HEADER_SHA256}:${DP_OPTION_HEADER_SHA256}:"
        "${DP_CRR_SOURCE_SHA256}:${DP_BLACK_SCHOLES_SOURCE_SHA256}:"
        "${DP_OPTION_SOURCE_SHA256}"
    )
    string(SHA256 DP_CRR_IMPLEMENTATION_SHA256 "${DP_CRR_IDENTITY_INPUT}")
    string(
        CONCAT
        DP_LSM_IDENTITY_INPUT
        "${DP_BINDING_SOURCE_SHA256}:${DP_LSM_HEADER_SHA256}:"
        "${DP_BLACK_SCHOLES_HEADER_SHA256}:${DP_OPTION_HEADER_SHA256}:"
        "${DP_LSM_SOURCE_SHA256}:${DP_BLACK_SCHOLES_SOURCE_SHA256}:"
        "${DP_OPTION_SOURCE_SHA256}"
    )
    string(SHA256 DP_LSM_IMPLEMENTATION_SHA256 "${DP_LSM_IDENTITY_INPUT}")
    pybind11_add_module(_core MODULE bindings/python/module.cpp)
    target_link_libraries(_core PRIVATE dp::core)
    target_compile_features(_core PRIVATE cxx_std_20)
    target_compile_definitions(
        _core
        PRIVATE
            DP_BUILD_CONFIGURATION="$<CONFIG>"
            DP_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
            DP_CRR_HEADER_SHA256="${DP_CRR_HEADER_SHA256}"
            DP_CRR_IMPLEMENTATION_SHA256="${DP_CRR_IMPLEMENTATION_SHA256}"
            DP_CRR_SOURCE_SHA256="${DP_CRR_SOURCE_SHA256}"
            DP_LSM_HEADER_SHA256="${DP_LSM_HEADER_SHA256}"
            DP_LSM_IMPLEMENTATION_SHA256="${DP_LSM_IMPLEMENTATION_SHA256}"
            DP_LSM_SOURCE_SHA256="${DP_LSM_SOURCE_SHA256}"
    )
    install(TARGETS _core LIBRARY DESTINATION differentiable_pricing)
endif()

install(TARGETS dp_core EXPORT dpTargets)
install(DIRECTORY cpp/include/ DESTINATION include)
