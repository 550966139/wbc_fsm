cmake_minimum_required(VERSION 3.14)
project(g1_onnx_rl LANGUAGES CXX)

include(CTest)
option(WBC_BUILD_CONTROLLER "Build the controller (requires SDK2, Eigen and ONNX Runtime)" ON)
option(WBC_BUILD_CONFIG_TESTS "Build configuration tests when nlohmann_json is available" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation for the target architecture")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WBC_BUILD_CONTROLLER)
    list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics")
    find_package(Threads REQUIRED)
    find_package(nlohmann_json 3.7.3 REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(unitree_sdk2 REQUIRED)

    # The runtime must match CMAKE_SYSTEM_PROCESSOR. Never select an x64
    # archive implicitly when building on PC2 or cross-compiling.
    if(ONNXRUNTIME_ROOT)
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            PATHS "${ONNXRUNTIME_ROOT}"
            PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
            NO_DEFAULT_PATH)
        find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
            PATHS "${ONNXRUNTIME_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
    else()
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            PATH_SUFFIXES onnxruntime onnxruntime/core/session)
        find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
    endif()
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime headers/library not found. Set ONNXRUNTIME_ROOT to an "
            "installation for ${CMAKE_SYSTEM_PROCESSOR}, or use "
            "-DWBC_BUILD_CONTROLLER=OFF for host-only regression tests.")
    endif()
    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

    file(GLOB_RECURSE SRC_LIST CONFIGURE_DEPENDS "src/*/*.cpp" "src/*/*.cc")
    add_executable(wbc_fsm src/main.cpp ${SRC_LIST})
    target_include_directories(wbc_fsm PRIVATE include)
    target_compile_definitions(wbc_fsm PRIVATE PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")
    target_link_libraries(wbc_fsm PRIVATE
        ONNXRuntime::ONNXRuntime unitree_sdk2 Threads::Threads
        nlohmann_json::nlohmann_json Eigen3::Eigen)
endif()

if(BUILD_TESTING)
    function(add_port_test target source test_name)
        add_executable(${target} ${source})
        target_include_directories(${target} PRIVATE include tests)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME ${test_name} COMMAND ${target})
    endfunction()

    add_port_test(g1_policy23_test tests/policy23_test.cpp g1_policy23_contract)
    add_port_test(g1_safety_test tests/safety_test.cpp g1_safety_regression)

    if(WBC_BUILD_CONTROLLER)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            set(FSM_TEST_SOURCES ${SRC_LIST})
            list(FILTER FSM_TEST_SOURCES EXCLUDE REGEX "/interface/")
            add_executable(g1_fsm23_test tests/fsm23_test.cpp ${FSM_TEST_SOURCES})
            target_include_directories(g1_fsm23_test PRIVATE include tests)
            target_compile_definitions(g1_fsm23_test PRIVATE PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")
            target_link_libraries(g1_fsm23_test PRIVATE ONNXRuntime::ONNXRuntime
                Threads::Threads nlohmann_json::nlohmann_json Eigen3::Eigen)
            add_test(NAME g1_onnx_and_fsm_offline
                COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/onnx_preflight_test.py
                $<TARGET_FILE:wbc_fsm> ${PROJECT_SOURCE_DIR} $<TARGET_FILE:g1_fsm23_test>)
            set_tests_properties(g1_onnx_and_fsm_offline PROPERTIES TIMEOUT 60)
        else()
            message(WARNING "Python3 not found; offline ONNX/FSM integration tests are skipped.")
        endif()
    endif()

    if(WBC_BUILD_CONFIG_TESTS)
        find_package(nlohmann_json 3.7.3 QUIET)
        if(nlohmann_json_FOUND)
            add_port_test(g1_config23_test tests/config23_test.cpp g1_config23_validation)
            target_link_libraries(g1_config23_test PRIVATE nlohmann_json::nlohmann_json)
            target_compile_definitions(g1_config23_test PRIVATE
                PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")
            find_package(Eigen3 QUIET)
            if(Eigen3_FOUND)
                add_port_test(g1_control23_test tests/control23_test.cpp g1_control23_commands)
                target_link_libraries(g1_control23_test PRIVATE nlohmann_json::nlohmann_json Eigen3::Eigen)
                target_compile_definitions(g1_control23_test PRIVATE
                    PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")
            else()
                message(WARNING "Eigen3 was not found; command integration tests are skipped.")
            endif()
        else()
            message(WARNING "nlohmann_json was not found; configuration validation tests are skipped.")
        endif()
    endif()
endif()
