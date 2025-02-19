# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 17)

set(TARGET_NAME "openvino_gqa_extension")

# The OpenVINO installed from PyPI can be used to find OpenVINO_DIR
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
            OUTPUT_VARIABLE OpenVINO_DIR_PY
            ERROR_QUIET)
    endif()
endif()


find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")

set(SRC group_query_attention ov_extension_gqa.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

# [cmake:extension]
install(TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

