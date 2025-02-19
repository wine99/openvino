// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "group_query_attention.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({

        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<ov::op::GroupQueryAttention>>(),

        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<ov::op::GroupQueryAttention>>()
    }));
//! [ov_extension:entry_point]
// clang-format on
