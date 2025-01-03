// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cmath"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector matmulnbits(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 3);
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits
    const auto inputs = node.get_ov_inputs();
    const auto& a = inputs[0];  // required
    ov::Output<ov::Node> b;
    const auto& b_quantized = inputs[1];                                                 // required
    const auto& scales = inputs[2];                                                      // required
    ov::Output<ov::Node> zero_points;                                                    // optional, input[3]
    ov::Output<ov::Node> group_idx;                                                      // optional, input[4]
    ov::Output<ov::Node> bias;                                                           // optional, input[5]
    const auto K = node.get_attribute_value<int64_t>("K");                               // required
    const auto N = node.get_attribute_value<int64_t>("N");                               // required
    const auto accuracy_level = node.get_attribute_value<int64_t>("accuracy_level", 0);  // optional, default unset(0)
    const auto block_size = node.get_attribute_value<int64_t>("block_size");             // required
    const auto bits = node.get_attribute_value<int64_t>(
        "bits",
        4);  // required, in docs: number of bits used for weight quantization (default 4)

    const uint64_t n_blocks_per_col = (K + block_size - 1) / block_size;
    const auto blob_size = static_cast<int64_t>(ceil(block_size * bits / 8));

    CHECK_VALID_NODE(node, n_blocks_per_col > 0, "Wrong blocks count: ", n_blocks_per_col);
    CHECK_VALID_NODE(node, blob_size > 0, "Wrong blob size: ", blob_size);
    // in documentation: ...Input B is a 2D constant Matrix.
    CHECK_VALID_NODE(node,
                     dynamic_cast<v0::Constant*>(b_quantized.get_node()) != nullptr,
                     "MatMulNBits limitation: accepting only a constant as a B input");
    CHECK_VALID_NODE(node,
                     b_quantized.get_partial_shape().rank() == 3,
                     "Expected rank of quantized weights is 3 [N][n_blocks_per_col][blob_size], got: ",
                     b_quantized.get_partial_shape().rank());
    CHECK_VALID_NODE(node,
                     a.get_element_type() == ov::element::f16 || a.get_element_type() == ov::element::f32 ||
                         a.get_element_type() == ov::element::dynamic,
                     "Unsupported input A type, accepted dynamic, FP16, FP32, got: ",
                     a.get_element_type());
    CHECK_VALID_NODE(
        node,
        b_quantized.get_element_type() == ov::element::u8 || b_quantized.get_element_type() == ov::element::i32,
        "Unsupported input B type, accepted FP16, FP32, got: ",
        b_quantized.get_element_type());

    CHECK_VALID_NODE(node,
                     block_size >= 16 && (block_size % 2 == 0),
                     "Wrong block size, should be >=16 and be a power of 2, got: ",
                     block_size);
    CHECK_VALID_NODE(node, accuracy_level >= 0 && accuracy_level <= 4, "Unsupported accuracy level: ", accuracy_level);

    if (inputs.size() > 3) {
        zero_points = inputs[3];
        CHECK_VALID_NODE(node,
                         zero_points.get_element_type() == ov::element::u8 ||
                             zero_points.get_element_type() == ov::element::i32 ||
                             zero_points.get_element_type() == ov::element::f32 ||
                             zero_points.get_element_type() == ov::element::f16,
                         "Unsupported input zero_points type, accepted U8, I32, FP16, FP32, got: ",
                         zero_points.get_element_type());
    }

    if (inputs.size() > 4) {
        group_idx = inputs[4];
        CHECK_VALID_NODE(node,
                         group_idx.get_element_type() == ov::element::i32,
                         "Unsupported input group_idx type, accepted I32, got: ",
                         group_idx.get_element_type());
    }

    if (inputs.size() > 5) {
        bias = inputs[5];
        CHECK_VALID_NODE(node,
                         bias.get_element_type() == a.get_element_type() ||
                             a.get_element_type() == ov::element::dynamic ||
                             bias.get_element_type() == ov::element::dynamic,
                         "Unsupported input bias type, must be equal to input A type, got: ",
                         bias.get_element_type());
        CHECK_VALID_NODE(node,
                         bias.get_partial_shape() == PartialShape{N},
                         "Wrong bias shape, expected [",
                         N,
                         "], got: ",
                         bias.get_partial_shape());
    }

    {
        const auto b_const = std::dynamic_pointer_cast<v0::Constant>(b_quantized.get_node_shared_ptr());

        ov::Output<ov::Node> casted_b;
        ov::Shape casted_b_shape;
        ov::Output<ov::Node> default_zp;
        // Casting/converting data of source constant.
        // For further calculations (sub and/or multiply) we need to reshape it from [N][n_blocks_per_col][blob_size *
        // X] to [N * n_blocks_per_col][blob_size * X] (where X is amount of values in 1 byte) because scale and
        // zero_point are represented as: ...with shape like: [N * n_blocks_per_col]...
        switch (bits) {
        case 2:
            casted_b_shape = ov::Shape{static_cast<size_t>(N),
                                       static_cast<size_t>(n_blocks_per_col),
                                       static_cast<size_t>(blob_size * 4)};
            casted_b = std::make_shared<v0::Constant>(ov::element::u2, casted_b_shape, b_const->get_data_ptr());
            if (a.get_element_type() != ov::element::dynamic) {
                default_zp = std::make_shared<v0::Constant>(a.get_element_type(), Shape{}, 2);
            } else {
                default_zp =
                    std::make_shared<v1::ConvertLike>(a,
                                                      std::make_shared<v0::Constant>(ov::element::f32, Shape{}, 2.f));
            }
            break;
        case 4:
            casted_b_shape = ov::Shape{static_cast<size_t>(N),
                                       static_cast<size_t>(n_blocks_per_col),
                                       static_cast<size_t>(blob_size * 2)};
            casted_b = std::make_shared<v0::Constant>(ov::element::u4, casted_b_shape, b_const->get_data_ptr());
            if (a.get_element_type() != ov::element::dynamic) {
                default_zp = std::make_shared<v0::Constant>(a.get_element_type(), Shape{}, 8);
            } else {
                default_zp =
                    std::make_shared<v1::ConvertLike>(a,
                                                      std::make_shared<v0::Constant>(ov::element::f32, Shape{}, 8.f));
            }
            break;
        case 8:
            casted_b_shape = ov::Shape{static_cast<size_t>(N),
                                       static_cast<size_t>(n_blocks_per_col),
                                       static_cast<size_t>(blob_size)};
            casted_b = op::util::reshape(b_const, casted_b_shape);
            if (a.get_element_type() != ov::element::dynamic) {
                default_zp = std::make_shared<v0::Constant>(a.get_element_type(), Shape{}, 128);
            } else {
                default_zp =
                    std::make_shared<v1::ConvertLike>(a,
                                                      std::make_shared<v0::Constant>(ov::element::f32, Shape{}, 128.f));
            }
            break;
        default:
            FRONT_END_THROW("Unsupported bits count");
            break;
        }

        // Possible issue with slice implementation, had to move convertion before slice, instead of slicing uint4
        // TODO: Ticket

        // TODO: convert target type according to AccuracyLevel
        const auto converted_b = std::make_shared<v0::Convert>(casted_b, ov::element::f32);

        // TODO: Need to collect performance data in case constant folding is applied. Possible some perf/mem-gap

        // TODO: shape and type of zp
        if (!zero_points.get_node_shared_ptr()) {
            zero_points = default_zp;
        }
        bool slice_needed = n_blocks_per_col * blob_size > K;

        // Simple case where we can slice before dequantization
        if (n_blocks_per_col == 1) {
            // Removing unused items in case block is bigger than column count
            // For example, if data is (uint8)[1,2,3,4,5,6] then block will be (uint8)[1,2,3,4,5,6,0,0,0,0,0,0,0,0,0,0].
            // And last zeros are unused.
            ov::Output<ov::Node> sub_b;
            if (slice_needed) {
                const auto zero_const = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 0);
                const auto one_const = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
                const auto elements_const =
                    std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, static_cast<int32_t>(K));
                const auto axis_const = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 2);
                const auto slice_b =
                    std::make_shared<v8::Slice>(converted_b, zero_const, elements_const, one_const, axis_const);

                sub_b = std::make_shared<v1::Subtract>(slice_b, zero_points);
            } else {
                sub_b = std::make_shared<v1::Subtract>(converted_b, zero_points);
            }

            const auto scales_reshaped =
                op::util::reshape(scales, ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col), 1});
            const auto scaled_b = std::make_shared<v1::Multiply>(sub_b, scales_reshaped);
            b = op::util::reshape(scaled_b, ov::Shape{static_cast<size_t>(N), static_cast<size_t>(K)});
        } else {
            const auto sub_b = std::make_shared<v1::Subtract>(converted_b, zero_points);
            const auto scales_reshaped =
                op::util::reshape(scales, ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col), 1});
            const auto scaled_b = std::make_shared<v1::Multiply>(sub_b, scales_reshaped);
            const auto reshaped_b =
                op::util::reshape(scaled_b, ov::Shape{static_cast<size_t>(N), static_cast<size_t>(K)});

            if (slice_needed) {
                const auto zero_const = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 0);
                const auto one_const = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
                const auto elements_const =
                    std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, static_cast<int32_t>(K));
                const auto axis_const = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
                b = std::make_shared<v8::Slice>(reshaped_b, zero_const, elements_const, one_const, axis_const);
            } else {
                b = reshaped_b;
            }
        }
    }

    const auto matmul = std::make_shared<v0::MatMul>(a, b, false, true);
    if (bias.get_node_shared_ptr()) {
        return {std::make_shared<v1::Add>(matmul, bias)};
    } else {
        return {matmul};
    }
}

ONNX_OP("MatMulNBits", OPSET_SINCE(1), com_microsoft::opset_1::matmulnbits, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
