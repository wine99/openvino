// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
using namespace ov::op;
using namespace ov::op::v0;
using namespace ov::op::v1;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector simplified_layer_normalization(const ov::frontend::onnx::Node& node) {
    // TODO checks input output axes
    // TODO why add bias to input at the begining in skip_layer_norm.cpp?
    // TODO stash_type?
    // TODO ov::Output vs std::shard_ptr

    auto nodes = node.get_ov_inputs();
    auto num_nodes = nodes.size();
    int hidden_size_dim = 2;

    auto input = nodes.at(0);
    auto scale = nodes.at(1);

    auto input_squared =
        std::make_shared<v1::Power>(input, v0::Constant::create(input.get_element_type(), {}, {2}));  // X^2
    auto var = std::make_shared<v1::ReduceMean>(input_squared,
                                                Constant::create(ov::element::i32, ov::Shape{1}, {hidden_size_dim}),
                                                true);
    auto var_eps = std::make_shared<Add>(
        var,
        v0::Constant::create(ov::element::f32, ov::Shape{}, {node.get_attribute_value<float>("epsilon")}));
    auto std_dev = std::make_shared<Sqrt>(var_eps);
    auto inv_std_dev = std::make_shared<Divide>(v0::Constant::create(input.get_element_type(), {}, {1.0}), std_dev);
    auto normalized = std::make_shared<Multiply>(input, inv_std_dev);

    std::shared_ptr<ov::Node> result = std::make_shared<Multiply>(normalized, scale);
    if (num_nodes > 2) {
        auto bias = nodes.at(2);
        result = std::make_shared<Add>(result, bias);
    }

    return result->outputs();
}

ONNX_OP("SimplifiedLayerNormalization", OPSET_SINCE(1), ai_onnx::opset_1::simplified_layer_normalization);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
