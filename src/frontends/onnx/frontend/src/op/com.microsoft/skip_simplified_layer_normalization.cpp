// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/divide.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector skip_simplified_layer_normalization(const ov::frontend::onnx::Node& node) {
    auto nodes = node.get_ov_inputs();
    auto num_nodes = nodes.size();
    FRONT_END_GENERAL_CHECK(num_nodes >= 3 && num_nodes <= 4,
                            "SkipSimplifiedLayerNormalization takes 3 or 4 inputs. Provided " + std::to_string(num_nodes));

    // input + skip
    std::shared_ptr<ov::Node> input = std::make_shared<v1::Add>(nodes[0], nodes[1]);
    // add bias if available
    if (num_nodes == 4) {
        input = std::make_shared<v1::Add>(input, nodes[3]);
    }
    float eps = node.get_attribute_value<float>("epsilon");
    bool simplified = true;
    // reduce over hidden_size
    int hidden_size_dim = 2;
    const auto reduction_axes = v0::Constant::create(ov::element::i32, ov::Shape{1}, {hidden_size_dim});
    
    std::shared_ptr<ov::Node> mean = std::make_shared<v1::ReduceMean>(input, reduction_axes, true);
    auto sqr_const = v0::Constant::create(ov::element::f32, ov::Shape{1}, {2});
    auto square = std::make_shared<v1::Power>(input, sqr_const);
    auto mean_square = std::make_shared<v1::ReduceMean>(square, reduction_axes, true);
    auto eps_node = v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto eps_add = std::make_shared<v1::Add>(mean_square, eps_node);
    auto sqrt = std::make_shared<v0::Sqrt>(eps_add);
    std::shared_ptr<ov::Node> result = std::make_shared<v1::Divide>(input, sqrt);

    auto one_node = v0::Constant::create(ov::element::f32, ov::Shape{}, {1});
    std::shared_ptr<ov::Node> inv_std_var = std::make_shared<v1::Divide>(one_node, sqrt); 

    // multiply by gamma
    result = std::make_shared<v1::Multiply>(result, nodes[2]);

    return {result, mean, inv_std_var, input};
}
ONNX_OP("SkipSimplifiedLayerNormalization", OPSET_SINCE(1), com_microsoft::opset_1::skip_simplified_layer_normalization, MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
