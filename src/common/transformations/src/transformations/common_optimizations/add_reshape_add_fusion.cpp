// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/add_reshape_add_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

ov::pass::AddReshapeAddFusion::AddReshapeAddFusion() {
    // Pattern : x->Add(c1)->Reshape->Add(c2) where c1 and c2 are constants with one element
    auto m_const1 = ov::pass::pattern::wrap_type<v0::Constant>();
    auto m_add1 = ov::pass::pattern::wrap_type<v1::Add>({ov::pass::pattern::any_input(), m_const1});
    auto m_reshape = ov::pass::pattern::wrap_type<v1::Reshape>({m_add1, ov::pass::pattern::any_input()});
    auto m_const2 = ov::pass::pattern::wrap_type<v0::Constant>();
    auto m_add2 = ov::pass::pattern::wrap_type<v1::Add>({m_reshape, m_const2});

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_add2, "AddReshapeAddFusion");
    register_matcher(m, [=](ov::pass::pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto add2_node = std::dynamic_pointer_cast<v1::Add>(pattern_map.at(m_add2));
        auto reshape_node = std::dynamic_pointer_cast<v1::Reshape>(pattern_map.at(m_reshape));
        auto add1_node = std::dynamic_pointer_cast<v1::Add>(pattern_map.at(m_add1));
        auto const1_node = std::dynamic_pointer_cast<v0::Constant>(pattern_map.at(m_const1));
        auto const2_node = std::dynamic_pointer_cast<v0::Constant>(pattern_map.at(m_const2));

        if (const1_node->is_dynamic() || const2_node->is_dynamic())
            return false;
        if (ov::shape_size(const1_node->get_shape()) != 1 || ov::shape_size(const2_node->get_shape()) != 1)
            return false;

        auto fused_const = op::util::eltwise_fold<ov::op::v1::Add>(const1_node, const2_node);
        auto shape_const =
            ov::op::v0::Constant::create(ov::element::i64, {const2_node->get_shape().size()}, const2_node->get_shape());
        auto fused_const_reshape = register_new_node<v1::Reshape>(fused_const, shape_const, false);
        ov::OutputVector fused_const_reshape_output(fused_const_reshape->get_output_size());
        OPENVINO_ASSERT(fused_const_reshape->constant_fold(fused_const_reshape_output, {fused_const, shape_const}),
                        "Can not constant fold");

        auto x = add1_node->input_value(0);
        // The reshape node now takes "x" directly
        reshape_node->input(0).replace_source_output(x);

        auto fused_add = register_new_node<v1::Add>(reshape_node->output(0), fused_const_reshape_output[0]);
        fused_add->set_friendly_name(add2_node->get_friendly_name());
        ov::copy_runtime_info({add2_node, add1_node, const1_node, const2_node}, fused_add);
        ov::replace_node(add2_node, fused_add);

        return true;
    });
}
