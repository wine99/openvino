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
    // Pattern : x->Add(c1)->Reshape->Add(c2) where c1 and c2 is a Constant with one element and c2
    auto c1 = ov::pass::pattern::wrap_type<v0::Constant>();
    auto add1 = ov::pass::pattern::wrap_type<v1::Add>({ov::pass::pattern::any_input(), c1});
    auto reshape = ov::pass::pattern::wrap_type<v1::Reshape>({add1, ov::pass::pattern::any_input()});
    auto c2 = ov::pass::pattern::wrap_type<v0::Constant>();
    auto add2 = ov::pass::pattern::wrap_type<v1::Add>({reshape, c2});

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add2, "AddReshapeAddFusion");
    register_matcher(m, [=](ov::pass::pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto add2_node = std::dynamic_pointer_cast<v1::Add>(pattern_map.at(add2));
        auto reshape_node = std::dynamic_pointer_cast<v1::Reshape>(pattern_map.at(reshape));
        auto add1_node = std::dynamic_pointer_cast<v1::Add>(pattern_map.at(add1));
        auto c1_node = std::dynamic_pointer_cast<v0::Constant>(pattern_map.at(c1));
        auto c2_node = std::dynamic_pointer_cast<v0::Constant>(pattern_map.at(c2));

        if (ov::shape_size(c1_node->get_shape()) != 1 || ov::shape_size(c2_node->get_shape()) != 1)
            return false;

        auto x = add1_node->input_value(0);

        std::shared_ptr<v0::Constant> fused_const;
        switch (c1_node->get_element_type()) {
        case ov::element::i64: {
            int64_t val1 = *c1_node->get_data_ptr<int64_t>();
            int64_t val2 = *c2_node->get_data_ptr<int64_t>();
            int64_t fused_val = val1 + val2;
            fused_const = v0::Constant::create(c1_node->get_element_type(), c2_node->get_shape(), {fused_val});
            break;
        }
        case ov::element::i32: {
            int32_t val1 = *c1_node->get_data_ptr<int32_t>();
            int32_t val2 = *c2_node->get_data_ptr<int32_t>();
            int32_t fused_val = val1 + val2;
            fused_const = v0::Constant::create(c1_node->get_element_type(), c2_node->get_shape(), {fused_val});
            break;
        }
        case ov::element::f32: {
            float val1 = *c1_node->get_data_ptr<float>();
            float val2 = *c2_node->get_data_ptr<float>();
            float fused_val = val1 + val2;
            fused_const = v0::Constant::create(c1_node->get_element_type(), c2_node->get_shape(), {fused_val});
            break;
        }
        case ov::element::f16: {
            auto val1 = *c1_node->get_data_ptr<ov::float16>();
            auto val2 = *c2_node->get_data_ptr<ov::float16>();
            auto fused_val = val1 + val2;
            fused_const = v0::Constant::create(c1_node->get_element_type(), c2_node->get_shape(), {fused_val});
            break;
        }
        default:
            return false;
        }

        // The reshape node now takes "x" directly
        reshape_node->input(0).replace_source_output(x);

        auto fused_add = register_new_node<v1::Add>(reshape_node->output(0), fused_const);
        fused_add->set_friendly_name(add2_node->get_friendly_name());
        ov::copy_runtime_info({add2_node, add1_node, c1_node, c2_node}, fused_add);
        ov::replace_node(add2_node, fused_add);

        return true;
    });
}
