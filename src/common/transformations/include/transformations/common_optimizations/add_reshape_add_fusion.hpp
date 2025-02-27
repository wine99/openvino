// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API AddReshapeAddFusion;

}  // namespace pass
}  // namespace ov

/**
 * Pattern: x -> Add(c1) -> Reshape -> Add(c2) where c1 and c2 is a Constant with one element and c2
 * The first Add can be fused into the second one.
 */
class ov::pass::AddReshapeAddFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("AddReshapeAddFusion");
    AddReshapeAddFusion();
};
