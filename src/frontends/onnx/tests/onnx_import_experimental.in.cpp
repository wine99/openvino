// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_simplified_layer_norm_with_scale) {
    const auto model = convert_model("experimental/simplified_layer_normalization_with_scale.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.54881352f, 0.71518934f, 0.60276335f, 0.54488319f, 0.42365479f, 0.64589411f,
                                0.43758720f, 0.89177299f, 0.96366274f, 0.38344151f, 0.79172504f, 0.52889490f,
                                0.56804454f, 0.92559665f, 0.07103606f, 0.08712930f, 0.02021840f, 0.83261985f,
                                0.77815676f, 0.87001216f, 0.97861832f, 0.79915857f, 0.46147937f, 0.78052920});
    test_case.add_expected_output<float>(
        {0.09043995f, 0.23571462f, 0.2979913f,  0.35916904f, 0.06733498f, 0.20531465f, 0.20864812f, 0.5669476f,
         0.13689055f, 0.10893753f, 0.3373992f,  0.30052305f, 0.10405416f, 0.3391008f,  0.03903707f, 0.06384123f,
         0.00282f,    0.23226243f, 0.32560462f, 0.48538631f, 0.12585294f, 0.20554786f, 0.17804245f, 0.40151258f});
    test_case.run_with_tolerance_as_fp(1e-6f);
}
