/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MUL_H_

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include "fault_injection.h"

namespace tflite {
namespace reference_integer_ops {

// Maximum dimension supported by the broadcast mul operation.
constexpr int kMaxMulBroadcastDim = 6;

template <typename InputType, typename OutputType>
void MulElementwise(int size, const ArithmeticParams& params,
                    const InputType* input1_data, const InputType* input2_data,
                    OutputType* output_data) {

  for (int i = 0; i < size; ++i) {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<OutputType>(clamped_output);
  }
}

template <typename T>
inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("Mul/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

// Mul with 16 bit inputs and int8_t outputs.
inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16_t* input1_data,
                const RuntimeShape& input2_shape, const int16_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  ruy::profiler::ScopeLabel label("Mul/Int16Int8");
  int32_t output_offset = params.output_offset;
  int32_t output_activation_min = params.quantized_activation_min;
  int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    int16_t rescaled_result =
        gemmlowp::RoundingDivideByPOT(unclamped_result.raw(), 8);
    int16_t clamped_result = std::min<int16_t>(
        output_activation_max - output_offset, rescaled_result);
    clamped_result = std::max<int16_t>(output_activation_min - output_offset,
                                       clamped_result);
    output_data[i] = output_offset + clamped_result;
  }
}

template <typename T>
inline void BroadcastMul6DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastMul6DSlow");
  NdArrayDesc<kMaxMulBroadcastDim> desc1;
  NdArrayDesc<kMaxMulBroadcastDim> desc2;
  // The input shapes are extended as part of NdArrayDesc initialization.
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(kMaxMulBroadcastDim, output_shape);
  // Cache output shape dimensions.
  int32_t extended_output_shape_dims[kMaxMulBroadcastDim];
  std::memcpy(extended_output_shape_dims, extended_output_shape.DimsData(),
              sizeof(extended_output_shape_dims));

  size_t input1_offset_a = 0;
  size_t input2_offset_a = 0;
  size_t output_offset_a = 0;
  int cnt = 0;
  for (int a = 0; a < extended_output_shape_dims[0]; ++a) {
    size_t input1_offset_d = input1_offset_a;
    size_t input2_offset_d = input2_offset_a;
    size_t output_offset_d = output_offset_a;
    for (int d = 0; d < extended_output_shape_dims[1]; ++d) {
      size_t input1_offset_b = input1_offset_d;
      size_t input2_offset_b = input2_offset_d;
      size_t output_offset_b = output_offset_d;
      for (int b = 0; b < extended_output_shape_dims[2]; ++b) {
        size_t input1_offset_y = input1_offset_b;
        size_t input2_offset_y = input2_offset_b;
        size_t output_offset_y = output_offset_b;
        for (int y = 0; y < extended_output_shape_dims[3]; ++y) {
          size_t input1_offset_x = input1_offset_y;
          size_t input2_offset_x = input2_offset_y;
          size_t output_offset_x = output_offset_y;
          for (int x = 0; x < extended_output_shape_dims[4]; ++x) {
            size_t input1_offset_c = input1_offset_x;
            size_t input2_offset_c = input2_offset_x;
            size_t output_offset_c = output_offset_x;
            for (int c = 0; c < extended_output_shape_dims[5]; ++c) {
              const int32_t input1_val =
                  params.input1_offset + input1_data[input1_offset_c];
              const int32_t input2_val =
                  params.input2_offset + input2_data[input2_offset_c];
              const int32_t unclamped_result =
                  params.output_offset +
                  MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                                params.output_multiplier,
                                                params.output_shift);
              cnt ++;
              const int32_t clamped_output = std::min(
                  params.quantized_activation_max,
                  std::max(params.quantized_activation_min, unclamped_result));
              output_data[output_offset_c] = static_cast<T>(clamped_output);
              input1_offset_c += desc1.strides[5];
              input2_offset_c += desc2.strides[5];
              ++output_offset_c;
            }
            input1_offset_x += desc1.strides[4];
            input2_offset_x += desc2.strides[4];
            output_offset_x += extended_output_shape_dims[5];
          }
          input1_offset_y += desc1.strides[3];
          input2_offset_y += desc2.strides[3];
          output_offset_y +=
              extended_output_shape_dims[4] * extended_output_shape_dims[5];
        }
        input1_offset_b += desc1.strides[2];
        input2_offset_b += desc2.strides[2];
        output_offset_b += extended_output_shape_dims[3] *
                           extended_output_shape_dims[4] *
                           extended_output_shape_dims[5];
      }
      input1_offset_d += desc1.strides[1];
      input2_offset_d += desc2.strides[1];
      output_offset_d +=
          extended_output_shape_dims[2] * extended_output_shape_dims[3] *
          extended_output_shape_dims[4] * extended_output_shape_dims[5];
    }
    input1_offset_a += desc1.strides[0];
    input2_offset_a += desc2.strides[0];
    output_offset_a +=
        extended_output_shape_dims[1] * extended_output_shape_dims[2] *
        extended_output_shape_dims[3] * extended_output_shape_dims[4] *
        extended_output_shape_dims[5];
  }


  // extended_output_shape is expected to be 6D.
  // The first four dims are a broadcast "prefix" that should flatten to 1 element.
  // The last two dims (Dims(4), Dims(5)) form the 2D logical output we log/inject into.
  assert(extended_output_shape.DimensionsCount() == 6);
  assert(extended_output_shape.Dims(0) + extended_output_shape.Dims(1) +
            extended_output_shape.Dims(2) + extended_output_shape.Dims(3) ==
        4);  // NOTE: this implies each of the first 4 dims is 1.

  // Define logical 2D view (x_dim, y_dim) over output.
  const int x_dim = extended_output_shape.Dims(4);
  const int y_dim = extended_output_shape.Dims(5);

  // -------------------- Fault Injection (FI) --------------------
  // Convention for BroadcastMul6DSlow output (batch/prefix collapsed):
  //   c_dim = 1
  //   x = [0 .. x_dim-1]
  //   y = [0 .. y_dim-1]
  // locations.txt lines: "c x y bit" where c must be 0.
  FaultInjection FI;
  FI.init("BroadcastMul6DSlow");
  FI.save_profile(/*c_dim=*/1, /*x_dim=*/x_dim, /*y_dim=*/y_dim, /*numOps=*/cnt);

  if (FI.isFaultyLayer()) {
    for (const auto& p : FI.injectLocations) {
      const int c = p.first.first;          // must be 0 (since c_dim=1)
      const int x = p.first.second.first;   // Dims(4)
      const int y = p.first.second.second;  // Dims(5)

      assert(c==0);

      // Flatten (x,y) into the linear output buffer (row-major with stride y_dim).
      const int index = y + y_dim * x;

      // No explicit cast: keep old behavior (implicit conversion back to element type).
      output_data[index] = FI.doFaultInjection(output_data[index], p);
    }
  }

  if (FI.isLoggedLayer()) {
    auto out = FI.open_output_file();

    // Header required by Python reader: "<c_dim> <x_dim> <y_dim>"
    FaultInjection::write_header(out, /*c_dim=*/1, /*x_dim=*/x_dim, /*y_dim=*/y_dim);

    // Write values for a (1, x_dim, y_dim) tensor.
    // We omit c loop since c_dim == 1.
    for (int x = 0; x < x_dim; ++x) {
      for (int y = 0; y < y_dim; ++y) {
        out << static_cast<int>(output_data[y + y_dim * x]) << " ";
      }
      out << "\n";
    }
  }
  // ------------------ End Fault Injection (FI) ------------------
  
}

template <typename T>
inline void BroadcastMul4DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  BroadcastMul6DSlow(params, input1_shape, input1_data, input2_shape,
                     input2_data, output_shape, output_data);
}

}  // namespace reference_integer_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MUL_H_
