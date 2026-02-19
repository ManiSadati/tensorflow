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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>
#include <string>

#include "tensorflow/lite/kernels/internal/common.h"
#include "fault_injection.h"

namespace tflite {
namespace reference_integer_ops {

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.
template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const InputType* input_data, const RuntimeShape& filter_shape,
    const WeightType* filter_data, const RuntimeShape& bias_shape,
    const BiasType* bias_data, const RuntimeShape& output_shape,
    OutputType* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);


  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[out_c], output_shift[out_c]);
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
}


template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape,
                    const InputType* input_data,
                    const RuntimeShape& filter_shape,
                    const WeightType* filter_data,
                    const RuntimeShape& bias_shape, const BiasType* bias_data,
                    const RuntimeShape& output_shape, OutputType* output_data) {

  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);


  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  int cnt = 0;
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
        cnt ++;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }

  // -------------------- Fault Injection (FI) --------------------
  // FI uses a global op counter (./fi/layer_num.txt) to assign a unique "layer index"
  // to each kernel invocation during inference. Python selects the target layer by
  // writing ./fi/mode.txt with a fault_layer index. If current_layer_num == fault_layer,
  // this op becomes the target for profiling / injection.
  FaultInjection FI;
  FI.init("FullyConnected");

  // Profiling: record output tensor dimensions for this op invocation (only if this op
  // matches the requested fault_layer and mode == profiling).
  //
  // c_dim = 1 (FullyConnected output is effectively [batches, output_depth])
  // numOps is a rough count of MACs (accum_depth * batches * output_depth).
  FI.save_profile(/*c_dim=*/1, /*x_dim=*/batches, /*y_dim=*/output_depth, /*numOps=*/cnt);

  // Injection: locations.txt stores tuples (c, x, y, bit).
  //
  // We map (x, y) -> flat index in output_data:
  //   index = y + output_depth * x
  if (FI.isFaultyLayer()) {
    for (const auto& p : FI.injectLocations) {
      // out_c + output_depth * b
      int index = p.first.second.second + output_depth * p.first.second.first;  
      output_data[index] = FI.doFaultInjection(output_data[index], p);
    }
  }

  // Optional logging: dump the output tensor for this op invocation when requested.
  if(FI.isLoggedLayer()){

    // Open canonical FI output file:
    // ./fi/output_<currentLayer>-<faultLayer>-<img>-<type>-<iter>.txt
    auto out = FI.open_output_file();

    // Header required by Python reader: "<c_dim> <x_dim> <y_dim>"
    FaultInjection::write_header(out,
                                /*c_dim=*/1,
                                /*x_dim=*/batches,
                                /*y_dim=*/output_depth);
  
     
    // Write values in loop order: c -> x -> y
    for (int x = 0; x < batches; ++x) {
      for (int y = 0; y < output_depth; ++y) {
        out << static_cast<int>(output_data[y + output_depth * x]) << " ";
      }
      out << "\n";
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
