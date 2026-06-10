#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<30,15> qdense_1_result_t;
typedef ap_fixed<8,3> weight2_t;
typedef ap_fixed<16,6> qdense_1_bias_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<47,22> batch_normalization_result_t;
typedef ap_fixed<16,6> batch_normalization_scale_t;
typedef ap_fixed<16,6> batch_normalization_bias_t;
typedef ap_ufixed<4,0,AP_RND_CONV,AP_SAT,0> layer5_t;
typedef ap_fixed<18,8> q_activation_table_t;
typedef ap_fixed<19,9> qdense_2_result_t;
typedef ap_fixed<8,3> weight6_t;
typedef ap_fixed<16,6> qdense_2_bias_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<36,16> batch_normalization_1_result_t;
typedef ap_fixed<16,6> batch_normalization_1_scale_t;
typedef ap_fixed<16,6> batch_normalization_1_bias_t;
typedef ap_ufixed<4,0,AP_RND_CONV,AP_SAT,0> layer9_t;
typedef ap_fixed<18,8> q_activation_1_table_t;
typedef ap_fixed<18,8> output_result_t;
typedef ap_fixed<8,3> weight10_t;
typedef ap_fixed<16,6> output_bias_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> output_sigmoid_table_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
