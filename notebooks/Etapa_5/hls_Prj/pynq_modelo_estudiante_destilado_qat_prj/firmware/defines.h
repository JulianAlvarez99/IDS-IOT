#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 18
#define N_LAYER_2 32
#define N_LAYER_2 32
#define N_LAYER_2 32
#define N_LAYER_6 16
#define N_LAYER_6 16
#define N_LAYER_6 16
#define N_LAYER_10 1
#define N_LAYER_10 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<8,2>, 18*1> input_t;
typedef ap_fixed<8,2> model_default_t;
typedef nnet::array<ap_fixed<8,2>, 32*1> layer2_t;
typedef ap_fixed<8,3> weight2_t;
typedef ap_uint<1> layer2_index;
typedef nnet::array<ap_fixed<8,2>, 32*1> layer4_t;
typedef nnet::array<ap_fixed<8,2>, 32*1> layer5_t;
typedef ap_fixed<18,8> q_activation_table_t;
typedef nnet::array<ap_fixed<8,2>, 16*1> layer6_t;
typedef ap_fixed<8,3> weight6_t;
typedef ap_uint<1> layer6_index;
typedef nnet::array<ap_fixed<8,2>, 16*1> layer8_t;
typedef nnet::array<ap_fixed<8,2>, 16*1> layer9_t;
typedef ap_fixed<18,8> q_activation_1_table_t;
typedef nnet::array<ap_fixed<8,2>, 1*1> layer10_t;
typedef ap_fixed<8,3> weight10_t;
typedef ap_uint<1> layer10_index;
typedef nnet::array<ap_fixed<8,2>, 1*1> result_t;
typedef ap_fixed<18,8> output_sigmoid_table_t;

#endif
