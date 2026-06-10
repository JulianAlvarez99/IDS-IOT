#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t qdense_1_input[18],
    result_t layer11_out[1]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=qdense_1_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=qdense_1_input,layer11_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 576>(w2, "w2.txt");
        nnet::load_weights_from_txt<qdense_1_bias_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<batch_normalization_scale_t, 32>(s4, "s4.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 32>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 512>(w6, "w6.txt");
        nnet::load_weights_from_txt<qdense_2_bias_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<batch_normalization_1_scale_t, 16>(s8, "s8.txt");
        nnet::load_weights_from_txt<batch_normalization_1_bias_t, 16>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight10_t, 16>(w10, "w10.txt");
        nnet::load_weights_from_txt<output_bias_t, 1>(b10, "b10.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    qdense_1_result_t layer2_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    batch_normalization_result_t layer4_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    qdense_2_result_t layer6_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    batch_normalization_1_result_t layer8_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer9_t layer9_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    output_result_t layer10_out[1];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    nnet::dense<input_t, qdense_1_result_t, config2>(qdense_1_input, layer2_out, w2, b2); // qdense_1

    nnet::normalize<qdense_1_result_t, batch_normalization_result_t, config4>(layer2_out, layer4_out, s4, b4); // batch_normalization

    nnet::relu<batch_normalization_result_t, layer5_t, relu_config5>(layer4_out, layer5_out); // q_activation

    nnet::dense<layer5_t, qdense_2_result_t, config6>(layer5_out, layer6_out, w6, b6); // qdense_2

    nnet::normalize<qdense_2_result_t, batch_normalization_1_result_t, config8>(layer6_out, layer8_out, s8, b8); // batch_normalization_1

    nnet::relu<batch_normalization_1_result_t, layer9_t, relu_config9>(layer8_out, layer9_out); // q_activation_1

    nnet::dense<layer9_t, output_result_t, config10>(layer9_out, layer10_out, w10, b10); // output

    nnet::sigmoid<output_result_t, result_t, sigmoid_config11>(layer10_out, layer11_out); // output_sigmoid

}

