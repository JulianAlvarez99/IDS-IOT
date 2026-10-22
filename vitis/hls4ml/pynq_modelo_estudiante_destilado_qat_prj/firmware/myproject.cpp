#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &qdense_1_input,
    hls::stream<result_t> &layer11_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=qdense_1_input,layer11_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
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
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=1
    nnet::dense<input_t, layer2_t, config2>(qdense_1_input, layer2_out, w2, b2); // qdense_1

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=1
    nnet::normalize<layer2_t, layer4_t, config4>(layer2_out, layer4_out, s4, b4); // batch_normalization

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=1
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // q_activation

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=1
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // qdense_2

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1
    nnet::normalize<layer6_t, layer8_t, config8>(layer6_out, layer8_out, s8, b8); // batch_normalization_1

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=1
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // q_activation_1

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // output

    nnet::sigmoid<layer10_t, result_t, sigmoid_config11>(layer10_out, layer11_out); // output_sigmoid

}
