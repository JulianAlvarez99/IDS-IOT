#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "defines.h"
#include <ap_axi_sdata.h>

// Definición de la estructura AXI Stream de 32 bits
typedef ap_axis<32, 2, 5, 6> axis_int_t;

// Prototipo original de hls4ml
void myproject(
    hls::stream<input_t> &qdense_1_input,
    hls::stream<result_t> &layer11_out
);

// NUEVO: Prototipo de la función Top Level para la FPGA
void inference(hls::stream<axis_int_t>& input, int *result);

#endif
