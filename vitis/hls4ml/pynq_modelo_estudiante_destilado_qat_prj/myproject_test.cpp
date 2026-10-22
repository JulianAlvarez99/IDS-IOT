#include <iostream>
#include "firmware/myproject.h"

int main() {
    // Vector de prueba emulando un paquete de red (18 características extraídas con SHAP)
    // Valores de ejemplo. Puedes poner los valores normalizados de una fila de tu dataset.
    float network_packet[N_INPUT_1_1] = {
        0.15, 0.02, 1.34, 0.00, 0.88, 0.12, 0.00, 0.00, 1.05,
        2.13, 0.01, 0.00, 0.00, 0.45, 0.00, 1.12, 0.33, 0.05
    };

    hls::stream<axis_int_t> in_stream;
    int inferenceResult;

    // Cargar el stream simulando el envío del DMA desde Python
    for(int i = 0; i < N_INPUT_1_1; i++) {
        axis_int_t val;

        union {
            unsigned int i;
            float f;
        } converter;

        converter.f = network_packet[i];
        val.data = converter.i;

        val.last = (i == N_INPUT_1_1 - 1) ? 1 : 0;
        val.keep = 1; val.strb = 1; val.user = 1; val.id = 0; val.dest = 0;

        in_stream << val;
    }

    std::cout << "--- Iniciando Inferencia de Tráfico IoT ---" << std::endl;

    // Ejecutar Bloque IP
    inference(in_stream, &inferenceResult);

    std::cout << "Predicción del Modelo: ";
    if(inferenceResult == 1) {
        std::cout << "[1] ALERTA: Tráfico Malicioso Detectado!" << std::endl;
    } else {
        std::cout << "[0] Tráfico Legítimo." << std::endl;
    }
    std::cout << "-------------------------------------------" << std::endl;

    return 0;
}
