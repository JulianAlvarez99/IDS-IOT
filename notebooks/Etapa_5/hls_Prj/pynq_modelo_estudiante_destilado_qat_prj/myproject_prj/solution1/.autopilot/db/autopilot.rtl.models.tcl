set SynModuleInfo {
  {SRCNAME Loop_VITIS_LOOP_86_1_proc1 MODELNAME Loop_VITIS_LOOP_86_1_proc1 RTLNAME inference_Loop_VITIS_LOOP_86_1_proc1
    SUBMODULES {
      {MODELNAME inference_fpext_32ns_64_3_no_dsp_1 RTLNAME inference_fpext_32ns_64_3_no_dsp_1 BINDTYPE op TYPE fpext IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME inference_ashr_54ns_32ns_54_2_1 RTLNAME inference_ashr_54ns_32ns_54_2_1 BINDTYPE op TYPE ashr IMPL auto_pipe LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME inference_regslice_both RTLNAME inference_regslice_both BINDTYPE interface TYPE interface_regslice INSTNAME inference_regslice_both_U}
      {MODELNAME inference_flow_control_loop_pipe RTLNAME inference_flow_control_loop_pipe BINDTYPE interface TYPE internal_upc_flow_control INSTNAME inference_flow_control_loop_pipe_U}
    }
  }
  {SRCNAME Block_inference_for.cond.i.exit_proc2 MODELNAME Block_inference_for_cond_i_exit_proc2 RTLNAME inference_Block_inference_for_cond_i_exit_proc2}
  {SRCNAME dense<array<ap_fixed,18u>,array<ap_fixed<8,2,5,3,0>,32u>,config2> MODELNAME dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s RTLNAME inference_dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s
    SUBMODULES {
      {MODELNAME inference_mux_185_8_1_1 RTLNAME inference_mux_185_8_1_1 BINDTYPE op TYPE mux IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME inference_mul_8s_8s_13_1_1 RTLNAME inference_mul_8s_8s_13_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME inference_mux_42_8_1_1 RTLNAME inference_mux_42_8_1_1 BINDTYPE op TYPE mux IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME inference_dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s_outidx_1_ROM_bkb RTLNAME inference_dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s_outidx_1_ROM_bkb BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME inference_dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s_w2_V_ROM_AUTOcud RTLNAME inference_dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s_w2_V_ROM_AUTOcud BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME normalize<array<ap_fixed,32u>,array<ap_fixed<8,2,5,3,0>,32u>,config4> MODELNAME normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_s RTLNAME inference_normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_s
    SUBMODULES {
      {MODELNAME inference_mul_8s_7ns_14_1_1 RTLNAME inference_mul_8s_7ns_14_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME relu<array<ap_fixed,32u>,array<ap_fixed<8,2,5,3,0>,32u>,relu_config5> MODELNAME relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_s RTLNAME inference_relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_s}
  {SRCNAME dense<array<ap_fixed,32u>,array<ap_fixed<8,2,5,3,0>,16u>,config6> MODELNAME dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s RTLNAME inference_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s
    SUBMODULES {
      {MODELNAME inference_mux_325_8_1_1 RTLNAME inference_mux_325_8_1_1 BINDTYPE op TYPE mux IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME inference_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s_outidx_ROM_AUdEe RTLNAME inference_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s_outidx_ROM_AUdEe BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME inference_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s_w6_V_ROM_AUTOeOg RTLNAME inference_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s_w6_V_ROM_AUTOeOg BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME normalize<array<ap_fixed,16u>,array<ap_fixed<8,2,5,3,0>,16u>,config8> MODELNAME normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_s RTLNAME inference_normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_s
    SUBMODULES {
      {MODELNAME inference_mul_8s_8ns_14_1_1 RTLNAME inference_mul_8s_8ns_14_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME relu<array<ap_fixed,16u>,array<ap_fixed<8,2,5,3,0>,16u>,relu_config9> MODELNAME relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_s RTLNAME inference_relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_s}
  {SRCNAME dense<array<ap_fixed,16u>,array<ap_fixed<8,2,5,3,0>,1u>,config10> MODELNAME dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_s RTLNAME inference_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_s
    SUBMODULES {
      {MODELNAME inference_mux_164_8_1_1 RTLNAME inference_mux_164_8_1_1 BINDTYPE op TYPE mux IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME inference_mul_8s_6s_13_1_1 RTLNAME inference_mul_8s_6s_13_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME inference_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_s_w10_V_ROM_AUTfYi RTLNAME inference_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_s_w10_V_ROM_AUTfYi BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME sigmoid<array,array<ap_fixed<8,2,5,3,0>,1u>,sigmoid_config11> MODELNAME sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_s RTLNAME inference_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_s
    SUBMODULES {
      {MODELNAME inference_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_s_sigmoid_table_ROg8j RTLNAME inference_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_s_sigmoid_table_ROg8j BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME myproject MODELNAME myproject RTLNAME inference_myproject
    SUBMODULES {
      {MODELNAME inference_fifo_w256_d1_S RTLNAME inference_fifo_w256_d1_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME layer2_out_U}
      {MODELNAME inference_fifo_w256_d1_S RTLNAME inference_fifo_w256_d1_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME layer4_out_U}
      {MODELNAME inference_fifo_w256_d1_S RTLNAME inference_fifo_w256_d1_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME layer5_out_U}
      {MODELNAME inference_fifo_w128_d1_S RTLNAME inference_fifo_w128_d1_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME layer6_out_U}
      {MODELNAME inference_fifo_w128_d1_S RTLNAME inference_fifo_w128_d1_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME layer8_out_U}
      {MODELNAME inference_fifo_w128_d1_S RTLNAME inference_fifo_w128_d1_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME layer9_out_U}
      {MODELNAME inference_fifo_w8_d1_S RTLNAME inference_fifo_w8_d1_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME layer10_out_U}
      {MODELNAME inference_start_for_normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0 RTLNAME inference_start_for_normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0_U}
      {MODELNAME inference_start_for_relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0 RTLNAME inference_start_for_relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0_U}
      {MODELNAME inference_start_for_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0 RTLNAME inference_start_for_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0_U}
      {MODELNAME inference_start_for_normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0 RTLNAME inference_start_for_normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0_U}
      {MODELNAME inference_start_for_relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0 RTLNAME inference_start_for_relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0_U}
      {MODELNAME inference_start_for_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0 RTLNAME inference_start_for_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0_U}
      {MODELNAME inference_start_for_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0 RTLNAME inference_start_for_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0_U}
    }
  }
  {SRCNAME Block_inference_for.cond.i.exit13_proc3 MODELNAME Block_inference_for_cond_i_exit13_proc3 RTLNAME inference_Block_inference_for_cond_i_exit13_proc3
    SUBMODULES {
      {MODELNAME inference_dcmp_64ns_64ns_1_4_no_dsp_1 RTLNAME inference_dcmp_64ns_64ns_1_4_no_dsp_1 BINDTYPE op TYPE dcmp IMPL auto LATENCY 3 ALLOW_PRAGMA 1}
      {MODELNAME inference_lshr_64ns_32ns_64_2_1 RTLNAME inference_lshr_64ns_32ns_64_2_1 BINDTYPE op TYPE lshr IMPL auto_pipe LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME inference_shl_64ns_32ns_64_2_1 RTLNAME inference_shl_64ns_32ns_64_2_1 BINDTYPE op TYPE shl IMPL auto_pipe LATENCY 1 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME inference MODELNAME inference RTLNAME inference IS_TOP 1
    SUBMODULES {
      {MODELNAME inference_fifo_w8_d18_S RTLNAME inference_fifo_w8_d18_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME features_in_data_V_U}
      {MODELNAME inference_fifo_w144_d2_S RTLNAME inference_fifo_w144_d2_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME internal_input_U}
      {MODELNAME inference_fifo_w8_d2_S RTLNAME inference_fifo_w8_d2_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME internal_output_U}
      {MODELNAME inference_start_for_Block_inference_for_cond_i_exit_proc2_U0 RTLNAME inference_start_for_Block_inference_for_cond_i_exit_proc2_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_Block_inference_for_cond_i_exit_proc2_U0_U}
      {MODELNAME inference_start_for_myproject_U0 RTLNAME inference_start_for_myproject_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_myproject_U0_U}
      {MODELNAME inference_start_for_Block_inference_for_cond_i_exit13_proc3_U0 RTLNAME inference_start_for_Block_inference_for_cond_i_exit13_proc3_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_Block_inference_for_cond_i_exit13_proc3_U0_U}
    }
  }
}
