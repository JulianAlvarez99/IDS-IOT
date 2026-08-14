
wire kernel_monitor_reset;
wire kernel_monitor_clock;
wire kernel_monitor_report;
assign kernel_monitor_reset = ~ap_rst_n;
assign kernel_monitor_clock = ap_clk;
assign kernel_monitor_report = 1'b0;
wire [0:0] axis_block_sigs;
wire [13:0] inst_idle_sigs;
wire [11:0] inst_block_sigs;
wire kernel_block;

assign axis_block_sigs[0] = ~Loop_VITIS_LOOP_86_1_proc1_U0.input_r_TDATA_blk_n;

assign inst_idle_sigs[0] = Loop_VITIS_LOOP_86_1_proc1_U0.ap_idle;
assign inst_block_sigs[0] = (Loop_VITIS_LOOP_86_1_proc1_U0.ap_done & ~Loop_VITIS_LOOP_86_1_proc1_U0.ap_continue) | ~Loop_VITIS_LOOP_86_1_proc1_U0.features_in_data_V_blk_n;
assign inst_idle_sigs[1] = Block_inference_for_cond_i_exit_proc2_U0.ap_idle;
assign inst_block_sigs[1] = (Block_inference_for_cond_i_exit_proc2_U0.ap_done & ~Block_inference_for_cond_i_exit_proc2_U0.ap_continue) | ~Block_inference_for_cond_i_exit_proc2_U0.features_in_data_V_blk_n | ~Block_inference_for_cond_i_exit_proc2_U0.internal_input_blk_n;
assign inst_idle_sigs[2] = myproject_U0.ap_idle;
assign inst_block_sigs[2] = (myproject_U0.ap_done & ~myproject_U0.ap_continue) | ~myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.internal_input_blk_n | ~myproject_U0.sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0.internal_output_blk_n;
assign inst_idle_sigs[3] = Block_inference_for_cond_i_exit13_proc3_U0.ap_idle;
assign inst_block_sigs[3] = (Block_inference_for_cond_i_exit13_proc3_U0.ap_done & ~Block_inference_for_cond_i_exit13_proc3_U0.ap_continue) | ~Block_inference_for_cond_i_exit13_proc3_U0.internal_output_blk_n;
assign inst_idle_sigs[4] = myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.ap_idle;
assign inst_block_sigs[4] = (myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.ap_done & ~myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.ap_continue);
assign inst_idle_sigs[5] = myproject_U0.normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0.ap_idle;
assign inst_block_sigs[5] = (myproject_U0.normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0.ap_done & ~myproject_U0.normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0.ap_continue);
assign inst_idle_sigs[6] = myproject_U0.relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0.ap_idle;
assign inst_block_sigs[6] = (myproject_U0.relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0.ap_done & ~myproject_U0.relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0.ap_continue);
assign inst_idle_sigs[7] = myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.ap_idle;
assign inst_block_sigs[7] = (myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.ap_done & ~myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.ap_continue);
assign inst_idle_sigs[8] = myproject_U0.normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0.ap_idle;
assign inst_block_sigs[8] = (myproject_U0.normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0.ap_done & ~myproject_U0.normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0.ap_continue);
assign inst_idle_sigs[9] = myproject_U0.relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0.ap_idle;
assign inst_block_sigs[9] = (myproject_U0.relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0.ap_done & ~myproject_U0.relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0.ap_continue);
assign inst_idle_sigs[10] = myproject_U0.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.ap_idle;
assign inst_block_sigs[10] = (myproject_U0.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.ap_done & ~myproject_U0.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.ap_continue);
assign inst_idle_sigs[11] = myproject_U0.sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0.ap_idle;
assign inst_block_sigs[11] = (myproject_U0.sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0.ap_done & ~myproject_U0.sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0.ap_continue);

assign inst_idle_sigs[12] = 1'b0;
assign inst_idle_sigs[13] = Loop_VITIS_LOOP_86_1_proc1_U0.ap_idle;

inference_hls_deadlock_idx0_monitor inference_hls_deadlock_idx0_monitor_U (
    .clock(kernel_monitor_clock),
    .reset(kernel_monitor_reset),
    .axis_block_sigs(axis_block_sigs),
    .inst_idle_sigs(inst_idle_sigs),
    .inst_block_sigs(inst_block_sigs),
    .block(kernel_block)
);


always @ (kernel_block or kernel_monitor_reset) begin
    if (kernel_block == 1'b1 && kernel_monitor_reset == 1'b0) begin
        find_kernel_block = 1'b1;
    end
    else begin
        find_kernel_block = 1'b0;
    end
end
