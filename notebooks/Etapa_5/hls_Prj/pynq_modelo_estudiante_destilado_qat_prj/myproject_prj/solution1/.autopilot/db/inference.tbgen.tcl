set moduleName inference
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set C_modelName {inference}
set C_modelType { void 0 }
set C_modelArgList {
	{ input_r_V_data_V int 32 regular {axi_s 0 volatile  { input_r Data } }  }
	{ input_r_V_keep_V int 4 regular {axi_s 0 volatile  { input_r Keep } }  }
	{ input_r_V_strb_V int 4 regular {axi_s 0 volatile  { input_r Strb } }  }
	{ input_r_V_user_V int 2 regular {axi_s 0 volatile  { input_r User } }  }
	{ input_r_V_last_V int 1 regular {axi_s 0 volatile  { input_r Last } }  }
	{ input_r_V_id_V int 5 regular {axi_s 0 volatile  { input_r ID } }  }
	{ input_r_V_dest_V int 6 regular {axi_s 0 volatile  { input_r Dest } }  }
	{ result int 32 regular {pointer 1}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "input_r_V_data_V", "interface" : "axis", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_keep_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_strb_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_user_V", "interface" : "axis", "bitwidth" : 2, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_id_V", "interface" : "axis", "bitwidth" : 5, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_dest_V", "interface" : "axis", "bitwidth" : 6, "direction" : "READONLY"} , 
 	{ "Name" : "result", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 17
set portList { 
	{ input_r_TDATA sc_in sc_lv 32 signal 0 } 
	{ input_r_TKEEP sc_in sc_lv 4 signal 1 } 
	{ input_r_TSTRB sc_in sc_lv 4 signal 2 } 
	{ input_r_TUSER sc_in sc_lv 2 signal 3 } 
	{ input_r_TLAST sc_in sc_lv 1 signal 4 } 
	{ input_r_TID sc_in sc_lv 5 signal 5 } 
	{ input_r_TDEST sc_in sc_lv 6 signal 6 } 
	{ result sc_out sc_lv 32 signal 7 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ input_r_TVALID sc_in sc_logic 1 invld 6 } 
	{ input_r_TREADY sc_out sc_logic 1 inacc 6 } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ result_ap_vld sc_out sc_logic 1 outvld 7 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
}
set NewPortList {[ 
	{ "name": "input_r_TDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "input_r_V_data_V", "role": "default" }} , 
 	{ "name": "input_r_TKEEP", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "input_r_V_keep_V", "role": "default" }} , 
 	{ "name": "input_r_TSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "input_r_V_strb_V", "role": "default" }} , 
 	{ "name": "input_r_TUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "input_r_V_user_V", "role": "default" }} , 
 	{ "name": "input_r_TLAST", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "input_r_V_last_V", "role": "default" }} , 
 	{ "name": "input_r_TID", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "input_r_V_id_V", "role": "default" }} , 
 	{ "name": "input_r_TDEST", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "input_r_V_dest_V", "role": "default" }} , 
 	{ "name": "result", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "result", "role": "default" }} , 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "input_r_TVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "input_r_V_dest_V", "role": "default" }} , 
 	{ "name": "input_r_TREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "inacc", "bundle":{"name": "input_r_V_dest_V", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "result_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "result", "role": "ap_vld" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "12", "13", "72", "76", "77", "78", "79", "80", "81"],
		"CDFG" : "inference",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "113", "EstimateLatencyMax" : "114",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"InputProcess" : [
			{"ID" : "1", "Name" : "Loop_VITIS_LOOP_86_1_proc1_U0"}],
		"OutputProcess" : [
			{"ID" : "72", "Name" : "Block_inference_for_cond_i_exit13_proc3_U0"}],
		"Port" : [
			{"Name" : "input_r_V_data_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "Loop_VITIS_LOOP_86_1_proc1_U0", "Port" : "input_r_V_data_V"}]},
			{"Name" : "input_r_V_keep_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "Loop_VITIS_LOOP_86_1_proc1_U0", "Port" : "input_r_V_keep_V"}]},
			{"Name" : "input_r_V_strb_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "Loop_VITIS_LOOP_86_1_proc1_U0", "Port" : "input_r_V_strb_V"}]},
			{"Name" : "input_r_V_user_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "Loop_VITIS_LOOP_86_1_proc1_U0", "Port" : "input_r_V_user_V"}]},
			{"Name" : "input_r_V_last_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "Loop_VITIS_LOOP_86_1_proc1_U0", "Port" : "input_r_V_last_V"}]},
			{"Name" : "input_r_V_id_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "Loop_VITIS_LOOP_86_1_proc1_U0", "Port" : "input_r_V_id_V"}]},
			{"Name" : "input_r_V_dest_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "Loop_VITIS_LOOP_86_1_proc1_U0", "Port" : "input_r_V_dest_V"}]},
			{"Name" : "result", "Type" : "Vld", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "72", "SubInstance" : "Block_inference_for_cond_i_exit13_proc3_U0", "Port" : "result"}]},
			{"Name" : "outidx_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "myproject_U0", "Port" : "outidx_1"}]},
			{"Name" : "w2_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "myproject_U0", "Port" : "w2_V"}]},
			{"Name" : "outidx", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "myproject_U0", "Port" : "outidx"}]},
			{"Name" : "w6_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "myproject_U0", "Port" : "w6_V"}]},
			{"Name" : "w10_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "myproject_U0", "Port" : "w10_V"}]},
			{"Name" : "sigmoid_table", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "myproject_U0", "Port" : "sigmoid_table"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
		"CDFG" : "Loop_VITIS_LOOP_86_1_proc1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "26", "EstimateLatencyMax" : "26",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "features_in_data_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["12"], "DependentChan" : "76", "DependentChanDepth" : "18", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "features_in_data_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "input_r_V_data_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r",
				"BlockSignal" : [
					{"Name" : "input_r_TDATA_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "input_r_V_keep_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r"},
			{"Name" : "input_r_V_strb_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r"},
			{"Name" : "input_r_V_user_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r"},
			{"Name" : "input_r_V_last_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r"},
			{"Name" : "input_r_V_id_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r"},
			{"Name" : "input_r_V_dest_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "input_r"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_86_1", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter7", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter7", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "1"}}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.fpext_32ns_64_3_no_dsp_1_U1", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.ashr_54ns_32ns_54_2_1_U2", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.flow_control_loop_pipe_U", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.regslice_both_input_r_V_data_V_U", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.regslice_both_input_r_V_keep_V_U", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.regslice_both_input_r_V_strb_V_U", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.regslice_both_input_r_V_user_V_U", "Parent" : "1"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.regslice_both_input_r_V_last_V_U", "Parent" : "1"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.regslice_both_input_r_V_id_V_U", "Parent" : "1"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Loop_VITIS_LOOP_86_1_proc1_U0.regslice_both_input_r_V_dest_V_U", "Parent" : "1"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Block_inference_for_cond_i_exit_proc2_U0", "Parent" : "0",
		"CDFG" : "Block_inference_for_cond_i_exit_proc2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "17", "EstimateLatencyMax" : "17",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "1",
		"StartFifo" : "start_for_Block_inference_for_cond_i_exit_proc2_U0_U",
		"Port" : [
			{"Name" : "features_in_data_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["1"], "DependentChan" : "76", "DependentChanDepth" : "18", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "features_in_data_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "internal_input", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["13","14"], "DependentChan" : "77", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "internal_input_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.myproject_U0", "Parent" : "0", "Child" : ["14", "34", "36", "37", "49", "51", "52", "56", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71"],
		"CDFG" : "myproject",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "87", "EstimateLatencyMax" : "88",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "12",
		"StartFifo" : "start_for_myproject_U0_U",
		"InputProcess" : [
			{"ID" : "14", "Name" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0"}],
		"OutputProcess" : [
			{"ID" : "56", "Name" : "sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0"}],
		"Port" : [
			{"Name" : "internal_input", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["12"], "DependentChan" : "77", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0", "Port" : "internal_input"}]},
			{"Name" : "internal_output", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["72"], "DependentChan" : "78", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "56", "SubInstance" : "sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0", "Port" : "internal_output"}]},
			{"Name" : "outidx_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0", "Port" : "outidx_1"}]},
			{"Name" : "w2_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0", "Port" : "w2_V"}]},
			{"Name" : "outidx", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "37", "SubInstance" : "dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0", "Port" : "outidx"}]},
			{"Name" : "w6_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "37", "SubInstance" : "dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0", "Port" : "w6_V"}]},
			{"Name" : "w10_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "52", "SubInstance" : "dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0", "Port" : "w10_V"}]},
			{"Name" : "sigmoid_table", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "56", "SubInstance" : "sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0", "Port" : "sigmoid_table"}]}]},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0", "Parent" : "13", "Child" : ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"],
		"CDFG" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "Rewind", "UnalignedPipeline" : "0", "RewindPipeline" : "1", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "75", "EstimateLatencyMax" : "76",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "internal_input", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["12"], "DependentChan" : "77", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "internal_input_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer2_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["34"], "DependentChan" : "58", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer2_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "outidx_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "w2_V", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "ReuseLoop", "PipelineType" : "rewind",
				"LoopDec" : {"FSMBitwidth" : "2", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter4", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter4", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "has_continue" : "1"}}]},
	{"ID" : "15", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.outidx_1_U", "Parent" : "14"},
	{"ID" : "16", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.w2_V_U", "Parent" : "14"},
	{"ID" : "17", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_185_8_1_1_U22", "Parent" : "14"},
	{"ID" : "18", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U23", "Parent" : "14"},
	{"ID" : "19", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U24", "Parent" : "14"},
	{"ID" : "20", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U25", "Parent" : "14"},
	{"ID" : "21", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U26", "Parent" : "14"},
	{"ID" : "22", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U27", "Parent" : "14"},
	{"ID" : "23", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U28", "Parent" : "14"},
	{"ID" : "24", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U29", "Parent" : "14"},
	{"ID" : "25", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U30", "Parent" : "14"},
	{"ID" : "26", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U31", "Parent" : "14"},
	{"ID" : "27", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U32", "Parent" : "14"},
	{"ID" : "28", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U33", "Parent" : "14"},
	{"ID" : "29", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U34", "Parent" : "14"},
	{"ID" : "30", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U35", "Parent" : "14"},
	{"ID" : "31", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U36", "Parent" : "14"},
	{"ID" : "32", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U37", "Parent" : "14"},
	{"ID" : "33", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U38", "Parent" : "14"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0", "Parent" : "13", "Child" : ["35"],
		"CDFG" : "normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "12", "EstimateLatencyMax" : "12",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "14",
		"StartFifo" : "start_for_normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0_U",
		"Port" : [
			{"Name" : "layer2_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["14"], "DependentChan" : "58", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer2_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer4_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["36"], "DependentChan" : "59", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer4_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "35", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0.mul_8s_7ns_14_1_1_U46", "Parent" : "34"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0", "Parent" : "13",
		"CDFG" : "relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "34",
		"StartFifo" : "start_for_relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0_U",
		"Port" : [
			{"Name" : "layer4_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["34"], "DependentChan" : "59", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer4_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer5_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["37"], "DependentChan" : "60", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer5_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0", "Parent" : "13", "Child" : ["38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48"],
		"CDFG" : "dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "Rewind", "UnalignedPipeline" : "0", "RewindPipeline" : "1", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "68", "EstimateLatencyMax" : "69",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0_U",
		"Port" : [
			{"Name" : "layer5_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["36"], "DependentChan" : "60", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer5_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer6_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["49"], "DependentChan" : "61", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer6_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "outidx", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "w6_V", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "ReuseLoop", "PipelineType" : "rewind",
				"LoopDec" : {"FSMBitwidth" : "2", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter5", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter5", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "has_continue" : "1"}}]},
	{"ID" : "38", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.outidx_U", "Parent" : "37"},
	{"ID" : "39", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.w6_V_U", "Parent" : "37"},
	{"ID" : "40", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mux_325_8_1_1_U52", "Parent" : "37"},
	{"ID" : "41", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U53", "Parent" : "37"},
	{"ID" : "42", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U54", "Parent" : "37"},
	{"ID" : "43", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U55", "Parent" : "37"},
	{"ID" : "44", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U56", "Parent" : "37"},
	{"ID" : "45", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U57", "Parent" : "37"},
	{"ID" : "46", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U58", "Parent" : "37"},
	{"ID" : "47", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U59", "Parent" : "37"},
	{"ID" : "48", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U60", "Parent" : "37"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0", "Parent" : "13", "Child" : ["50"],
		"CDFG" : "normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "13", "EstimateLatencyMax" : "13",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "37",
		"StartFifo" : "start_for_normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0_U",
		"Port" : [
			{"Name" : "layer6_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["37"], "DependentChan" : "61", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer6_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer8_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["51"], "DependentChan" : "62", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer8_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "50", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0.mul_8s_8ns_14_1_1_U66", "Parent" : "49"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0", "Parent" : "13",
		"CDFG" : "relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "49",
		"StartFifo" : "start_for_relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0_U",
		"Port" : [
			{"Name" : "layer8_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["49"], "DependentChan" : "62", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer8_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer9_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["52"], "DependentChan" : "63", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer9_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0", "Parent" : "13", "Child" : ["53", "54", "55"],
		"CDFG" : "dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "Rewind", "UnalignedPipeline" : "0", "RewindPipeline" : "1", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "19", "EstimateLatencyMax" : "20",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "51",
		"StartFifo" : "start_for_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0_U",
		"Port" : [
			{"Name" : "layer9_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["51"], "DependentChan" : "63", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer9_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer10_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["56"], "DependentChan" : "64", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer10_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "w10_V", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "ReuseLoop", "PipelineType" : "rewind",
				"LoopDec" : {"FSMBitwidth" : "2", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter4", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter4", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "has_continue" : "1"}}]},
	{"ID" : "53", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.w10_V_U", "Parent" : "52"},
	{"ID" : "54", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.mux_164_8_1_1_U72", "Parent" : "52"},
	{"ID" : "55", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.mul_8s_6s_13_1_1_U73", "Parent" : "52"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0", "Parent" : "13", "Child" : ["57"],
		"CDFG" : "sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "3", "EstimateLatencyMax" : "3",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "52",
		"StartFifo" : "start_for_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0_U",
		"Port" : [
			{"Name" : "layer10_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["52"], "DependentChan" : "64", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer10_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "internal_output", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["72"], "DependentChan" : "78", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "internal_output_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "sigmoid_table", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "57", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.myproject_U0.sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0.sigmoid_table_U", "Parent" : "56"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.layer2_out_U", "Parent" : "13"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.layer4_out_U", "Parent" : "13"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.layer5_out_U", "Parent" : "13"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.layer6_out_U", "Parent" : "13"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.layer8_out_U", "Parent" : "13"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.layer9_out_U", "Parent" : "13"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.layer10_out_U", "Parent" : "13"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.start_for_normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0_U", "Parent" : "13"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.start_for_relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0_U", "Parent" : "13"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.start_for_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0_U", "Parent" : "13"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.start_for_normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0_U", "Parent" : "13"},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.start_for_relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0_U", "Parent" : "13"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.start_for_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0_U", "Parent" : "13"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.myproject_U0.start_for_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0_U", "Parent" : "13"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Block_inference_for_cond_i_exit13_proc3_U0", "Parent" : "0", "Child" : ["73", "74", "75"],
		"CDFG" : "Block_inference_for_cond_i_exit13_proc3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "13",
		"StartFifo" : "start_for_Block_inference_for_cond_i_exit13_proc3_U0_U",
		"Port" : [
			{"Name" : "internal_output", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["13","56"], "DependentChan" : "78", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "internal_output_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "result", "Type" : "Vld", "Direction" : "O"}]},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Block_inference_for_cond_i_exit13_proc3_U0.dcmp_64ns_64ns_1_4_no_dsp_1_U98", "Parent" : "72"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Block_inference_for_cond_i_exit13_proc3_U0.lshr_64ns_32ns_64_2_1_U99", "Parent" : "72"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.Block_inference_for_cond_i_exit13_proc3_U0.shl_64ns_32ns_64_2_1_U100", "Parent" : "72"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.features_in_data_V_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.internal_input_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.internal_output_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_Block_inference_for_cond_i_exit_proc2_U0_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_myproject_U0_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_Block_inference_for_cond_i_exit13_proc3_U0_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	inference {
		input_r_V_data_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_keep_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_strb_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_user_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_last_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_id_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_dest_V {Type I LastRead 0 FirstWrite -1}
		result {Type O LastRead -1 FirstWrite 10}
		outidx_1 {Type I LastRead -1 FirstWrite -1}
		w2_V {Type I LastRead -1 FirstWrite -1}
		outidx {Type I LastRead -1 FirstWrite -1}
		w6_V {Type I LastRead -1 FirstWrite -1}
		w10_V {Type I LastRead -1 FirstWrite -1}
		sigmoid_table {Type I LastRead -1 FirstWrite -1}}
	Loop_VITIS_LOOP_86_1_proc1 {
		features_in_data_V {Type O LastRead -1 FirstWrite 7}
		input_r_V_data_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_keep_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_strb_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_user_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_last_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_id_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_dest_V {Type I LastRead 0 FirstWrite -1}}
	Block_inference_for_cond_i_exit_proc2 {
		features_in_data_V {Type I LastRead 17 FirstWrite -1}
		internal_input {Type O LastRead -1 FirstWrite 17}}
	myproject {
		internal_input {Type I LastRead 1 FirstWrite -1}
		internal_output {Type O LastRead -1 FirstWrite 3}
		outidx_1 {Type I LastRead -1 FirstWrite -1}
		w2_V {Type I LastRead -1 FirstWrite -1}
		outidx {Type I LastRead -1 FirstWrite -1}
		w6_V {Type I LastRead -1 FirstWrite -1}
		w10_V {Type I LastRead -1 FirstWrite -1}
		sigmoid_table {Type I LastRead -1 FirstWrite -1}}
	dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s {
		internal_input {Type I LastRead 1 FirstWrite -1}
		layer2_out {Type O LastRead -1 FirstWrite 5}
		outidx_1 {Type I LastRead -1 FirstWrite -1}
		w2_V {Type I LastRead -1 FirstWrite -1}}
	normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_s {
		layer2_out {Type I LastRead 0 FirstWrite -1}
		layer4_out {Type O LastRead -1 FirstWrite 12}}
	relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_s {
		layer4_out {Type I LastRead 0 FirstWrite -1}
		layer5_out {Type O LastRead -1 FirstWrite 1}}
	dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s {
		layer5_out {Type I LastRead 2 FirstWrite -1}
		layer6_out {Type O LastRead -1 FirstWrite 6}
		outidx {Type I LastRead -1 FirstWrite -1}
		w6_V {Type I LastRead -1 FirstWrite -1}}
	normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_s {
		layer6_out {Type I LastRead 0 FirstWrite -1}
		layer8_out {Type O LastRead -1 FirstWrite 13}}
	relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_s {
		layer8_out {Type I LastRead 0 FirstWrite -1}
		layer9_out {Type O LastRead -1 FirstWrite 1}}
	dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_s {
		layer9_out {Type I LastRead 2 FirstWrite -1}
		layer10_out {Type O LastRead -1 FirstWrite 5}
		w10_V {Type I LastRead -1 FirstWrite -1}}
	sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_s {
		layer10_out {Type I LastRead 0 FirstWrite -1}
		internal_output {Type O LastRead -1 FirstWrite 3}
		sigmoid_table {Type I LastRead -1 FirstWrite -1}}
	Block_inference_for_cond_i_exit13_proc3 {
		internal_output {Type I LastRead 0 FirstWrite -1}
		result {Type O LastRead -1 FirstWrite 10}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "113", "Max" : "114"}
	, {"Name" : "Interval", "Min" : "72", "Max" : "72"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	input_r_V_data_V { axis {  { input_r_TDATA in_data 0 32 } } }
	input_r_V_keep_V { axis {  { input_r_TKEEP in_data 0 4 } } }
	input_r_V_strb_V { axis {  { input_r_TSTRB in_data 0 4 } } }
	input_r_V_user_V { axis {  { input_r_TUSER in_data 0 2 } } }
	input_r_V_last_V { axis {  { input_r_TLAST in_data 0 1 } } }
	input_r_V_id_V { axis {  { input_r_TID in_data 0 5 } } }
	input_r_V_dest_V { axis {  { input_r_TDEST in_data 0 6 }  { input_r_TVALID in_vld 0 1 }  { input_r_TREADY in_acc 1 1 } } }
	result { ap_vld {  { result out_data 1 32 }  { result_ap_vld out_vld 1 1 } } }
}

set maxi_interface_dict [dict create]

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
