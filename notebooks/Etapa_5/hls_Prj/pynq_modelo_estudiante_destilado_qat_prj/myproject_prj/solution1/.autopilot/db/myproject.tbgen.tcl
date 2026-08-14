set moduleName myproject
set isTopModule 0
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
set C_modelName {myproject}
set C_modelType { void 0 }
set C_modelArgList {
	{ internal_input int 144 regular {fifo 0 volatile }  }
	{ internal_output int 8 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "internal_input", "interface" : "fifo", "bitwidth" : 144, "direction" : "READONLY"} , 
 	{ "Name" : "internal_output", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 16
set portList { 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ internal_input_dout sc_in sc_lv 144 signal 0 } 
	{ internal_input_empty_n sc_in sc_logic 1 signal 0 } 
	{ internal_input_read sc_out sc_logic 1 signal 0 } 
	{ internal_output_din sc_out sc_lv 8 signal 1 } 
	{ internal_output_full_n sc_in sc_logic 1 signal 1 } 
	{ internal_output_write sc_out sc_logic 1 signal 1 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
}
set NewPortList {[ 
	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "internal_input_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":144, "type": "signal", "bundle":{"name": "internal_input", "role": "dout" }} , 
 	{ "name": "internal_input_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "internal_input", "role": "empty_n" }} , 
 	{ "name": "internal_input_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "internal_input", "role": "read" }} , 
 	{ "name": "internal_output_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "internal_output", "role": "din" }} , 
 	{ "name": "internal_output_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "internal_output", "role": "full_n" }} , 
 	{ "name": "internal_output_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "internal_output", "role": "write" }} , 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "21", "23", "24", "36", "38", "39", "43", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58"],
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
		"InputProcess" : [
			{"ID" : "1", "Name" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0"}],
		"OutputProcess" : [
			{"ID" : "43", "Name" : "sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0"}],
		"Port" : [
			{"Name" : "internal_input", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0", "Port" : "internal_input"}]},
			{"Name" : "internal_output", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0", "Port" : "internal_output"}]},
			{"Name" : "outidx_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0", "Port" : "outidx_1"}]},
			{"Name" : "w2_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0", "Port" : "w2_V"}]},
			{"Name" : "outidx", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "24", "SubInstance" : "dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0", "Port" : "outidx"}]},
			{"Name" : "w6_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "24", "SubInstance" : "dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0", "Port" : "w6_V"}]},
			{"Name" : "w10_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "39", "SubInstance" : "dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0", "Port" : "w10_V"}]},
			{"Name" : "sigmoid_table", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0", "Port" : "sigmoid_table"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
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
			{"Name" : "internal_input", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "internal_input_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer2_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["21"], "DependentChan" : "45", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer2_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "outidx_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "w2_V", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "ReuseLoop", "PipelineType" : "rewind",
				"LoopDec" : {"FSMBitwidth" : "2", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter4", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter4", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "has_continue" : "1"}}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.outidx_1_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.w2_V_U", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_185_8_1_1_U22", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U23", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U24", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U25", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U26", "Parent" : "1"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U27", "Parent" : "1"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U28", "Parent" : "1"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U29", "Parent" : "1"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mul_8s_8s_13_1_1_U30", "Parent" : "1"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U31", "Parent" : "1"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U32", "Parent" : "1"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U33", "Parent" : "1"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U34", "Parent" : "1"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U35", "Parent" : "1"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U36", "Parent" : "1"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U37", "Parent" : "1"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_U0.mux_42_8_1_1_U38", "Parent" : "1"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0", "Parent" : "0", "Child" : ["22"],
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
		"StartSource" : "1",
		"StartFifo" : "start_for_normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0_U",
		"Port" : [
			{"Name" : "layer2_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["1"], "DependentChan" : "45", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer2_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer4_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["23"], "DependentChan" : "46", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer4_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0.mul_8s_7ns_14_1_1_U46", "Parent" : "21"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0", "Parent" : "0",
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
		"StartSource" : "21",
		"StartFifo" : "start_for_relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0_U",
		"Port" : [
			{"Name" : "layer4_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["21"], "DependentChan" : "46", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer4_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer5_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["24"], "DependentChan" : "47", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer5_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0", "Parent" : "0", "Child" : ["25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35"],
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
		"StartSource" : "23",
		"StartFifo" : "start_for_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0_U",
		"Port" : [
			{"Name" : "layer5_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["23"], "DependentChan" : "47", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer5_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer6_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["36"], "DependentChan" : "48", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer6_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "outidx", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "w6_V", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "ReuseLoop", "PipelineType" : "rewind",
				"LoopDec" : {"FSMBitwidth" : "2", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter5", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter5", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "has_continue" : "1"}}]},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.outidx_U", "Parent" : "24"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.w6_V_U", "Parent" : "24"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mux_325_8_1_1_U52", "Parent" : "24"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U53", "Parent" : "24"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U54", "Parent" : "24"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U55", "Parent" : "24"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U56", "Parent" : "24"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U57", "Parent" : "24"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U58", "Parent" : "24"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U59", "Parent" : "24"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0.mul_8s_8s_13_1_1_U60", "Parent" : "24"},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0", "Parent" : "0", "Child" : ["37"],
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
		"StartSource" : "24",
		"StartFifo" : "start_for_normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0_U",
		"Port" : [
			{"Name" : "layer6_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["24"], "DependentChan" : "48", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer6_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer8_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["38"], "DependentChan" : "49", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer8_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0.mul_8s_8ns_14_1_1_U66", "Parent" : "36"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0", "Parent" : "0",
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
		"StartSource" : "36",
		"StartFifo" : "start_for_relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0_U",
		"Port" : [
			{"Name" : "layer8_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["36"], "DependentChan" : "49", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer8_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer9_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["39"], "DependentChan" : "50", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer9_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0", "Parent" : "0", "Child" : ["40", "41", "42"],
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
		"StartSource" : "38",
		"StartFifo" : "start_for_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0_U",
		"Port" : [
			{"Name" : "layer9_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["38"], "DependentChan" : "50", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer9_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer10_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["43"], "DependentChan" : "51", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer10_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "w10_V", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "ReuseLoop", "PipelineType" : "rewind",
				"LoopDec" : {"FSMBitwidth" : "2", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter4", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter4", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "has_continue" : "1"}}]},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.w10_V_U", "Parent" : "39"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.mux_164_8_1_1_U72", "Parent" : "39"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0.mul_8s_6s_13_1_1_U73", "Parent" : "39"},
	{"ID" : "43", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0", "Parent" : "0", "Child" : ["44"],
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
		"StartSource" : "39",
		"StartFifo" : "start_for_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0_U",
		"Port" : [
			{"Name" : "layer10_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["39"], "DependentChan" : "51", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer10_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "internal_output", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "internal_output_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "sigmoid_table", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0.sigmoid_table_U", "Parent" : "43"},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.layer2_out_U", "Parent" : "0"},
	{"ID" : "46", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.layer4_out_U", "Parent" : "0"},
	{"ID" : "47", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.layer5_out_U", "Parent" : "0"},
	{"ID" : "48", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.layer6_out_U", "Parent" : "0"},
	{"ID" : "49", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.layer8_out_U", "Parent" : "0"},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.layer9_out_U", "Parent" : "0"},
	{"ID" : "51", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.layer10_out_U", "Parent" : "0"},
	{"ID" : "52", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_normalize_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_config4_U0_U", "Parent" : "0"},
	{"ID" : "53", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_relu_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_32u_relu_config5_U0_U", "Parent" : "0"},
	{"ID" : "54", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_U0_U", "Parent" : "0"},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_normalize_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_config8_U0_U", "Parent" : "0"},
	{"ID" : "56", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_relu_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_16u_relu_config9_U0_U", "Parent" : "0"},
	{"ID" : "57", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_U0_U", "Parent" : "0"},
	{"ID" : "58", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_U0_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
		sigmoid_table {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "87", "Max" : "88"}
	, {"Name" : "Interval", "Min" : "72", "Max" : "72"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	internal_input { ap_fifo {  { internal_input_dout fifo_port_we 0 144 }  { internal_input_empty_n fifo_status 0 1 }  { internal_input_read fifo_data 1 1 } } }
	internal_output { ap_fifo {  { internal_output_din fifo_port_we 1 8 }  { internal_output_full_n fifo_status 0 1 }  { internal_output_write fifo_data 1 1 } } }
}
