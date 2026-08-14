set moduleName sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_s
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set C_modelName {sigmoid<array,array<ap_fixed<8,2,5,3,0>,1u>,sigmoid_config11>}
set C_modelType { void 0 }
set C_modelArgList {
	{ layer10_out int 8 regular {fifo 0 volatile }  }
	{ internal_output int 8 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "layer10_out", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "internal_output", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 17
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ layer10_out_dout sc_in sc_lv 8 signal 0 } 
	{ layer10_out_num_data_valid sc_in sc_lv 2 signal 0 } 
	{ layer10_out_fifo_cap sc_in sc_lv 2 signal 0 } 
	{ layer10_out_empty_n sc_in sc_logic 1 signal 0 } 
	{ layer10_out_read sc_out sc_logic 1 signal 0 } 
	{ internal_output_din sc_out sc_lv 8 signal 1 } 
	{ internal_output_num_data_valid sc_in sc_lv 2 signal 1 } 
	{ internal_output_fifo_cap sc_in sc_lv 2 signal 1 } 
	{ internal_output_full_n sc_in sc_logic 1 signal 1 } 
	{ internal_output_write sc_out sc_logic 1 signal 1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "layer10_out_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "layer10_out", "role": "dout" }} , 
 	{ "name": "layer10_out_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "layer10_out", "role": "num_data_valid" }} , 
 	{ "name": "layer10_out_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "layer10_out", "role": "fifo_cap" }} , 
 	{ "name": "layer10_out_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "layer10_out", "role": "empty_n" }} , 
 	{ "name": "layer10_out_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "layer10_out", "role": "read" }} , 
 	{ "name": "internal_output_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "internal_output", "role": "din" }} , 
 	{ "name": "internal_output_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "internal_output", "role": "num_data_valid" }} , 
 	{ "name": "internal_output_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "internal_output", "role": "fifo_cap" }} , 
 	{ "name": "internal_output_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "internal_output", "role": "full_n" }} , 
 	{ "name": "internal_output_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "internal_output", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
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
		"Port" : [
			{"Name" : "layer10_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "1", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "layer10_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "internal_output", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "internal_output_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "sigmoid_table", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sigmoid_table_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	sigmoid_array_array_ap_fixed_8_2_5_3_0_1u_sigmoid_config11_s {
		layer10_out {Type I LastRead 0 FirstWrite -1}
		internal_output {Type O LastRead -1 FirstWrite 3}
		sigmoid_table {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "3", "Max" : "3"}
	, {"Name" : "Interval", "Min" : "3", "Max" : "3"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	layer10_out { ap_fifo {  { layer10_out_dout fifo_port_we 0 8 }  { layer10_out_num_data_valid fifo_status_num_data_valid 0 2 }  { layer10_out_fifo_cap fifo_update 0 2 }  { layer10_out_empty_n fifo_status 0 1 }  { layer10_out_read fifo_data 1 1 } } }
	internal_output { ap_fifo {  { internal_output_din fifo_port_we 1 8 }  { internal_output_num_data_valid fifo_status_num_data_valid 0 2 }  { internal_output_fifo_cap fifo_update 0 2 }  { internal_output_full_n fifo_status 0 1 }  { internal_output_write fifo_data 1 1 } } }
}
