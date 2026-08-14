set moduleName Block_inference_for_cond_i_exit_proc2
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
set C_modelName {Block_inference_for.cond.i.exit_proc2}
set C_modelType { void 0 }
set C_modelArgList {
	{ features_in_data_V int 8 regular {fifo 0 volatile }  }
	{ internal_input int 144 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "features_in_data_V", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "internal_input", "interface" : "fifo", "bitwidth" : 144, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 20
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ features_in_data_V_dout sc_in sc_lv 8 signal 0 } 
	{ features_in_data_V_num_data_valid sc_in sc_lv 6 signal 0 } 
	{ features_in_data_V_fifo_cap sc_in sc_lv 6 signal 0 } 
	{ features_in_data_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ features_in_data_V_read sc_out sc_logic 1 signal 0 } 
	{ internal_input_din sc_out sc_lv 144 signal 1 } 
	{ internal_input_num_data_valid sc_in sc_lv 2 signal 1 } 
	{ internal_input_fifo_cap sc_in sc_lv 2 signal 1 } 
	{ internal_input_full_n sc_in sc_logic 1 signal 1 } 
	{ internal_input_write sc_out sc_logic 1 signal 1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "features_in_data_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "dout" }} , 
 	{ "name": "features_in_data_V_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "num_data_valid" }} , 
 	{ "name": "features_in_data_V_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "fifo_cap" }} , 
 	{ "name": "features_in_data_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "empty_n" }} , 
 	{ "name": "features_in_data_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "read" }} , 
 	{ "name": "internal_input_din", "direction": "out", "datatype": "sc_lv", "bitwidth":144, "type": "signal", "bundle":{"name": "internal_input", "role": "din" }} , 
 	{ "name": "internal_input_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "internal_input", "role": "num_data_valid" }} , 
 	{ "name": "internal_input_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "internal_input", "role": "fifo_cap" }} , 
 	{ "name": "internal_input_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "internal_input", "role": "full_n" }} , 
 	{ "name": "internal_input_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "internal_input", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
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
		"Port" : [
			{"Name" : "features_in_data_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "18", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "features_in_data_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "internal_input", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0","0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "internal_input_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	Block_inference_for_cond_i_exit_proc2 {
		features_in_data_V {Type I LastRead 17 FirstWrite -1}
		internal_input {Type O LastRead -1 FirstWrite 17}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "17", "Max" : "17"}
	, {"Name" : "Interval", "Min" : "17", "Max" : "17"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	features_in_data_V { ap_fifo {  { features_in_data_V_dout fifo_port_we 0 8 }  { features_in_data_V_num_data_valid fifo_status_num_data_valid 0 6 }  { features_in_data_V_fifo_cap fifo_update 0 6 }  { features_in_data_V_empty_n fifo_status 0 1 }  { features_in_data_V_read fifo_data 1 1 } } }
	internal_input { ap_fifo {  { internal_input_din fifo_port_we 1 144 }  { internal_input_num_data_valid fifo_status_num_data_valid 0 2 }  { internal_input_fifo_cap fifo_update 0 2 }  { internal_input_full_n fifo_status 0 1 }  { internal_input_write fifo_data 1 1 } } }
}
