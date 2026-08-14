set moduleName Loop_VITIS_LOOP_86_1_proc1
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set C_modelName {Loop_VITIS_LOOP_86_1_proc1}
set C_modelType { void 0 }
set C_modelArgList {
	{ features_in_data_V int 8 regular {fifo 1 volatile }  }
	{ input_r_V_data_V int 32 regular {axi_s 0 volatile  { input_r Data } }  }
	{ input_r_V_keep_V int 4 regular {axi_s 0 volatile  { input_r Keep } }  }
	{ input_r_V_strb_V int 4 regular {axi_s 0 volatile  { input_r Strb } }  }
	{ input_r_V_user_V int 2 regular {axi_s 0 volatile  { input_r User } }  }
	{ input_r_V_last_V int 1 regular {axi_s 0 volatile  { input_r Last } }  }
	{ input_r_V_id_V int 5 regular {axi_s 0 volatile  { input_r ID } }  }
	{ input_r_V_dest_V int 6 regular {axi_s 0 volatile  { input_r Dest } }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "features_in_data_V", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "input_r_V_data_V", "interface" : "axis", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_keep_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_strb_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_user_V", "interface" : "axis", "bitwidth" : 2, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_id_V", "interface" : "axis", "bitwidth" : 5, "direction" : "READONLY"} , 
 	{ "Name" : "input_r_V_dest_V", "interface" : "axis", "bitwidth" : 6, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 24
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ input_r_TVALID sc_in sc_logic 1 invld 7 } 
	{ features_in_data_V_din sc_out sc_lv 8 signal 0 } 
	{ features_in_data_V_num_data_valid sc_in sc_lv 6 signal 0 } 
	{ features_in_data_V_fifo_cap sc_in sc_lv 6 signal 0 } 
	{ features_in_data_V_full_n sc_in sc_logic 1 signal 0 } 
	{ features_in_data_V_write sc_out sc_logic 1 signal 0 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ input_r_TDATA sc_in sc_lv 32 signal 1 } 
	{ input_r_TREADY sc_out sc_logic 1 inacc 7 } 
	{ input_r_TKEEP sc_in sc_lv 4 signal 2 } 
	{ input_r_TSTRB sc_in sc_lv 4 signal 3 } 
	{ input_r_TUSER sc_in sc_lv 2 signal 4 } 
	{ input_r_TLAST sc_in sc_lv 1 signal 5 } 
	{ input_r_TID sc_in sc_lv 5 signal 6 } 
	{ input_r_TDEST sc_in sc_lv 6 signal 7 } 
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
 	{ "name": "input_r_TVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "input_r_V_dest_V", "role": "default" }} , 
 	{ "name": "features_in_data_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "din" }} , 
 	{ "name": "features_in_data_V_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "num_data_valid" }} , 
 	{ "name": "features_in_data_V_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "fifo_cap" }} , 
 	{ "name": "features_in_data_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "full_n" }} , 
 	{ "name": "features_in_data_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "features_in_data_V", "role": "write" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "input_r_TDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "input_r_V_data_V", "role": "default" }} , 
 	{ "name": "input_r_TREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "inacc", "bundle":{"name": "input_r_V_dest_V", "role": "default" }} , 
 	{ "name": "input_r_TKEEP", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "input_r_V_keep_V", "role": "default" }} , 
 	{ "name": "input_r_TSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "input_r_V_strb_V", "role": "default" }} , 
 	{ "name": "input_r_TUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "input_r_V_user_V", "role": "default" }} , 
 	{ "name": "input_r_TLAST", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "input_r_V_last_V", "role": "default" }} , 
 	{ "name": "input_r_TID", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "input_r_V_id_V", "role": "default" }} , 
 	{ "name": "input_r_TDEST", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "input_r_V_dest_V", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
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
			{"Name" : "features_in_data_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "18", "DependentChanType" : "0",
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fpext_32ns_64_3_no_dsp_1_U1", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ashr_54ns_32ns_54_2_1_U2", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_input_r_V_data_V_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_input_r_V_keep_V_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_input_r_V_strb_V_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_input_r_V_user_V_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_input_r_V_last_V_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_input_r_V_id_V_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_input_r_V_dest_V_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	Loop_VITIS_LOOP_86_1_proc1 {
		features_in_data_V {Type O LastRead -1 FirstWrite 7}
		input_r_V_data_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_keep_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_strb_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_user_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_last_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_id_V {Type I LastRead 0 FirstWrite -1}
		input_r_V_dest_V {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "26", "Max" : "26"}
	, {"Name" : "Interval", "Min" : "26", "Max" : "26"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	features_in_data_V { ap_fifo {  { features_in_data_V_din fifo_port_we 1 8 }  { features_in_data_V_num_data_valid fifo_status_num_data_valid 0 6 }  { features_in_data_V_fifo_cap fifo_update 0 6 }  { features_in_data_V_full_n fifo_status 0 1 }  { features_in_data_V_write fifo_data 1 1 } } }
	input_r_V_data_V { axis {  { input_r_TDATA in_data 0 32 } } }
	input_r_V_keep_V { axis {  { input_r_TKEEP in_data 0 4 } } }
	input_r_V_strb_V { axis {  { input_r_TSTRB in_data 0 4 } } }
	input_r_V_user_V { axis {  { input_r_TUSER in_data 0 2 } } }
	input_r_V_last_V { axis {  { input_r_TLAST in_data 0 1 } } }
	input_r_V_id_V { axis {  { input_r_TID in_data 0 5 } } }
	input_r_V_dest_V { axis {  { input_r_TVALID in_vld 0 1 }  { input_r_TREADY in_acc 1 1 }  { input_r_TDEST in_data 0 6 } } }
}
