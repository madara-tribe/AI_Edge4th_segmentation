#!/bin/sh
FREEZE_DIR=freeze_tfpb
FROZEN_GRAPH_FILENAME=frozen_graph.pb
QUANT_DIR=quantized_model
INPUT_NODE="input_1"
Q_OUTPUT_NODE="conv2d_transpose_1/conv2d_transpose" # output node of quantized CNN

vai_q_tensorflow quantize \
	 --input_frozen_graph  ${FREEZE_DIR}/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         ${INPUT_NODE} \
	 --input_shapes        ?,400,640,3 \
	 --output_nodes        ${Q_OUTPUT_NODE} \
	 --output_dir          ${QUANT_DIR}/ \
	 --method              1 \
	 --input_fn            graph_input_fn.calib_input \
	 --calib_iter          15 \
	 --gpu 0