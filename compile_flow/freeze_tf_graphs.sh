#!/bin/sh
INFER_GRAPH_FILENAME=infer_graph.pb
CHKPT_DIR=train_ck
CHKPT_FILENAME=fine.ckpt
FREEZE_DIR=freeze_tfpb
FROZEN_GRAPH_FILENAME=frozen_graph.pb
OUTPUT_NODE='activation_30/truediv'
freeze_graph \
	--input_graph       ${CHKPT_DIR}/${INFER_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${FROZEN_GRAPH_FILENAME} \
	--output_node_names ${OUTPUT_NODE}
