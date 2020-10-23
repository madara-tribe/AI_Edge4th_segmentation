#! /bin/sh
CNN=unet
COMPILE_DIR=output_compile
QUANT_DIR=quantized_model
TARGET=custom
ARCH=${TARGET}.json
vai_c_tensorflow \
 	 --frozen_pb ${QUANT_DIR}/deploy_model.pb \
 	 --arch ${ARCH} \
 	 --output_dir ${COMPILE_DIR}/${CNN} \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}
