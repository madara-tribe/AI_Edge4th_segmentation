gitcolone:  # vitis-AI version
	git clone -b v1.1 https://github.com/Xilinx/Vitis-AI

xilinx:   #  when make app
	./docker_run.sh xilinx/vitis-ai:runtime-1.0.0-cpu

cpu:      # until compile
	./docker_run.sh xilinx/vitis-ai-cpu:latest

gpu:
	./docker_run.sh xilinx/vitis-ai-gpu:latest


