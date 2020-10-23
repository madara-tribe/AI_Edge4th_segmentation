run:
	docker build -t fcn8_train .
run2:
	docker build -t unet_train .

in:
	NV_GPU=3 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it fcn8_train:latest bash

in2:
	NV_GPU=0 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it unet_train:latest bash
