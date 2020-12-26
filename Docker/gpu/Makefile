run:
	docker build -t unet_train .
run2:
	docker build -t unet_noaug_train .
run3:
	docker build -t aug_pre .
run4:
	docker build -t aug_pre2 .

in:
	NV_GPU=2 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it unet_train:latest bash

in2:
	NV_GPU=0 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it unet_noaug_train:latest bash

in3:
	NV_GPU=5 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it aug_pre:latest bash

in4:
	NV_GPU=1 nvidia-docker run --runtime=nvidia -v /home/hagi/indexmap:/home/app/sig --rm -it aug_pre2:latest bash
