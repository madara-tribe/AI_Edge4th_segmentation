run:
	docker build -t cpupb .

in:
	NV_GPU=3 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it cpupb bash
