FROM ubuntu:18.04
RUN apt update && apt install x11-apps xterm sudo net-tools -y
ARG UID
ARG GID
ARG USER
RUN groupadd -g ${UID} $USER && \
    useradd -u ${GID} -g $USER -G sudo -r $USER && \
    mkdir /home/${USER} && \
    chown ${UID}:${GID} -R /home/${USER}
COPY sdk.sh /home/$USER/
RUN set -x && \
    cd /home/$USER/ && \
	mkdir AvnetTTC && \
	sudo chown ${UID}:${GID} AvnetTTC && \
	mkdir runtime && \
	cd runtime && \
	sudo apt update && \
	sudo apt install build-essential cmake python-dev python3-dev wget openssh-client gzip -y && \
	echo $PWD && \
	../sdk.sh -d $PWD -y && \
	sudo chmod 755 /home/Avnet/runtime/environment-setup-aarch64-xilinx-linux && \
	. /home/Avnet/runtime/environment-setup-aarch64-xilinx-linux && \
	wget https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz  -O vitis-ai_v1.1_dnndk.tar.gz && \
	tar xzvf vitis-ai_v1.1_dnndk.tar.gz && \
	cd vitis-ai_v1.1_dnndk && \
	echo $SDKTARGETSYSROOT && \
	sudo ./install.sh $SDKTARGETSYSROOT 
RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN echo 'alias ls="ls --color=auto"' >> /home/${USER}/.bashrc
RUN echo 'alias grep="grep --color=auto"' >> /home/${USER}/.bashrc
RUN echo 'alias fgrep="fgrep --color=auto"' >> /home/${USER}/.bashrc
RUN echo 'alias egrep="egrep --color=auto"' >> /home/${USER}/.bashrc
RUN echo 'alias ll="ls -alF"' >> /home/${USER}/.bashrc
RUN echo 'source /home/Avnet/runtime/environment-setup-aarch64-xilinx-linux' >> /home/${USER}/.bashrc
USER ${USER}
WORKDIR /home/${USER}
