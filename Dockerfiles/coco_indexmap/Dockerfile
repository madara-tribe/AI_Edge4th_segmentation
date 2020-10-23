FROM jsbroks/coco-annotator:python-env
# Build tools

WORKDIR /home/app

RUN pip install --upgrade pip

#RUN git clone https://github.com/cocodataset/cocoapi.git && \
  #  cd cocoapi/PythonAPI && make
RUN pip install Cython
RUN pip install scikit-image numpy matplotlib opencv-python tqdm
CMD [ "/bin/bash" ]                     
