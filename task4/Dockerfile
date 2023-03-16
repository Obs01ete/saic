FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y wget \
    && apt-get install -y python3.8 \
    && apt-get install -y python3-pip

COPY publisher.py /root

RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir opencv-python \
    && pip3 install --no-cache-dir torchvision \
    && pip3 install --no-cache-dir torch \
    && pip3 install --no-cache-dir paho-mqtt \
    && rm -rf ~/.cache/pip

RUN apt-get install -y ffmpeg libsm6 libxext6

RUN mkdir -p /root/.cache/torch/hub/checkpoints/
ADD https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /root/.cache/torch/hub/checkpoints/

WORKDIR /root

CMD ["python3", "publisher.py"]
