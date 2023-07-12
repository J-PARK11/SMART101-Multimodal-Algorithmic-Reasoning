FROM anibali/pytorch:1.7.0-cuda11.0-ubuntu20.04

# Install system libraries required by OpenCV.
RUN sudo apt-get update
RUN apt-get -y install git 

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

WORKDIR /vlar

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY baselines.py  ./baselines.py
COPY losses.py ./losses.py
COPY data_loader.py  ./data_loader.py
COPY net.py ./net.py
COPY solve_VLAR.py ./solve_VLAR.py
COPY main.py	 ./main.py
COPY build_vocab.py	./build_vocab.py
COPY globvars.py	 ./globvars.py
COPY utils.py ./utils.py

COPY checkpoints/ckpt_resnet50_bert_212.pth ./ckpt_resnet50_bert_212.pth
COPY data/icon-classes.txt  ./icon-classes.txt  
COPY data/SMART_info_v2.csv ./SMART_info_v2.csv

CMD ["python", "main.py", "--model_name", "resnet50", "--num_workers", "0", "--loss_type", "classifier", "--word_embed", "bert", "--split_type", "puzzle", "--challenge", "--phase", "test", "--pretrained_model_path", "ckpt_resnet50_bert_212.pth"]
