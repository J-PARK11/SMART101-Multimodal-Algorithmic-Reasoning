FROM python:3.8-slim-buster
#FROM anibali/pytorch:1.7.0-cuda11.0-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

# Install system libraries required by OpenCV.
RUN apt-get -y update 
RUN apt-get -y install git
RUN mkdir -p /checkpoints/BERT

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

COPY checkpoints/ckpt_resnet50_bert_212.pth /checkpoints/ckpt_resnet50_bert_212.pth
COPY data/icon-classes.txt  /checkpoints/icon-classes.txt  
COPY data/SMART_info_v2.csv /checkpoints/SMART_info_v2.csv

ADD ./checkpoints/BERT/config.json /checkpoints/BERT/config.json
ADD ./checkpoints/BERT/pytorch_model.bin /checkpoints/BERT/pytorch_model.bin
ADD ./checkpoints/BERT/special_tokens_map.json /checkpoints/BERT/special_tokens_map.json
ADD ./checkpoints/BERT/tokenizer_config.json /checkpoints/BERT/tokenizer_config.json
ADD ./checkpoints/BERT/vocab.txt /checkpoints/BERT/vocab.txt
ADD ./checkpoints/resnet50-11ad3fa6.pth /checkpoints/resnet50-11ad3fa6.pth

CMD ["python", "main.py", "--model_name", "resnet50", "--num_workers", "0", "--loss_type", "classifier", "--word_embed", "bert", "--split_type", "puzzle", "--challenge", "--phase", "val", "--pretrained_model_path", "/checkpoints/ckpt_resnet50_bert_212.pth"]
