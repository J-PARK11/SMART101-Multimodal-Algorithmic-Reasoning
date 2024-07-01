FROM python:3.9-slim
#FROM anibali/pytorch:1.7.0-cuda11.0-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

# Install system libraries required by OpenCV.
# 여기에다가도 만약에 추가적으로 Install 했으면 넣어야함.
RUN apt-get -y update 
RUN apt-get -y install git
RUN apt-get -y install libgl1-mesa-glx
RUN pip install --upgrade pip
# RUN apt-get install libgtk2.0-dev
# RUN mkdir -p /checkpoints/BERT
RUN mkdir -p /checkpoints/IBLIP-Flan-T5/multi_loss/checkpoints
RUN mkdir -p /checkpoints/RAW
RUN mkdir -p /checkpoints/Qwen/qwen_tokenizer.pt
RUN mkdir -p /checkpoints/Qwen/qwen.pt

COPY total_requirements.txt total_requirements.txt
# COPY requirements.txt requirements.txt
# COPY qwen_requirements.txt qwen_requirements.txt

RUN pip install -r total_requirements.txt
# RUN pip install huggingface-hub==0.17.0
# RUN pip install -r requirements.txt
# RUN pip install -r qwen_requirements.txt
# RUN pip install git+https://github.com/huggingface/transformers

COPY baselines.py  ./baselines.py
COPY losses.py ./losses.py
COPY data_loader.py  ./data_loader.py
COPY net.py ./net.py
COPY solve_VLAR.py ./solve_VLAR.py
COPY main.py	 ./main.py
COPY build_vocab.py	./build_vocab.py
COPY globvars.py	 ./globvars.py
COPY utils.py ./utils.py

# 체크포인트 학습해서 제출할 때 추가해야하면 여기다 넣어야 함. 
COPY checkpoints/IBLIP-Flan-T5/multi_loss/checkpoints/ckpt_IBLIP_none_5087.pth /checkpoints/IBLIP-Flan-T5/multi_loss/checkpoints/ckpt_IBLIP_none_5087.pth

COPY checkpoints/Qwen/qwen_tokenizer.pt/qwen.tiktoken /checkpoints/Qwen/qwen_tokenizer.pt/qwen.tiktoken
COPY checkpoints/Qwen/qwen_tokenizer.pt/special_tokens_map.json /checkpoints/Qwen/qwen_tokenizer.pt/special_tokens_map.json
COPY checkpoints/Qwen/qwen_tokenizer.pt/tokenizer_config.json /checkpoints/Qwen/qwen_tokenizer.pt/tokenizer_config.json

COPY checkpoints/Qwen/qwen.pt/config.json /checkpoints/Qwen/qwen.pt/config.json
COPY checkpoints/Qwen/qwen.pt/generation_config.json /checkpoints/Qwen/qwen.pt/generation_config.json
COPY checkpoints/Qwen/qwen.pt/model-00001-of-00004.safetensors /checkpoints/Qwen/qwen.pt/model-00001-of-00004.safetensors
COPY checkpoints/Qwen/qwen.pt/model-00002-of-00004.safetensors /checkpoints/Qwen/qwen.pt/model-00002-of-00004.safetensors
COPY checkpoints/Qwen/qwen.pt/model-00003-of-00004.safetensors /checkpoints/Qwen/qwen.pt/model-00003-of-00004.safetensors
COPY checkpoints/Qwen/qwen.pt/model-00004-of-00004.safetensors /checkpoints/Qwen/qwen.pt/model-00004-of-00004.safetensors
COPY checkpoints/Qwen/qwen.pt/model.safetensors.index.json /checkpoints/Qwen/qwen.pt/model.safetensors.index.json

COPY checkpoints/RAW/icon-classes.txt  /checkpoints/RAW/icon-classes.txt
COPY checkpoints/RAW/SMART_info_v2.csv /checkpoints/RAW/SMART_info_v2.csv
COPY checkpoints/vocab_puzzle_all_monolithic.pkl /checkpoints/vocab_puzzle_all_monolithic.pkl

# ADD ./checkpoints/BERT/config.json /checkpoints/BERT/config.json
# ADD ./checkpoints/BERT/pytorch_model.bin /checkpoints/BERT/pytorch_model.bin
# ADD ./checkpoints/BERT/special_tokens_map.json /checkpoints/BERT/special_tokens_map.json
# ADD ./checkpoints/BERT/tokenizer_config.json /checkpoints/BERT/tokenizer_config.json
# ADD ./checkpoints/BERT/vocab.txt /checkpoints/BERT/vocab.txt

CMD ["python", "main.py", "--model_name", "IBLIP", "--num_workers", "4", "--loss_type", "Both", "--word_embed", "none", "--split_type", "puzzle", "--challenge", "--phase", "val", "--use_option_prompting", "--LLM_type", "t5_xl", "--vocab_path", "./checkpoints/vocab_puzzle_all_monolithic.pkl", "--pretrained_model_path", "/checkpoints/IBLIP-Flan-T5/multi_loss/checkpoints/ckpt_IBLIP_none_5087.pth"]
