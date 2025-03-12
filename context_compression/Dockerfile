FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y vim curl && \
    pip install sentencepiece transformers datasets tiktoken matplotlib vastai

ENV DATA_DIR=/data
WORKDIR /workspace
RUN mkdir $DATA_DIR

COPY download_hellaswag.py ./
RUN python download_hellaswag.py
RUN mkdir $DATA_DIR/edu_fineweb10B
COPY ["*.npy", "$DATA_DIR/edu_fineweb10B"]
RUN apt-get update && apt-get install -y build-essential nvtop
RUN pip install wandb pytest mup ipykernel ipywidgets coloredlogs git+https://github.com/andrew-healey/vast-ai-api
