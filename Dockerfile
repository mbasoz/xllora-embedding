FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.10 python3-pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN pip install --no-cache-dir wheel
RUN python3 -m pip install --upgrade pip wheel \
 && pip install --no-cache-dir torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
      --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/simcse_train.py /app/
COPY src/evaluation_mteb.py /app/
COPY src/lora-prompting.py /app/
COPY src/generate_answers_multigpu.py /app/
COPY src/generate_answers_mgpu_orch.py /app/

