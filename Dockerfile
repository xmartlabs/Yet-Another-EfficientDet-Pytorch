FROM nvcr.io/nvidia/pytorch:20.03-py3

USER root
RUN apt-get update && apt-get install -y libcurl4-openssl-dev zlib1g-dev pkg-config libcairo2-dev libcups2-dev

# Install requirements and module code
WORKDIR /workspace/src/

COPY requirements.txt /workspace/src/requirements.txt
RUN pip install --upgrade pip && python3 -m pip install -r /workspace/src/requirements.txt

COPY build.sh /workspace/src/build.sh
# RUN chmod +x build.sh
RUN ./build.sh

ENTRYPOINT [ "/bin/bash" ]
