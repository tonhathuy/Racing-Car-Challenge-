FROM huynhduc/tensorrt-plugins:alpha

COPY ./ /re/

RUN apt update && apt install -y libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 libxrender-dev nano

RUN pip install -r /re/requirements.txt

WORKDIR /base

CMD nvidia-smi; sh start_service.sh
