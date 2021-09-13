FROM huynhduc/tensorrt-plugins:alpha
ARG NVIDIA_NAME

COPY . /backup/

WORKDIR /workspace 

RUN cp /backup/Models_onnx/* Models_onnx/ && \
    cp /backup/yolo/* tensorrt_demos/yolo/
RUN trtexec --onnx=/workspace/Models_onnx/pinet_1block.onnx --saveEngine=/workspace/Models_trt/pinet_1block.trt --verbose; \
    trtexec --onnx=/workspace/Models_onnx/pinet_2block.onnx --saveEngine=/workspace/Models_trt/pinet_2block.trt --verbose 
RUN cd /workspace/tensorrt_demos/yolo && \
    python yolo_to_onnx.py -m yolov4-tiny-224; \
    python onnx_to_tensorrt.py -m yolov4-tiny-224; \
    trtexec --loadEngine=yolov4-tiny-224.trt --plugins=/workspace/tensorrt_demos/plugins/libyolo_layer.so
RUN mkdir backup/$NVIDIA_NAME
    cp yolov4-tiny-224.trt /backup/$NVIDIA_NAME/
    cd /workspace/ && cp Models_trt/* /backup/$NVIDIA_NAME/
