# Racing-Car-Challenge
## TensorRT converter with docker
    
    git clone -b tensorrt_convert https://github.com/tonhathuy/Racing-Car-Challenge-.git
    cd Racing-Car-Challenge-
    
### 1. Run docker container 
    docker run -it --gpus all --name tensorrt_convert -e NVIDIA_NAME='1050ti' -v $(pwd):/backup/ huynhduc/tensorrt-plugins:alpha bash

note: change value NVIDIA_NAME

### 2. Convert pinet and yolo in container

#### - Convert Pinet
    cp /backup/Models_onnx/* Models_onnx/ && cp /backup/yolo/* tensorrt_demos/yolo
    trtexec --onnx=/workspace/Models_onnx/pinet_1block.onnx --saveEngine=/workspace/Models_trt/pinet_1block.trt --verbose
    trtexec --onnx=/workspace/Models_onnx/pinet_2block.onnx --saveEngine=/workspace/Models_trt/pinet_2block.trt --verbose


#### - Convert YOLO
    cd /workspace/tensorrt_demos/yolo
    python yolo_to_onnx.py -m yolov4-tiny-224
    python onnx_to_tensorrt.py -m yolov4-tiny-224
    trtexec --loadEngine=yolov4-tiny-224.trt --plugins=/workspace/tensorrt_demos/plugins/libyolo_layer.so

#### - Copy to host volume
    mkdir /backup/$NVIDIA_NAME
    cp yolov4-tiny-224.trt /backup/$NVIDIA_NAME/
    cd /workspace/ && cp Models_trt/* /backup/$NVIDIA_NAME/
    
### 3. Test 
    
    cd /workspace/tensorrt_demos/
    python trt_yolo.py --image data-tfs-00002973.jpg -m yolov4-tiny-224
    
    cd /backup/demo_trt/
    python pinet_trt.py 
