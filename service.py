####default lib
import os
import base64
import logging
import time
import timeit
import datetime
import pydantic
####need install lib
import uvicorn
import cv2
import traceback
import asyncio
import numpy as np
####custom modules
import rcode
####default lib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from pydantic import BaseModel
from configparser import ConfigParser
####
now = datetime.datetime.now()
#######################################
from src.pinet_trt import *
import src.common as common
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
from src.utils.visualization import BBoxVisualization
from src.utils.yolo_with_plugins import TrtYOLO
from src.utils.yolo_classes import get_cls_dict
#####LOAD CONFIG####
config = ConfigParser()
config.read("config/service.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
LOG_PATH = str(config.get('main', 'LOG_PATH'))
MODEL_PINET_TRT_PATH = str(config.get('model', 'MODEL_PINET_TRT_PATH'))
MODEL_YOLO_TRT_PATH = str(config.get('model', 'MODEL_YOLO_TRT_PATH'))
#######################################
app = FastAPI()
#######################################
#####CREATE LOGGER#####
logging.basicConfig(filename=os.path.join(LOG_PATH, now.strftime("%d%m%y_%H%M%S")+".log"), filemode="w",
                level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)
#######################################
class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
class PredictData(BaseModel):
#    images: Images
    images: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
#######################################
####LOAD MODEL HERE
# You can set the logger severity higher to suppress messages (or lower to display more messages).
print("LOAD MODE", MODEL_PINET_TRT_PATH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with open(MODEL_PINET_TRT_PATH, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
inputs, outputs, bindings, stream = common.allocate_buffers(engine)

print("LOAD MODE", MODEL_YOLO_TRT_PATH)
cls_dict = get_cls_dict(11)
trt_yolo = TrtYOLO(MODEL_YOLO_TRT_PATH, 11, False)
#######################################
print("SERVICE_IP", SERVICE_IP)
print("SERVICE_PORT", SERVICE_PORT)
print("LOG_PATH", LOG_PATH)
print("API READY")
#######################################
@app.post('/predict')
async def predict(data: PredictData):
    ###################
    #####
    logger.info("predict")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            images = jsonable_encoder(data.images)
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        for image in images:
            image_decoded = base64.b64decode(image)
            jpg_as_np = np.frombuffer(image_decoded, dtype=np.uint8)
            process_image = cv2.imdecode(jpg_as_np, flags=1)
            
            process_image = cv2.resize(process_image,(512,256))
            test_image = to_np(process_image) / 255.0
            w_ratio = 512 * 1.0 / process_image.shape[1]
            h_ratio = 256 * 1.0 / process_image.shape[0]
            with engine.create_execution_context() as context:
                inputs[0].host = np.ascontiguousarray(test_image)
                trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                confidences = trt_outputs[1].reshape(1,-1,32,64)
                offsets = trt_outputs[2].reshape(1,-1,32,64)
                instances = trt_outputs[3].reshape(1,-1,32,64)
            xs, ys, ti = test([confidences, offsets, instances], process_image, w_ratio, h_ratio , 0.72)
            predicts = xs, ys
            boxes, confs, clss = trt_yolo.detect(process_image, 0.5)
            # vis = BBoxVisualization(cls_dict)
            # img = vis.draw_bboxes(process_image, boxes, confs, clss)
            # cv2.imwrite("test/trt_YOLO_result.jpg",img)
            print("PINET",xs)
            
            boxes, confs, clss = [i.tolist() for i in (boxes, confs, clss)]
            print("YOLO", boxes, confs, clss)
            # cv2.imwrite("test/trt_pinet_tested.jpg", ti)
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts_pinet': predicts, 'predicts_yolo': {'bboxes': boxes, 'conf': confs, 'class':clss},
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_binary')
async def predict_binary(binary_file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file = await binary_file.read()
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(bytes_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        process_image = cv2.resize(process_image,(512,256))
        predicts = net.predict(process_image, warp=False)
        image_points = net.get_image_points()
        # cv2.imwrite("_tested.jpg", image_points)
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result
        
@app.post('/predict_multi_binary')
async def predict_binary(binary_files: Optional[List[UploadFile]] = File(None)):
    ###################
    #####
    logger.info("predict_multi_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file_list = []
            for binary_file in binary_files:
                bytes_file_list.append(await binary_file.read())
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        process_image_list = []
        for bytes_file in bytes_file_list:
            nparr = np.fromstring(bytes_file, np.uint8)
            process_image = cv2.imdecode(nparr, flags=1)
            process_image = cv2.resize(process_image,(512,256))
            predicts = net.predict(process_image, warp=False)
            image_points = net.get_image_points()
            cv2.imwrite("_tested.jpg", image_points)
            process_image_list.append(process_image)

        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_multipart')
async def predict_multipart(argument: Optional[float] = Form(...),
                binary_file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_multipart")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file = await binary_file.read()
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(bytes_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        process_image = cv2.resize(process_image,(512,256))
        predicts = net.predict(process_image, warp=False)
        image_points = net.get_image_points()
        cv2.imwrite("_tested.jpg", image_points)
        
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

if __name__ == '__main__':
    uvicorn.run(app, port=SERVICE_PORT, host=SERVICE_IP)

