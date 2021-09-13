#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

# This sample uses a UFF MNIST model to create a TensorRT Inference Engine
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import cv2
import time

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

from utils.display import open_window, set_display, show_fps

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

class param:
    def __init__(self):
        self.color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),
                    (100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]

        self.x_size = 512
        self.y_size = 256
        self.resize_ratio = 8
        self.grid_x = self.x_size//self.resize_ratio #64
        self.grid_y = self.y_size//self.resize_ratio #32

        self.threshold_point = 0.81
        self.threshold_instance = 0.08

        self.grid_location = np.zeros((self.grid_y, self.grid_x, 2))
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                self.grid_location[y][x][0] = x
                self.grid_location[y][x][1] = y

p = param() 

def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

# draw_points on original img.
def draw_points(x, y, image, w_ratio, h_ratio):
    color_index = 0
    for id in range(len(x)):
        color_index += 1
        if color_index > 12:
            color_index = 12  # 最多显示12种不同颜色，代表实例
        x_l = x[id]
        x_list = [int(x / w_ratio) for x in x_l]
        y_l = y[id]
        y_list = [int(y / h_ratio) for y in y_l]
        for pts in zip(x_list, y_list):
            image = cv2.circle(image, (int(pts[0]), int(pts[1])), 8, p.color[color_index], -1)  # 5
    return image

def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh

    grid = p.grid_location[mask]

    offset = offsets[mask]

    feature = instance[mask]
   
    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y


def test(model_output, test_image, w_ratio, h_ratio ,thresh=p.threshold_point):

    confidence, offset, instance = model_output[0],model_output[1],model_output[2]
    
    out_x = []
    out_y = []
    out_image = []
    
    confidence = np.squeeze(confidence)
  
    offset = np.squeeze(offset)
    offset = np.rollaxis(offset, axis=2, start=0)
    offset = np.rollaxis(offset, axis=2, start=0)
    
    instance = np.squeeze(instance)
    instance = np.rollaxis(instance, axis=2, start=0)
    instance = np.rollaxis(instance, axis=2, start=0)
    
    # generate point and cluster
    raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

    # eliminate fewer points
    in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
            
    # sort points along y 
    in_x, in_y = sort_along_y(in_x, in_y) 

    result_image = draw_points(in_x, in_y,test_image, w_ratio, h_ratio)

    out_x.append(in_x)
    out_y.append(in_y)
        
    return out_x, out_y,  result_image

def to_np(test_image):
    test_image = np.rollaxis(test_image, axis=2, start=0)
    inputs = test_image.astype(np.float32)
    inputs = inputs[np.newaxis,:,:,:] 
    return inputs





def main():
    model_path = "/workspace/Models_trt/"
    model_file = os.path.join(model_path, "pinet_1block.trt")
    image = cv2.imread("/backup/demo_trt/data-tfs-00002973.jpg")

    image_ori = cv2.resize(image,(512,256)) 
    test_image = to_np(image_ori) / 255.0
    w_ratio = 512 * 1.0 / image_ori.shape[1]
    h_ratio = 256 * 1.0 / image_ori.shape[0]


    with open(model_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())


    # with engine.create_execution_context() as context:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    fps = 0.0
    tic = time.time()
    with engine.create_execution_context() as context:
        inputs[0].host = np.ascontiguousarray(test_image)
        # For more information on performing inference, refer to the introductory samples.
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # [print(i.shape) for i in trt_outputs]
        confidences = trt_outputs[1].reshape(1,-1,32,64)
        offsets = trt_outputs[2].reshape(1,-1,32,64)
        instances = trt_outputs[3].reshape(1,-1,32,64)
    xs, ys, ti = test([confidences, offsets, instances], image_ori, w_ratio, h_ratio , 0.72)
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
    tic = toc
    print("cv2.imwrite trt_tested.jpg)")
    ti = show_fps(ti, fps)
    cv2.imwrite("trt_tested.jpg", ti)
if __name__ == '__main__':
    main()

