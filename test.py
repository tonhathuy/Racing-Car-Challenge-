import cv2
import torch 
import time
import argparse
import numpy as np

from src import util
from src.net import Net
from src.parameters import Parameters
from src.processing_image import warp_image

if __name__ == "__main__":

    net = Net()
    net.load_model('model/34_tensor(0.7828)_lane_detection_network.pkl')
    img_path = "test/test.jpg"
    image = cv2.imread(img_path)
    image = cv2.resize(image,(512,256))
    x, y = net.predict(image, warp=False)
    print(x, y)
    image_points = net.get_image_points()
    cv2.imwrite("{}_tested.jpg".format(img_path.split('.jpg')[0]), image_points)