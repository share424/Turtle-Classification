import argparse
import sys
import numpy as np
import cv2
from PIL import Image
from segmentation import Segment
from glcm import GLCM
from his import HIS

def remove_background(image):
    segment = Segment(2)
    image = cv2.resize(image, (256, 256))
    label, segmented_image = segment.kmeans(image, 100)
    clean_image = segment.extract_object(image, label)
    return clean_image

def get_glcm_feature(image, feature=['contrast', 'correlation', 'energy', 'homogeneity']):
    glcm = GLCM(image)
    glcm.process()
    features = glcm.get_global_feature()
    out = []
    for f in feature:
        out.append(features[f])
    out = np.array(out)
    return out.flatten()

def get_his_feature(image):
    his = HIS(image)
    features = his.get_feature()
    return features


