import cv2
import numpy as np

class Segment:
    def __init__(self, segments=5):
        self.segments = segments
    
    def kmeans(self, image, iteration=10, epsilon=0.2):
        # reduce noise
        image = cv2.GaussianBlur(image, (7, 7), 0)
        vectorized = image.reshape(-1, 3)
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iteration, epsilon)
        ret, label, center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype('uint8')
    
    def extract_clean_component(self, image, label_image, label, flip=False):
        if(flip):
            label_image = (label_image - 1) * -1
        _, contours, _ = cv2.findContours(label_image.astype('uint8'), 1, 2)
        areas = [cv2.contourArea(c) for c in contours]
        idx = np.argmax(areas)
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        out = np.zeros(image.shape, np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        out[mask != 0] = image[mask != 0]
        return out
        
    def extract_component(self, image, label_image, label):
        component = np.zeros(image.shape, np.uint8)
        component[label_image==label] = image[label_image==label]
        return component
    
    def extract_object(self, image, label_image):
        if(label_image[0, 0] == 0):
            return self.extract_clean_component(image, label_image, 1)
        else:
            return self.extract_clean_component(image, label_image, 1, True)