import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2

class GLCM:
    def __init__(self, image):
        self.image = image
    
    def process(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(gray, bins)
        max_value = self.image.max()
        self.matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value+1, normed=False, symmetric=False)
    
    def get_global_feature(self):
        ret = {
            'contrast': self.contrast_feature(),
            'dissimilarity': self.contrast_feature(),
            'homogeneity': self.homogeneity_feature(),
            'energy': self.energy_feature(),
            'correlation': self.correlation_feature(),
            'ASM': self.asm_feature(),
        }
        return ret
    
    def contrast_feature(self):
        contrast = greycoprops(self.matrix_coocurrence, 'contrast')
        return contrast

    def dissimilarity_feature(self):
        dissimilarity = greycoprops(self.matrix_coocurrence, 'dissimilarity')
        return dissimilarity

    def homogeneity_feature(self):
        homogeneity = greycoprops(self.matrix_coocurrence, 'homogeneity')
        return homogeneity

    def energy_feature(self):
        energy = greycoprops(self.matrix_coocurrence, 'energy')
        return energy

    def correlation_feature(self):
        correlation = greycoprops(self.matrix_coocurrence, 'correlation')
        return correlation

    def asm_feature(self):
        asm = greycoprops(self.matrix_coocurrence, 'ASM')
        return asm