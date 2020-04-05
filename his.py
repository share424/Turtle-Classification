import cv2
import numpy as np

class HIS:
    def __init__(self, image):
        image_hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        image_hsl = image_hsl.T
        self.hue = image_hsl[0]
        self.intensity = image_hsl[1]
        self.saturation = image_hsl[2]
        self.image = image_hsl.T
    
    def Hue(self):
        return self.hue
    
    def Intensity(self):
        return self.intensity
    
    def Saturation(self):
        return self.saturation
    
    def get(self):
        ret = {
            'Mean H': self.hue.mean(),
            'Mean I': self.intensity.mean(),
            'Mean S': self.saturation.mean(),
            'Std H': self.hue.std(),
            'Std I': self.intensity.std(),
            'Std S': self.saturation.std()
        }
        return ret
    
    def get_feature(self):
        out = np.array([
            self.hue.mean(),
            self.intensity.mean(),
            self.saturation.mean(),
            self.hue.std(),
            self.intensity.std(),
            self.saturation.std()
        ])
        return out