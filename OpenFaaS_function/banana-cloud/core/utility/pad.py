from torchvision.transforms.functional import pad, equalize
import numpy as np
import numbers
import cv2 as cv
from PIL import Image



def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h, 256])
    w_padding = (max_wh - w) / 2
    h_padding = (max_wh - h) / 2

    if w_padding > 0:
        l_pad = w_padding if w_padding % 1 == 0 else w_padding+0.5 #col
        r_pad = w_padding if w_padding % 1 == 0 else w_padding-0.5 #col
    else:
        l_pad, r_pad = 0,0
         
    if h_padding > 0:
        t_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5 #row
        b_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5 #row
    else:
        t_pad, b_pad = 0,0 
    
    padding = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]
    #print(padding)
    return padding
       

class NewPad(object):
    def __init__(self, fill=255, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        # img_cv = np.array(img)
        # img_cv = img_cv[:, :, ::-1].copy() 
        # img_cv = cv.cvtColor(img_cv, cv.COLOR_BGR2HSV)
        # img = Image.fromarray(img_cv)
    
        return pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


class Equalize(object):
    def __call__(self, img):
        return equalize(img)