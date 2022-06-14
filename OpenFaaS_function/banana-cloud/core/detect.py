#from objectDetect import run
from .test import FruitsEvaluationNet
import cv2
import numpy as np
import os
import time
from pathlib import Path
from .objectDetect_v2 import run


print('detect.py: ', Path.cwd())

color = {
    'underipe': (0,255,0),
    'barely ripe': (33,245,223),
    'ripe': (0,255,255),
    'very ripe': (7,194,240),
    'overipe': (14,132,162),
    'rotten': (0,51,102)
}

color_text ={
    'underipe': (0,0,0),
    'barely ripe': (0,0,0),
    'ripe': (0,0,0),
    'very ripe': (255,255,255),
    'overipe': (255,255,255),
    'rotten': (255,255,255)
}

def img_labelling(img_dic, state_label, img_cv):
    im = img_cv #un immagine
    for i, box in enumerate(img_dic['box_list']): #per tutti i box
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        diff_w = p2[0]-p1[0]
        prop = (0.7*diff_w)/290

        im = cv2.rectangle(im, p1, p2, color[state_label[i]], 2)
        text = state_label[i] + ' ' + img_dic['label_list'][i]
        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, prop, 2)
        cv2.rectangle(im, p1, (int(box[0])+textSize[0][0] + 7 ,int(box[1]) + 7 + (textSize[0][1])*2), color[state_label[i]], -1)
        im = cv2.putText(im, text, (int(box[0]) + 6, int(box[1]) +  textSize[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 
        prop, color_text[state_label[i]], 2, cv2.LINE_AA)
    
    #cv2.imwrite('/home/app/function/core/risultati/' + os.path.basename(img['img_name']).replace('.jpg', '') + '_crop.jpg', im)
    return im


def detect(img):
    root= '/home/app/function/core/'

    #start_time= time.time()
    img_list = run(weights=root+'yolov5/best.pt', cv_img = img)

    net = FruitsEvaluationNet()

    if not os.path.isdir('crop'):
        os.mkdir('crop')
    start_time= time.time()
    for img_dic in img_list:
        label = []
        for crop in img_dic['crop_list']:
            label.append(net.test(crop))
        im = img_labelling(img_dic, label, img)

    return im

