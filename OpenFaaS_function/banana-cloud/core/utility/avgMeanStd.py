import cv2 as cv
#import json
import os
from math import sqrt
import sys
#import copy
import numpy as np
from PIL import Image

from pad import get_padding
#import matplotlib.pyplot as plt
#from JsonAnnotations import JsonAnnotations
#from JsonAnnotationsState import JsonAnnotationsState

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_dt', dest='dir_dt', type = str, help='Directory contenente le cartelle delle immagini e i file json')
parser.set_defaults(dir_dt = None)
args = parser.parse_args()

def main():
    print('weeee')
    if args.dir_dt is None:
        print("[Augumentation dataset] ")
        print("Errore: Non hai inserito il nome della sottocartella del dataset.")
        print("Fix: Si consiglia di avere la cartella del dataset nella stessa cartella di lavoro.")
        print("(per problemi di file non trovati, modificare la root manualmente nel codice)")
        print("Esempio: python Augumentation.py --dir_dt Fruit")
        exit(1)

    root = '/home/a.dimarino/Scrivania/FruitsEvaluationNet/tools/dataset/' + args.dir_dt + '/'

    crop_path = root + 'cropped/'

    cont_img = 0

    mean_bgr = [0,0,0]
    std_bgr = [0,0,0]
    b_mean, g_mean, r_mean =  0.253, 0.300, 0.313
    #b_mean, g_mean, r_mean = 0, 0, 0  
    #std [82.21333840417539, 85.74588808090277, 86.82481277394689]
    b_std, g_std, r_std = 0,0,0
    b_sum, g_sum, r_sum = 0,0,0
    num_p = 0

    print(crop_path)

    for i in range(1,7):
        print('sto nella cartella: ', i)
        img_l = os.listdir(crop_path + str(i))
        cont_img += len(img_l)
        for img_name in img_l:
            #print('Sto processandro img: ', img_name)
            img = cv.imread(crop_path + str(i) + '/' + img_name)
            #img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            im_pil = Image.fromarray(img)
            pad = get_padding(im_pil)
            img = cv.copyMakeBorder(img, pad[1], pad[3], pad[0], pad[2], cv.BORDER_CONSTANT, 255)
            img = cv.resize(img, (256,256))

            img = img/255  
            # b_sum += np.sum(img[:,:,0].reshape(-1), dtype=np.longfloat)   
            # g_sum += np.sum(img[:,:,1].reshape(-1), dtype=np.longfloat)    
            # r_sum += np.sum(img[:,:,2].reshape(-1), dtype=np.longfloat)

            b_std += np.sum(np.power(img[:,:,0].reshape(-1) - b_mean, 2))
            g_std += np.sum(np.power(img[:,:,1].reshape(-1) - g_mean, 2))
            r_std += np.sum(np.power(img[:,:,2].reshape(-1) - r_mean, 2))
            num_p += (img.shape[0] * img.shape[1])
            #print(b_sum)       

    # mean_bgr = [ b_sum/num_p, g_sum/num_p, r_sum/num_p]
    #
    std_bgr = [sqrt(b_std/(num_p-1)), sqrt(g_std/(num_p-1)), sqrt(r_std/(num_p-1))]

    # print(mean_bgr)
    print(std_bgr)



if __name__ == '__main__':
    print('ooooooo')
    main()
