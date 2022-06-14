import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import os
#os.chdir('yolov5/')

#import sys
#sys.path.append('/home/quaternione/FruitsEvaluationNet-main/yolov5/models')

print('objectDetect.py: ', Path.cwd())

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

# print('print root: ', ROOT.resolve())

from .yolov5.models.experimental import attempt_load
from .yolov5.utils.datasets import LoadImages_P, LoadStreams
from .yolov5.utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from .yolov5.utils.plots import Annotator, colors
from .yolov5.utils.torch_utils import load_classifier, select_device, time_sync




@torch.no_grad()
def run(weights= 'yolov5/best.pt',  # model.pt path(s)
        source='./',  # file/dir/URL/glob, 0 for webcam
        imgsz=[640,640],  # inference size (pixels)
        conf_thres=0.39,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project= 'yolov5/runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        cv_img = None #opencv image
        ):
    #source = str(source)

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages_P(cv_img, img_size=imgsz, stride=stride, auto=pt).esegui()
    dataset = [dataset]
    #rint('dataset: ', dataset, flush=True)
    #print('dataset len: ', len(dataset), flush=True)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0

    img_list = []
    for img, im0s in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            pred = model(img, augment=augment, visualize=visualize)[0]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3


        img_dict = {
            'img_name': 'img_webapp',
            'label_list': [],
            'crop_list': [],
            'box_list': []
        }

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
           
            s, im0, frame = '', im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):                    
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}_{conf:.2f}')
                    cc = save_one_box(xyxy, imc, gain=1 ,pad=0 ,BGR=True, save=False)
                    img_dict['crop_list'].append(cc) #save img cropped
                    img_dict['label_list'].append(label)
                    img_dict['box_list'].append(xyxy)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

        img_list.append(img_dict)


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t, flush=True)
    
    print("num img predette: ", len(img_list), flush=True)
    for img in img_list:
        print("l'img: " + img['img_name'] + " ha " + str(len(img['crop_list'])) + " crop box", flush=True)
    return img_list



