import os
import sys

ROOT_DIR = os.path.abspath("./")
YOLACT_PATH = os.path.abspath("../external/yolact")
sys.path.append(os.path.join(ROOT_DIR, YOLACT_PATH)) 
WEIGHTS_PATH = os.path.abspath("../weights/yolact/")

import cv2 as cv

from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess, undo_image_transformation
from data import cfg, set_cfg, set_dataset

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np

pesos = {
    'base' : WEIGHTS_PATH+"/yolact_base_54_800000.pth",
    'darknet' : WEIGHTS_PATH+"/yolact_darknet53_54_800000.pth",
    'im' : WEIGHTS_PATH+"/yolact_im700_54_800000.pth",
    'restnet' : WEIGHTS_PATH+"/yolact_resnet50_54_800000.pth"
}

class masksModule:
    def __init__(self, which = 0,):
        self.cuda = True
        self.goodPC = which==0
        global weights
        weights = dict()
        if which == 0: 
            weights = pesos
        else:
            key = list(pesos.keys())[which-1]
            value = list(pesos.values())[which-1]
            weights[key] = value
        with torch.no_grad():
            if self.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')

            if self.goodPC:

                self.base = Yolact()
                self.darknet = Yolact()
                self.im = Yolact()
                self.restnet = Yolact()

                self.base.load_weights(weights['base'])
                self.darknet.load_weights(weights['darknet'])
                self.im.load_weights(weights['im'])
                self.restnet.load_weights(weights['restnet'])


                self.base.eval()
                self.darknet.eval()
                self.im.eval()
                self.restnet.eval()


                if self.cuda:
                    self.base = self.base.cuda()
                    self.darknet = self.darknet.cuda()
                    self.im = self.im.cuda()
                    self.restnet = self.restnet.cuda()

                self.models = {
                    'base' : self.base,
                    'darknet' : self.darknet,
                    'im' : self.im,
                    'restnet' : self.restnet
                }


    def genMasks(self, im1, num = 25):
        output = []
        if self.cuda:
            frame = torch.from_numpy(im1).cuda().float()
        else:
            frame = torch.from_numpy(im1).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))

        for key in weights:
            torch.no_grad()
            if self.goodPC:
                self.models[key].detect.use_fast_nms = True
                self.models[key].detect.use_cross_class_nms = False
                cfg.mask_proto_debug = False

                preds = self.models[key](batch)
                masks = self.__getMasks(preds, frame, numMasks = num)
                output = output + masks
            else:
                net = Yolact()
                net.load_weights(weights[key])
                net.eval()
            
                if self.cuda:
                    net = net.cuda()

                net.detect.use_fast_nms = True
                net.detect.use_cross_class_nms = False
                cfg.mask_proto_debug = False

                preds = net(batch)
                masks = self.__getMasks(preds, frame, numMasks = num)
                output = output + masks
            torch.cuda.empty_cache()


        return output

    def __getMasks(self, dets_out, img, numMasks=25):
        output = []
        h, w, _ = img.shape
        t = postprocess(dets_out, w, h)
    
        # Masks are drawn on the GPU, so don't copy
        idx = t[1].argsort(0, descending=True)[:numMasks]
        masks = t[3][idx]
        for index in range(numMasks):
            if self.cuda:
                image = masks[index].cpu().detach().numpy()
            else:
                image = masks[index].detach().numpy()
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR) 
            output.append(image)
            
        return output


