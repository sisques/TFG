import argparse
from opticalFlowModule import opticalFlow, lucas_kanade
from masksModuleClass import masksModule
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import os

cp_path = os.path.abspath("../weights/DYnamicCNN/")
np.set_printoptions(suppress=True)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    
    parser.add_argument('--inputImage1', default='../images/0.png', type=str)
    #parser.add_argument('--inputImage1', default='../images/vent1.png', type=str)
    #parser.add_argument('--inputImage1', default='../images/im1.png', type=str)
    #parser.add_argument('--inputImage1', default='../images/foto1.jpg', type=str)
    
    parser.add_argument('--inputImage2', default='../images/1.png', type=str)
    #parser.add_argument('--inputImage2', default='../images/vent2.png', type=str)
    #parser.add_argument('--inputImage2', default='../images/im2.png', type=str)
    #parser.add_argument('--inputImage2', default='../images/foto2.jpg', type=str)
    
    parser.add_argument('--outputPath', default='../images/out.png', type=str)

    parser.add_argument('--frameSkip', default=0, type=int)
    parser.add_argument('--videoPath', default=None, type=str)

    global args
    args = parser.parse_args(argv)

def decode_mask(img, W_H):
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tensor = tf.image.rgb_to_grayscale(img_tensor)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
    img_tensor = tf.image.resize(img_tensor, [W_H,W_H])
    img_tensor.set_shape([W_H,W_H,1])
    
    return img_tensor

def decode_opticalFlow(img,W_H):
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tensor = tf.image.rgb_to_grayscale(img_tensor)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
    img_tensor = tf.image.resize(img_tensor, [W_H,W_H])
    img_tensor.set_shape([W_H,W_H,1])
    
    return img_tensor



def toTensor(optFlow, masks, W_H=240):
    
    opticalFlow = decode_opticalFlow(optFlow, W_H)
    masks_out = []
    for m in masks:
        im = decode_mask(m, W_H)
        masks_out.append(im)
    masks_out = tf.convert_to_tensor(masks_out, tf.float32)

    images = tf.concat([[opticalFlow], masks_out], 0)

    batch = tf.expand_dims(images, 0)

    return batch

def genOutput(output, masks, shape, confidence=0.1):
    for batch in output:
        #batch = batch / max(batch)
        print(batch)

        img = np.zeros(masks[0].shape)
        for i in range(len(batch)):
            if batch[i] > confidence:
                
                img = img + 255*masks[i]# * colors[i]


    return np.uint8(img)


W_H=240

def obtainMasks(r1, goodPc = False):
    if goodPc:
        m = masksModule()
        masks = m.genMasks(r1)
    else: 
        m = masksModule(1)
        masks = m.genMasks(r1, 50)

        m = masksModule(3)
        masks = masks + m.genMasks(r1, 50)
    
    

    return masks



def detectImages(inputImage1, inputImage2, conf = 0.5):
    
    #Generar optical flow 
    im1 = cv.imread(inputImage1)
    im2 = cv.imread(inputImage2)
    init_shape = (im1.shape[1], im1.shape[0])
    
    optFlow = opticalFlow(inputImage1, inputImage2, W_H)
    
    width = int(W_H)
    height = int(W_H)
    dim = (width, height) 
    
    #r1 = im1.copy()
    r1 = cv.resize(im1, dim, interpolation = cv.INTER_AREA)
    optFlow = cv.resize(optFlow, dim, interpolation = cv.INTER_AREA)
    
    masks = obtainMasks(r1, goodPc=False)

    # Normalizar valores entre 0 y 1
    optFlow = optFlow / 255.0
    # masks ya está normalizado

    
    input_tensor =  toTensor(optFlow, masks)


    model = tf.keras.models.load_model(cp_path+"/reg_medium_50_5e-5.h5")
    
    #detectar

    output = model.predict(input_tensor)

    masks = genOutput(output, masks, init_shape, confidence = conf)
    res = im1.copy()
    h, w, d = im1.shape

    masks = cv.resize(masks, (w,h))
    
    for i in range(0, h):
        for j in range(0, w):
            for k in range(0,d):
                if(masks[i,j,k] != 0):
                    res[i,j,k] = masks[i,j,k]

    return res, masks


def detectVideo(videoPath, frameSkip):
    return frameSkip


def detectWebcam(frameSkip):
    cap = cv.VideoCapture(0)
    width = int(W_H)
    height = int(W_H)
    dim = (width, height) 
        

    if not cap.isOpened():
        raise IOError("Cannot open webcam")


    current = 0
    _, first_frame = cap.read()        


    while True:
        current +=1
        _, frame = cap.read()  
        cv.imshow("camera feed", frame)
        if current == frameSkip:
            current = 0
            _, second_frame = cap.read()        
            cv.imwrite(".f1.png", first_frame)
            cv.imwrite(".f2.png", second_frame)
            first_frame = second_frame.copy()


            res, _ = detectImages(".f1.png", ".f2.png")
            cv.imshow("res", res)

        c = cv.waitKey(1)
        if c == 27:
            break
    os.remove(".f1.png")
    os.remove(".f2.png")

## Parse arguments
if __name__ == '__main__':
    parse_args()
    if args.frameSkip > 0:
        if args.videoPath != None:
            print("Video input")
            detectVideo(args.videoPath, args.frameSkip)
        else:
            print("Webcam input")
            detectWebcam(args.frameSkip)
    else:
        print("Image input")
        res, _ = detectImages(args.inputImage1, args.inputImage2)
        cv.imwrite(args.outputPath, res)
