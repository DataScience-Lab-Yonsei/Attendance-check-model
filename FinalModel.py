import cv2 as cv
import numpy as np
import openface
from openface import openface
import sys
import dlib
import cv2
from mss import mss
from PIL import Image
from tensorflow.python.keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
# PyTorch Hub
import torch



if __name__ == "__main__":
    name = ['KyungHan','MinYong', 'YunSu', 'Won', 'YuRi', 'HaeRyung']
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
    print("loading keras model")
    model2 = load_model('models/cnn_model3_10.h5')


    bounding_box = {'top': 150, 'left': 100, 'width': 800, 'height': 1500}
    font = cv2.FONT_HERSHEY_PLAIN
    sct = mss()
    predictor_model = "./data/shape_predictor_68_face_landmarks.dat"
    face_pose_predictor = dlib.shape_predictor(predictor_model) 
    face_aligner = openface.AlignDlib(predictor_model)
    while True:
        sct_img = sct.grab(bounding_box)
        scr_img = np.array(sct_img)

        #cv2.imshow('screen', scr_img) # display screen in box
        scr_img = model(scr_img)
        res = scr_img.pandas().xyxy[0]
        startX = []
        startY = []
        endX = []
        endY = []
        detected_faces = dlib.rectangles()
        
        for i in range(len(res)):
            startX.append(res.iloc[i]['xmin'])
            endX.append(res.iloc[i]['xmax'])
            startY.append(res.iloc[i]['ymin'])
            endY.append(res.iloc[i]['ymax'])
        
        
        
        frame = np.squeeze(np.array(scr_img.imgs))
        frame_assign = frame.copy()

        for i in range(len(res)):
            #print(startX, startY, endX, endY)
            #좌측상단, 우측하단 ==> using cv2 => max box in output picture
    #         cv2.rectangle(frame, (int(startX[i]), int(endY[i])), (int(endX[i]), int(startY[i])),(0, 0, 255), 2)
            
            #for assignment of face using landmarks, then crop it
            dlibRect = dlib.rectangle(int(startX[i]), int(startY[i]), int(endX[i]), int(endY[i]))
            detected_faces.append(dlibRect)
        
        face_resize = np.empty([0,224,224,3])

        flag = False
        for i, face_rect in enumerate(detected_faces):
        # Get the the face's pose
            flag = True
            pose_landmarks = face_pose_predictor(frame_assign, face_rect)

        # Use openface to calculate and perform the face alignment
            alignedFace = face_aligner.align(534, frame_assign, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE) 
            cv2.imwrite("./aligned_face/aligned_face_{}_{}.jpg".format("zoom_yolo",i), alignedFace)
            res = resize(alignedFace, (224, 224, 3))
            face_resize = np.append(face_resize, np.expand_dims(res, axis=0), axis=0)
            
            cv2.rectangle(frame, (int(startX[i]), int(endY[i])), (int(endX[i]), int(startY[i])),(0, 0, 255), 2)

        if flag:
            pred = model2.predict(face_resize)
            for i, j in enumerate(pred):
                k = np.argmax(j)
                #print(k)
                cv2.putText(frame, name[k] , (int(startX[i]), int(endY[i] + 30)), font, 3, [0,0,255], 3)
            
        #cv2.imwrite("./aligned_face/aligned_face_{}_{}.jpg".format("zoom_yolo",i), alignedFace)
        cv.imshow('Testing',  frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break