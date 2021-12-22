import numpy as np
import cv2
import os

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('/home/valeronich/defectoscopy/output3_mono8_5250.avi', fourcc, 9.5454, (4096, 550))

input_folder = './detector/plates/'
for i in range(1, len(os.listdir(input_folder))):
    img_path = os.path.join(input_folder, 'content' + str(i) + '.jpg')
    print('Img path:', img_path)
    if not os.path.exists(img_path):
        print("Can't find image")
    frame = cv2.imread(img_path)

    #frame = cv2.flip(frame,0)

    out.write(frame)

out.release()


