#!/usr/bin/python3

import threading
import time
import cv2
from hand_tracker import HandTracker
import os

palm_model_path = ".\\models\\palm_detection.tflite"
landmark_model_path = ".\\models\\hand_landmark.tflite"
anchors_path = ".\\data\\anchors.csv" 

video_path = 'D:\\jester\\data\\148079'
fns = []
results = []


class myThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0, box_enlarge=1.5)
    def run(self):
        while len(fns) > 0:
            fn = fns.pop()
            fpath = video_path + '\\' + fn 
            img = cv2.imread(fpath)
            img = cv2.resize(img,(256,256))
            kp, box, conf = self.detector(img[:,:,::-1])
            results.insert(0, kp)
            # print('left?',len(fns))

# 创建新线程
threadNum = 3
threads = []
for _ in range(threadNum):
    threads.insert(0, myThread())

# 开启新线程

import time

t = time.time()
fns = os.listdir(video_path)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

t = time.time() - t

print('time: ', t)
# print(results)
print ("退出主线程")