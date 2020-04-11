import cv2
import threading
import time
from hand_tracker import HandTracker
from skeleton_classifier import Classifier
import os
import pandas as pd
import queue
import matplotlib.pyplot as plt
import pyautogui
import random

class Config():
    def __init__(self):
        self.cameraFps = 15 # 摄像头的FPS（会被骨架处理速度阻塞）
        self.windowSize = 12 # 启动分类的骨架帧数
        self.threadNum = 2 # 处理骨架的线程数
        self.palm_model_path = ".\\models\\palm_detection.tflite"
        self.landmark_model_path = ".\\models\\hand_landmark.tflite"
        self.anchors_path = ".\\data\\anchors.csv" 
        self.labelNum = 12
        self.labelFPath = 'D:\\jester\\jester-v1-labels.csv'
        w, h = pyautogui.size()
        self.screenWidth = w
        self.screenHeight = h

class SkeletonThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.detector = HandTracker(config.palm_model_path, config.landmark_model_path, config.anchors_path,
                       box_shift=0, box_enlarge=1.5)
    def run(self):
        while True:
            # 阻塞获取帧信息
            item = frameQueue.get()
            if item is None:
                break
            index = item[0]
            frame = item[1]

            # 阻塞获取线程锁
            thread = threadQueue.get() # 注意这里取到的 thread 并不是自己! 只是一个计数凭证
            
            # 调用模型进行骨架推断
            kp, box, conf = self.detector(frame[:,:,::-1]) 
            if not (kp is None or conf < 1e-7):
                skeletonQueue.put((index, kp))
            else:
                skeletonQueue.put(False)
            # 归还线程锁
            threadQueue.put(thread)

        print('线程结束')

class ClassifierThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.classifier = Classifier()
        self.skeletons = []
        # 分类器预热，(8, 21, 2)的数据
        testSkel = [[[1,2]]*21]*15
        self.classifier.test(testSkel)
        
        # 绘图相关
        self.scores = []
        plt.ion()
        plt.figure(1)

        self.painting = False

        # 分类状态机相关
        self.labelState = {'label': 0, 'count': 0, 'cd': 0}
    def run(self):        
        while True:
            skel = skeletonQueue.get()
            if skel is None:
                break
            elif skel is False: # 空手势清零前面骨架数据
                self.skeletons.clear()
                self.labelState = {'label': 0, 'count': 0, 'cd': 0}
                
                if self.painting is True:
                    pyautogui.keyDown('ctrlleft') 
                    pyautogui.press('a')
                    pyautogui.keyUp('ctrlleft')
                    self.painting = False 
            else:
                cX, cY = pyautogui.position()
                self.skeletons.append(skel)
                if self.painting is True:
                    if skel[1][4][0] > skel[1][8][0]:
                        pyautogui.mouseDown()
                    else:
                        pyautogui.mouseUp()
                    w = ((1-skel[1][0][0]/256)*config.screenWidth + cX)/2
                    h = (skel[1][0][1]/256*config.screenHeight + cY)/2
                    pyautogui.moveTo(w, h, duration=0.1) # duration=1, tween=pyautogui.easeInOutQuad                

                # 超过阈值分类 
                if len(self.skeletons) < config.windowSize:
                    continue

                # TODO: 每个窗口检测一次，并绘图；尝试先不排序
                if self.painting is True:
                    continue

                window = self.skeletons[-config.windowSize:]
                for i in range(len(window)):
                    window[i] = window[i][1]
                # self.skeletons.sort()
                label, scores = self.classifier.test(window)
                print('     分类', labelDict[label+1], scores[0][label])

                if self.labelState['cd'] > 0:
                    self.labelState['cd'] -= 1
                elif scores[0][label] > 0.4:
                    if self.labelState['label'] == label:
                        self.labelState['count'] += 1
                    else:
                        self.labelState['label'] = label
                        self.labelState['count'] = 1
                    
                    if self.labelState['count'] >= 3:
                        # TODO: 异步调用
                        print('检测到：', labelDict[label+1])
                        self.labelState['count'] = 0
                        self.labelState['cd'] = 3

                        if label == 0:
                            pyautogui.press('left')
                        elif label == 1:
                            pyautogui.press('right')
                        elif label == 2:
                            pass
                        elif label == 3:
                            pass
                        elif label == 4:
                            pyautogui.press('f5')
                        elif label == 5:
                            pyautogui.press('esc')
                        elif label == 6:
                            pyautogui.moveTo(cX - 200, cY, duration=1, tween=pyautogui.easeInOutQuad)
                        elif label == 7:
                            pyautogui.moveTo(cX + 200, cY, duration=1, tween=pyautogui.easeInOutQuad)
                        elif label == 8:
                            pyautogui.moveTo(cX, cY + 200, duration=1, tween=pyautogui.easeInOutQuad)
                        elif label == 9:
                            pyautogui.moveTo(cX, cY - 200, duration=1, tween=pyautogui.easeInOutQuad)
                        elif label == 10:
                            pyautogui.keyDown('ctrlleft') 
                            pyautogui.press('a')
                            pyautogui.keyUp('ctrlleft') 
                        elif label == 11:
                            pyautogui.keyDown('ctrlleft') 
                            pyautogui.press('p')
                            pyautogui.keyUp('ctrlleft') 
                            # 进入绘画状态
                            self.painting = True
                            
                self.scores.append(scores[0][7])
                # 绘图
                plt.plot(range(len(self.scores)), self.scores,c='r')



if __name__ == "__main__":
    config = Config()
    frameQueue = queue.Queue() # 摄像头帧队列
    threadQueue = queue.Queue() # 骨架提取线程队列
    skeletonQueue = queue.Queue() # 骨架数据队列

    # 读取label信息
    labelCsv = pd.read_csv(config.labelFPath, sep=';')
    labelDict = dict(zip(labelCsv.clsid, labelCsv.clsname))

    # 启动分类线程
    classifierThread = ClassifierThread()
    classifierThread.start()

    # 启动骨架提取线程
    for i in range(config.threadNum):
        thread = SkeletonThread()
        threadQueue.put(thread)
        thread.start()

    # 主线程，生产者 - 摄像头帧计数与生产
    cap = cv2.VideoCapture(0) # 开启摄像头
    frameCount = 0 # 用于帧顺序标记
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frameQueue.put((frameCount,cv2.resize(frame, (256,256))))
        frameCount += 1
        cv2.imshow('video', frame)

        while threadQueue.empty(): # 等待直至线程有空
            continue

        keyInput = cv2.waitKey(int(1000/config.cameraFps))
        if keyInput == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 终止线程
    for i in range(config.threadNum):
        frameQueue.put(None)
    for i in range(config.threadNum):
        thread = threadQueue.get()
        thread.join()
    skeletonQueue.put(None)
    classifierThread.join()
    