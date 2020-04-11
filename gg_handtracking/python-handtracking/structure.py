import queue
import threading
import time

def worker():
    while not end:
        frame = frameQueue.get()
        if frame is None:
            break
        thread = threadQueue.get() # 注意这里取到的 thread 并不是自己! 只是一个计数凭证
        time.sleep(0.5) # 模拟处理时间
        skels.append((frame[0],frame[1]+1000))
        threadQueue.put(thread)
        print('     处理了', frame)

threadQueue = queue.Queue()
frameQueue = queue.Queue()
skels = []

num_worker_threads = 3
classifyFrameNum = 5
end = False

for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    threadQueue.put(t)
    t.start()

count = 500 # 用来对帧进行重排

for i in range(30):
    frameQueue.put((count,i))
    count += 1

    if len(skels) >= classifyFrameNum:
        # 模拟分类
        temp = skels.copy()
        skels = []
        temp.sort()
        print('进行了分类', temp)

    while threadQueue.empty(): # 等待直至线程有空
        continue

    print('播放了', i)
    time.sleep(0)
# stop workers
end = True
for i in range(num_worker_threads):
    frameQueue.put(None)

for i in range(num_worker_threads):
    thread = threadQueue.get()
    thread.join()