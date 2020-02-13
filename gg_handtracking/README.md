# On-Device, Real-Time Hand Tracking with MediaPipe

## 概述

[博客链接(需翻墙)](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html)

严格来说这不是一篇论文，而是Google的一篇Blog，里面引入了两个内容：一个是Google 2019年下半年发布的机器学习编程框架 MediaPipe，另一个是基于MediaPipe实现的实时手部骨架提取模型。

该模型主要分为一个 矩形手掌检测器 以及一个 关节坐标预测器。相关的代码已经在MediaPipe的开源仓库中给出，同时我也在github上找到了python版本的复现，方便之后与自己训练的模型代码结合。

## 模型概述

手掌检测器 BlazePalm:
- 单发检测器模型(Single-shot detector) 
- 使用了非最大抑制算法(Non-max suppression)对重复检测进行了优化
- 使用了编解码特征提取器(encoder-decoder feature extractor)对多种场景的手掌提取进行了优化

关节坐标预测器 Hand Landmark Model:
- 基于手掌对21个关节坐标进行回归预测
- 手动标注30K的真实手掌图像
- 利用图像合成对模型进行了混合训练

## 优缺点分析

优点：
- 运行速度非常快，在程序上对模型的调用次数进行了优化，因此运行效率很高（对于CPU和内存占用都很少）
- 使用成本低，无需额外的设备（现在很多方案都是要配备千元以上的手套、相机等）

缺点：
- 相比较使用设备进行的骨架提取，精度有一定的损失