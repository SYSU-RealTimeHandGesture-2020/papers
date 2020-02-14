# Make Skeleton-based Action Recognition Model Smaller, Faster and Better

## 论文概述

论文使用1D-CNN模型 研究基于纯骨架数据的离线手势识别。区别于其他论文往往只考虑手势的其中一个方面，这篇论文考虑了手势动作的两个方面：
1. 关节位置变化（比如“抓”的动作）
2. 整体手掌运动（比如“向左挥手”的动作）

两个特征都是通过笛卡尔坐标变换计算得来，会在**模型概述**中更进一步说明

论文在 SHREC-2017 的离线骨架数据手势识别比赛中取得了最优的成绩([Leaderboards](https://paperswithcode.com/sota/hand-gesture-recognition-on-shrec-2017-track))，可视为该领域 **state-of-the-art** 级别的论文，因为论文较新，尚未有期刊发表。

## 模型概述
- 模型总体的**输入数据**是固定帧数(TODO)的手势视频集的手部骨架信息(简单来说就是每个视频用一帧帧的骨架数据表示)，每个骨架`22`个关节点(joint)

- 对于**关节位置变化特征**，模型计算了每帧内两两关节坐标的几何距离，并最终将一帧动作的该特征(TODO)通过矩阵计算flatten为一维向量作为1D-CNN模型的一个input. 每个一维向量的维度为 `N(N-1)/2, N = joint_num` (因为只算矩阵的一半即可)

- 对于**整体手掌运动**的特征，模型计算了帧与帧之间的关节坐标变化，并基于快慢动作的考虑计算了原速度和两倍速度的帧间坐标变化
    - 慢动作即 `K frame`  与 `K-1 frame` 的坐标变化
    - 快动作即 `K frame`  与 `K-2 frame` 的坐标变化 
    - 每帧的动作也是flatten为一维向量作为模型的input. (原本应该是二维的: `dim * N, dim = (x,y,z)`)

- 考虑到 `JCD` 和 `slow motion` 时间长度为 `K`, `fast motion` 时间长度为 `K/2`, 在卷积过程中还对前两个输入进行了`MaxPooling(stride=2)`; 之后将经过几层卷积以后得到的三部分feature maps串联起来，最终得到的maps的维度是 `(K/2) * filters` 

- TODO: embedding 层: latent vectors, correlation of joints, reduces the effect of skeleton noise.

- 串联后的feature maps又通过多层的卷积/池化层，最终导入全连接层.

## 优缺点分析

论文优点：
- 相比较其他模型更“轻”（特征更少，1D-CNN模型也更轻量）
- 比赛数据最好

论文缺点：
- 只是基于纯骨架数据，没有实际场景（或者说对于实际场景它只考虑了使用骨架捕捉相机、手套之类的设备）
- 只考虑了离线数据，没有考虑实时的手势检测