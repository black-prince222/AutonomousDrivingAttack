# AutonomousDrivingAttack
2022人工智能大赛自动驾驶目标检测攻击https://compete.zgc-aisc.com/activity/2/format
第七名方案。方案灵感来自https://github.com/THUrssq/Tianchi04

![比赛结果]([链接地址](https://github.com/black-prince222/AutonomousDrivingAttack/blob/main/ranking.jpg))

### 比赛介绍
官方给定五个视频。 用一个白盒模型yolov3和一个黑盒模型对视频进行目标检测， 若检测结果中不包含car、bus、truck这个类别，则这一帧图片可以得分。比赛需要提交一张patch图片和一张mask(只含0,255)。讲patchxmask的图像，贴在车厢侧面进行攻击。

### 比赛历程
（1）手动设计mask：包括“十”字型、“丰”字型、横“丰”字型、“米”字型的mask。 其中mask可以先创建全0的array(颜色影响不大)， 再用opencv划线。 其中“米”字型的效果最好。

（2）随机梯度下降学习patch。 把patch定义为可学习的参数，贴到车厢上。用sgd循环更新patch的值。

最终的成绩即是方案2，具体实现如下：

a. 在给定的五个视频中挑取一帧作为训练图片。 

b. 用传统视觉算法找到长方形车厢的位置。

c.用patch * mask的下采样到车厢大小， 贴图到车厢的位置。

d.用yolo模型对图片检测，选出所有预测框中，预测结果为bus、car、truck的类别概率之和作为损失函数。

e.循环训练patch， 直到loss不再变化或达到max_epoch


### 总结
由于是第一次参加对抗攻击类的比赛，且在比赛仅剩一周时才开始准备，仍有很多可提高的方向。如：
(1) 如何利用训练视频每一帧的图片进行训练。

(2) 参考对抗样本的论文，如DPatch

(3)自适应学习和聚类mask的形状。而不是人为规定。需要用knn等方法聚类形状。

(4）更多得考虑黑盒攻击

