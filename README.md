# pytorch-ssd

by Sheng Tao, Qijie Zhao, Feng Ni. (VDIG，PKU)

#### Reproduced the proposed results.

#### Besides, our proposed upgraded model will be opened soon!(Note that we are not developed from pure SSD)
- VOC2007

model | mAP
---|---
ssd300 | 77.27%
ssd512 | 79.89%
Ours300-vgg | 80.5%
Ours512-vgg | 82.1%
Ours300-resnet101 | 81.7%
Ours512-resnet101 | 82.7%




- KITTI

model&Input | mAP
---|---
ssd300,TBA | TBA
ssd512,TBA | TBA
Ours300| 80.2%
Ours512 | 82.6%
Ours800 | 86.7%(==>up to 87.9%)
Ours800-multi-scale| 89.83%(==>up to 90.08%)

- MS COCO

model&Input | mAP(0.5:0.95)
---|---
Ours300-vgg|30.1%(TBA)
Ours300-resnet101|32.1%
Ours300-vgg-multiscale|36.7%
Ours512|34.8%(TBA)
Ours512-vgg-multiscale|39.0%

**Still being under fixing**.
