# Review Paper

## 논문구현하기

### Pytorch 
ResNet, DenseNet : Scratch 
<br />
Vision Transformer, Transformer : Clone coding
### Numpy
MLP : Clone Coding

## Intro
전체적으로 Training 과정에 필요한 코드는 없고 Network 중심으로만 Coding

# AlexNet
- 논문 구조에서 분리하지 않고 하나의 flow로 구현
- BatchNorm 추가
- CIFAR-10 / epoch 20 / SGD(lr : 0.001, momentum : 0.9, weight_decay : 0.0005)
  - acc : 0.665 (Scratch Training)

# VGGNet
- 16 or 19 layers 기준으로 구현
- Max pooling, batch norm, ReLU 적용
- CIFAR-10 / epoch 20 / SGD(lr : 0.001, momentum : 0.9, weight_decay : 0.0005)
  - acc : 0.817 (Scratch Training)

# ResNet
- Skip Connection 적용
- ResNet101 기준으로 구현
- BatchNorm 적용
- Bottleneck은 변수를 통해 stride만 조절되는 형태로 refactoring
- CIFAR-10 / epoch 20 / SGD(lr : 0.001, momentum : 0.9, weight_decay : 0.0005)
  - acc : 0.691 (Scratch Training)

# SENet
- Squeeze-Extraction class 구현
- ResNet101 기준으로 SE 추가
- CIFAR-10 / epoch 20 / SGD(lr : 0.001, momentum : 0.9, weight_decay : 0.0005)
  - acc : 0.737 (Scratch Training)

# DenseNet
- DenseNet.py에 다시 구현, Skip Connection은 적용하지 않음
- CIFAR10 기준이라 7x7 conv 부분은 parameter 조정 필요 => padding = 0 으로 조정완료
- Unit Block 마다 skip connection 적용
- CIFAR-10 / epoch 20 / SGD(lr : 0.001, momentum : 0.9, weight_decay : 0.0005)
  - acc : 0.760 (Scratch Training)

# MobileNetv1
- Depthwise Separable Convolution은 Pytorch의 groups parameter와 1x1 Convolution을 combine하여 구현
- CIFAR-10 / epoch 20 / SGD(lr : 0.001, momentum : 0.9, weight_decay : 0.0005)
  - acc : 0.553 (Scratch Training)
# MobileNetv2
- CIFAR-10 / epoch 20 / SGD(lr : 0.001, momentum : 0.9, weight_decay : 0.0005)
  - acc : 0.469 (Scratch Training), 0.604 (epoch 75)

# ShuffleNetv1
- CIFAR-10 / epoch 20 / SGD(lr : 0.001, momentum : 0.9, weight_decay : 0.0005)
  - acc : 0.638 (Scratch Training) 

# Transformer
- Github 공식 코드 Clone Coding
  - 이후 안보고 coding 후 비교하는 방향으로 연습필요

# Vision Transformer
- Transformer 학습 후 다시 안보고 coding 연습
- CIFAR-10 기준으로 학습 코드 추가

# MLP
- Inductive Bias 를 줄이는 방향으로 연구가 되고있는 만큼 기초부터 다시 학습의 필요성을 느낌
- Clone Coding => 이론 복습 => 안보고 coding 연습
  - 순환반복
