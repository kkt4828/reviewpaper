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

# ResNet
- Skip Connection 적용
- BottleNeck 적용
  - 첫 maxpooling에 의한 size가 반으로 줄어든 block과 나머지 block에 대해서 다른 bottleneck 적용
    - 좀 더 개선하여 통합된 class로 구현이 필요함
- 논문 network만 보고 구현하여 refactoring이 필요함

# DenseNet
- 쥬피터 노트북으로 셀단위로 구현함
- DenseBlock 및 Skip Connection 구현
- DenseBlock 부분의 refactoring 필요
- DenseNet.py에 다시 구현, Skip Connection은 적용하지 않음
- CIFAR10 기준이라 7x7 conv 부분은 parameter 조정 필요

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
