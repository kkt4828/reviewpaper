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

# Transformer
- Github 공식 코드 Clone Coding
  - 이후 안보고 coding 후 비교하는 방향으로 연습필요

# Vision Transformer
- Transformer 학습 후 다시 안보고 coding 연습

# MLP
- Inductive Bias 를 줄이는 방향으로 연구가 되고있는 만큼 기초부터 다시 학습의 필요성을 느낌
- Clone Coding => 이론 복습 => 안보고 coding 연습
  - 순환반복
