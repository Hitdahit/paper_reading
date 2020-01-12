# Object Detection with Deep Learning: A Review

### I. Abstract
* 딥러닝 기반의 object detection 프레임워크 리뷰 논문.
* Convolutional Neural Network(CNN)부터 전형적인 object detection 아키텍쳐와 성능 향상을 위한 아키텍쳐 변형과 유용한 트릭들을 다룸.
* salient object detection, face detection, pedestrian detection 태스크들에 대한 간단한 설명과 그외 경험적 분석들과 future work에 관한 guideline들을 제시.
  
### II. Introduction
* image를 이해하기 위해서는 서로 다른 이미지들을 classify 하는 것만 집중해선 안되고, object들의 위치나 컨셉에 대한 정확한 평가에 대한 시도가 필요. (object detection의 정의.)
* object detection's problem definition
    1. object loclization: 이미지내에서 object이 위치하는 곳을 찾아내는 것.
    2. object classification: 이미지내의 object가 어떤 카테고리에 분류되는지 판별하는 것

* object detection의 3단계 진행과정:
   1. Informative region selection
    computationally expensive. produces too many redundant windows.
    그러나 윈도우의 갯수를 제한하면 만족스럽지 않은 지역들에 대한 윈도우들만 생성될 수 있다.
   2. Feature extraction
    서로 다른 객체들을 인지하기 위해서 객체의 시각적인 특징들이 필요하다.(semantic and robust representation)
   3. classification
    svm, ada boost, deformable part based model(dpm)등 여러가지 classifier들 중 하나를 채택하여 target object들을 구별하는데 이용한다.
       * DPM: a flexible model by combining object parts with deformation cost to handle severe deformations. ???무슨소린지 모르겠다

---
Based on these discriminant local feature descriptors and shallow learnable architectures, state of the art results have been obtained on PASCAL VOC object detection competition and real-time embedded systems have been obtained with a low burden on hardware.
---
* 딥러닝 이전의 object detection이 가진 문제점
  1. sliding window 방식의 후보 bounding box들의 생성은 쓸데없는 박스들을 더 많이 생성하여 비효율과 정확도 저하의 두마리 토끼를 모두 잡았다.
  2. semantic gap이 low-level descriptor들과 얕은 방식의 학습한 모델들로 극복하기엔 너무 gap이 큼.

* DNN의 도입
  * DNN과 CNN, R-CNN(regions with CNN) -> object detection 모델들의 대폭 성장
  * R-CNN 이후 Fast R-CNN, YOLO의 탄생

* 논문이 소개할 내용
  * 딥러닝 기반의 object detection 기술에 대한 리뷰와 그 기술들에 관한 고찰
  * 딥러닝과 CNN에 대한 소개
  * 일반적인 object detection 아키텍쳐 소개
  * salient object detection, face detection and pedestrian detection에 적용된 CNN에 대한 리뷰
  * 앞으로의 object detection의 발전 방향
  
### II. A BRIEF OVERVIEW OF DEEP LEARNING
#### A. The History: Birth, Decline and Prosperity
* The emergence of large scale annotated training data, such as ImageNet [, to fully exhibit its very large learning capacity;
* Fast development of high performance parallel computing systems, such as GPU clusters
* Significant advances in the design of network structures and training strategies. With unsupervised and layerwise pre-training guided by Auto-Encoder (AE) or Restricted Boltzmann Machine (RBM), a good initialization is provided. With dropout and data augmentation, the overfitting problem in training has been relieved. With batch normalization (BN), the training of very deep neural networks becomes quite efficient. Meanwhile, various network structures, such as AlexNet, Overfeat, GoogLeNet, VGG and ResNet, have been extensively studied to improve the performance.

#### B. Architecture and Advantages of CNN
* 가장 일반적인 CNN 아키텍쳐에는 VGG16이 있다.
* CNN의 각 layer들은 feature map이고, 인풋 레이어의 feature map은 3차원 픽셀의 행렬로 이뤄져 있다. 각 레이어들 뒤에 transformation을 넣을 수 있다(filtering, pooling)
* Filtering(convolution): filter matrix(learned weight)을 뉴런의 값들과 convolute하고 비선형 활성화 함수를 사용하여 결과를 얻는 연산
* Pooling(max, mean, L2): filtering의 결과를 요약.
* 이하 노트 공부 참조.

### III. GENERIC OBJECT DETECTION