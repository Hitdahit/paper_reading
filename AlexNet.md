# AlexNet

---

## Abstract

Imagenet 데이터(1000class 데이터) top1, top5 error 각각 37.5%, 17% 뜸. 이때 당시 sota

모델 구조 ->  5*(conv layer + maxpooling layer) + 3 * fc + 1000way softmax 

​	->  6천만개의 파라미터와 65만개 뉴런을 가짐.

fc의 overfitting을 방지하기 위해서 dropout을 차용해서 사용.



## Introduction

객체인식 분야에 대한 최근의 접근들은 필수적으로 머신러닝 기법들을 사용해왔다. 머신러닝 기법의 성능을 높이기 위하여 우리는 데이터셋을 더 모으거나, 더 강력한 모델을 가여오거나 오버피팅을 방지하는 기법들을 사용할 수있다. 그리고 실제로 간단한 인식 태스크들에서는 레이블을 보존하는 transformation기법들을 사용한 데이터 셋 증강기법이 유효하게 작용한다 (ex. MNIST). 그럼에도 불구하고 real world 데이터셋에서는 너무 많은 변수가 존재하고, 작은 이미지 데이터셋도 학습이 잘안되는 경우가 생겼다. 그래서 데이터셋들이 점점 커졌다.

그러고 나니까, 이제 더 큰 데이터셋을 학습할 수 있는데 네트워크가 필요해진 것. 물론, 엄청나게 복잡한 객체 인식 태스크는 다른 의미로 이 문제가 데이터 셋의 크기만이 문제가 아님을 시사하기도 하므로, 우리가 개발할 모델이 우리에게 없는데이터를 보충하기 위해 선수적으로 알아야 할 것이 더 많아져야 했다. 

CNN은 그 depth나 breadth 에 따라 능력이 크게 달라지며, 강력하고 매우정확한 추론을 해낼 수 있다. 심지어 다른 feed-forward네트워크에 비해서 CNN은 더 적은 파라미터를 가지므로 학습하기도 쉽다.

그럼에도 불구하고 CNN은 high-resolution 이미지에 대하여 컴퓨팅연산이 매우 크게 든다. 그러나 GPU로 해결.

우리의 contribution은 다음과 같다.

 우리는 지금까지의 cnn들중 가장 큰것 중 하나를 ImageNet에 학습 시켰고 best result를 냈다.

 우리 네트워크는 몇개의 새롭고 잘 사용하지 않는 feature를 포함하는데, 이것이 더 좋은 성능을 이끌어내고 학습 시간도 많이 줄였다. ( Section3에 언급)

  overfitting을 해결하기 위한 많은 방법들 사용(Section4에 언급)

  우리 네트워크는 5conv+3fc인데, 이중 하나의 conv라도 지우는 것은 더 안좋은 성능을 가져왔다(각  conv들은 모델 파라미터 수의 1%도 차지하지 않는다).

GTX 580 3기가 2개 기준으로 6일정도의 시간이 있어야 학습이 된다(최대 인내심 한계선)

## 2 The Dataset

생략 ^^

## 3 The Architecture

전체 아키텍쳐는 8개의 레이어를 가지게 된다. 5(conv+maxpooling) + 3*fc. 그 속에 우리는 몇가지 특징점들을 집어넣어 두었다. 

	### 1. ReLU Nonlinerlity

네트워크의 아웃풋은 f(x)=tanh(x) 혹은 f(x) = (1+e^-x)^-1을 사용한다.(활성화 함수)

==================================

**** 비선형함수를 쓰는 이유

conv는 선형 연산이므로, 이를 깊게 해봤자 층을 깊게 하는 의미가 없기 때문에 활성화 함수를 끼워서 층을 깊게 하는 의미를 만든다. (어차피 0~1로 normalize되는 것이나 다름없다.)



non-saturating nonlinearity 함수는 어떤 입력 x가 무한대로 갈때 함수의 값이 무한대로 가는 것을 의미하고

saturating nonlinearity 함수는 어떤 입력 x가 무한대로 갈때 함수의 값이 어떤 범위내에서만 움직이는 것을 의미

==================================

그런데 기존의 활성화 함수들(saturating nonlinearity)은 Gradient Descent 관점에서 볼때 non-saturating nonlinearity 에 비해서(ReLU)에 비해서 수렴속도가 매우 느리다. Figure 1은 수렴속도를 tanh와 ReLU를 비교한 것.

그래서 활성화 함수는 모든 레이어 뒤에 온다.

### 2. Training on Multiple GPUs

이때에는 GPU가 매우매우매우 후졌기 때문에 당연히 모델 사이즈도 제한되어 있었다. 거기다 대고 ImageNet이 1200만장인데 이걸 학습 시키기는 너무 힘들다. 그래서 두장 썼고 그때 GPU들도 서로서로의 데이터를 읽고 쓰는 것이 가능했다. 그래서 GPU 두장에 네트워크 절반을 쪼개서 각각 넣어두되 특정 레이어에서만 GPU가 서로 통신하게끔 해뒀다.

​	-> 예를들어서, layer3는 두개의 gPU에서 나온 kernel map들을 모두 받으나 layer4는 자신과 같은 gpu에 있는 layer3의 kernel map만 받아오는 식이다.

이러한 connectiviyt에 대한 문제는 cross validation이 필요하나, 이러한 방식의 통신은 계산하는데 방해가 되지 않을정도 즉 통신하는데 생기는 병목현상을 해결할 수 있었다. 

구현된 모델에서는 3번째 레이어와 3개의 fc를 제외하면 나머지 레이어들은 서로 다른 gpu 간에 연결되어 있지 않다. 

### 3. Local Response Normalization

ReLU의 또다른 장점인데, saturation을 막기위한 input normalization이 필요없다는 점이 있다. 적어도 positive 인풋만 들어간다면 학습이 된다는 장점. 그럼에도 불구하고 양수방향으로 무한히 커질 경우 주변 값들을 무시하게 될 수 있으므로 정규화를 하는 것이 일반적인 관점에서는 좋다. 아래는 정규화 식이다.

![image-20200720205612799](C:\Users\MIRL\AppData\Roaming\Typora\typora-user-images\image-20200720205612799.png)



 k,n,α,βk,n,α,β는 하이퍼 파라미터이며, AlexNet에서는 각각 k=2, n=5, α=10^−4, β=0.75 / k=2, n=5, α=10^−4,β=0.75로 설정함.

즉 LRN은 lateral inhibition(측면억제)을 구현한 것인데, 한마디로 매우 높은 하나의 픽셀이 다른 픽셀에 영향을 받지 않도록 n번째 필터의 결과 값을 n-2 ~ n+2번째의 결과 값으로 정규화한 것이다. 지금은 Batch Normalization으로 이를 대체하여 사용한다!

어쨌든 이레이어는 첫번째와 두번째 conv layer 뒤에 온다!



### 4. Overlapping Pooling

Pooling 레이어는 같은 커널의 인접한 뉴런들을 요약해주는 역할을 한다. 전통적으로는, pooling은 overlapping하지 않고 이웃 뉴런들을 요약 했다. 즉 2*2 풀링 커널을 2stride로 설정하여 이미지의 크기를 반으로 줄여왔다. 

그러나 stride를 커널 사이즈보다 더 크게 잡으면 우리는 오버 래핑으로 풀링을 구현하게 되는데, 이는 오버피팅이 되기 좀더 어렵게 하는 것을 관찰하였다고 저자들은 말한다...진짜?? 어쨌든 이로인해서 성능 개선이 있었다 한다.

또 이 풀링은 max pooling을 사용하는데, 항상 LRN 뒤에 붙어있다.

![image-20200720211153345](C:\Users\MIRL\AppData\Roaming\Typora\typora-user-images\image-20200720211153345.png)

![image-20200720211631211](C:\Users\MIRL\AppData\Roaming\Typora\typora-user-images\image-20200720211631211.png)

## 출처: https://datascienceschool.net/view-notebook/d19e803640094f76b93f11b850b920a4/

그외에 Overfitting을 줄이기 위하여 augmentation과 dropout(학습시간은 길어지나 overfitting은 해결)을 적용하였다 한다

augmentation: 원데이터도 사용하며, 원 데이터(256*256)를 좌우반전한 상태에서 임의로 224*224 patch를 뜯는다./

​	-> 이렇게 해서 1개의 이미지로 2048개의 이미지를 만들어내어 학습함

​	-> test 시에는 이미지의 상하좌우 중앙의 patch를 뜯어서 이 이미지들과 이를 반전한 이미지를 합쳐 총 10장에 대한 예측치의 평균을 내었다.

뿐만아니라 학습이미지의 모든 픽셀 값에 PCA를 수행하고 여기에 mean=0 sd =0.1의 정규분포에서 샘플링한 값인 알파값을 PCA가 수행 된 픽셀들에 곱해준뒤 원래 픽셀에 더해주어서 이미지에 노이즈도 주었다.