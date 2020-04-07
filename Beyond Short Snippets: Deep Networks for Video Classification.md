# Beyond Short Snippets: Deep Networks for Video Classification

---

## Abstract

* 이전 연구에서 비디오에서의 영상 정보를 결합할 때 시도했던 영상 길이보다 더 길게 결합할 수 있는 딥러닝 네트워크를 제안하고 평가

  * 이를 위해 사용한 방법들이 하는 역할

    1. 이 태스크에 맞는 다양한 CNN 튜닝 값들을 찾고, time dependent한 convolutional 피쳐 풀링 구조들을  찾음.

    2. video를 frame단위의 시퀀스순으로 모델링.

       -> 이를 위해 LSTM알고리즘 사용.(CNN output과 연결됨.)

* Sports 1milion, UCF-101데이터 셋 사용.



## 1. Introduction

* Video Understanding은 temporal component를 추가함으로써(motion등의 정보.) 더 많은 정보를 제공할 수 있게한다.
  * 그러나 그만큼 컴퓨팅 cost가 크다. (even for processing short video clips since each video might contain hundreds to thousands of frames, not all of which are useful)
* 이를 naive하게 구현하려면 video frame들을 이미지로 취급, CNN알고리즘들을 적용하여 각 예측 값들의 평균을 내어 video level의 prediction을 만들어 낼수는 있음.
  * 그러나 각 video의 frame들이 전체 video를 대표할 수 없으므로, 이를 사용한 방법은 정확한 정보를 만들어낼 수 없다.

* 그러므로 video classification을 하려면 video의 global description을 알아내는 것이 중요하다!

  * 그러나 비디오 길이는 가변적이고, 이것을 고정된 수의 파라미터들로 모델링해야하므로 쉬운 것은 아님.

  * 이를 해결하기 위해 두가지 접근을 만들었고, 평가함.

    1. feature pooling networks :  process each frame using a CNN and then combine frame-level information using various pooling layers.

    2. recurrent neural networks: derived from Long Short Term Memory (LSTM)  units, and uses memory cells to store, modify, and access internal state, allowing it to discover long range temporal relationships.

       Like feature-pooling, LSTM networks operate on frame-level CNN activations, and can learn how to integrate information over time

* video classification에서 중요한 문제 -> motion information을 얻는것

  * 이전 연구: input으로 frame stack을 사용.-> 연산이 비대해짐
  * 현 연구: 1초에 오직 한 개의 frame만 처리하기->implicit motion information will be lost.
    * 이를 보강하기 위해 explicit motion information을 추가함. -> optical flow.(allows us to retain the beneﬁts of motion information(typically achieved through high-fps sampling) while still capturing global video information)



* 요약: 
  1. global video level descriptor를 얻을 수 있는 CNN 아키텍쳐 만듬. -> 그리고 이 아키텍쳐가 더 좋음을 증명.
  2. FEATURE pooling architecture와 LSTM architecture가 파라미터들을 시간에 따라 공유 -> 파라미터값 개수가 늘어나지 않음
  3. optical flow 이미지드리 video classification에 있어서 성능을 크게 향상 시킬 수 있음을 보임.





## 2. Related Work

* 전통적으로 video recognition연구는 꽤 성공적이었음:(global video descriptor를 잘 얻어냄)

  * hand crafted feature 들 덕분(Histogrom of Optical Flow등)

* CNN으로 video recognition 시도를 해본 사례가 거의 없었음. 이때 당시엔.

  * motion feature 들을 학습 시키는 것이 힘들었음 ->  할 순 있었지만, 짧은길이의 프레임에 대해서만 가능했음

    -> global 한 정보를 얻을 수 없음

  * 그래서 LSTM을 사용해서 spatial-temporal feature들을 학습한 것.



## 3. Approach

* 각 video frame 처리에 사용된 CNN 아키텍쳐들: AlexNet, GoogLeNet.
  * Alexnet: 220*220 사이즈를 인풋으로 받고 11, 9, 5 크기의 conv layer들을 통과시킨다.(사이에 maxpooling, local contrast normalization)
  * GoogLeNet:



* feature pooling architectures
  * 주로 temporal feature pooling은 video classification에서 사용되거나 bag-of-word에서 사용됨.
  * 대개의 경우 이미지 혹은 모션 기반의 feature 들은 매 프레임마다 computed, quantized, pool됨
  * pooling 이 무조건 max pooling 이어야만 할 이유는 없다!
    * average, max pooling 둘중 뭐가 더 나은가?
      * fc, average pooling 모두 계산량이 커지기 때문에 학습을 효과적으로 할 수없었음
    * max pooling이 좀 더 sparse함. -> 그러므로 아래부턴  main feature aggregation으로  maxpooling사용할 것.



   * bag of word와는 다르게, 앞쪽레이어의 gradient 값들이 이미지에서 더 효과적인 feature 들을 뽑아오는 것을 확인

   * 이를 착안하여 max pooling을 총 5가지의 방법으로 variation을 줘서 실험해봄.

        * Conv pooling: conv layer 거친 후에 max pooling 보냄.

          전체 연산에서 conv layer의 출력의 time domain spatial information이 보존됨.

        * Late pooling: 일단 maxpooling 안준 상태로 2개의 fc까지 conv feature 들을 통과 시킴. high level info를 제공.

        * Slow pooling:위계적으로 frame level 정보들을 합침. 한마디로 풀링을 여러번. 이러면

          네트워크가 temporally local feature들을 여러개의 프레임들로부터 high level information을 combine 하기 전에 그룹화 시킬 수 있다. 

        * Local Pooling: slow pooling과 비슷하게, frame level feature들을 일단 풀링하여 합친다. 그 다음은 slow pooling과는 다르게, max pooling은 한번만 하고 fc를 두번 보낸다.(shared parameters)  마지막으로 softmax하나.

          max pooling을 한 번 덜하게됨으로써, temporal 정보의 잠재적 손실을 줄일 수 있다.

        * Time Domain Convolution: extra time-domain conv layer를 사용(pooling 전에). 풀링은 temporal domain에서 이뤄짐. 

          small temporal window이내에 있는 프레임들 간의 local relationship들을 포착하는 것이 목적.

* LSTM architecture
  * 시간적 순서와 상관없이 representation을 만드는 max-pooling과는 다르게, rnn을 사용하여 cnn activation들의 순서를 명시적으로 고려하는 것 제안
  * 일반적인 LSTM을 사용한 것으로 보이나, five-stacked LSTM layer를 사용함.



* Training and inference

  * max-pooling모델들은 cluster using Downpour Stochastic Gradient Descent를 사용하여 최적화되었음

    * lr = 1e^-5, momentum=0.9, decay=0.0005
    * momentum이란 더 빨리 학습되도록 gradient 값 바꿔주는 hyper parameter 만드는것.

  * LSTM학습에도 같은 optimization 기법사용. 단, lr=N*10^-5 N은 프레임 갯수.

  * CNN단은 transfer learning 사용(Alexnet, GoogLeNet, ImageNet, Sports-1M videos)

    * Network Expansion for Max-Pooling Networks  (무슨 방식으로 expansion한건지는 파악 못함.)

      * 아마도 프레임 1개로 봤을때와 여러개의 프레임을 봤을때를 말하는 듯.

      * 풀링이 서로 웨이트를 공유하는 CNN tower 이후에 이뤄지므로, one-stage모델을 two-stage모델로 확장시키는 것이 가능. (파라미터들이 비슷하므로.)

      * 어쨌든 작은 네트워크를 크게 확장해서, fine tuning하면

         achieve a signiﬁcant speedup compared to training a large network from scratch. 

    * LSTM training: 

      1. 클립 당 레이블로 학습하지 않고, 프레임당 레이블로 학습.

      2. gain값 g를 각 프레임마다 있는 backpropagated gradient에 적용.

         g는 0~1에서 0~T번 프레임에대하여 linear interpolate된 값.

         이 g를 사용해서 LSTM의 뒷 단 레이어에서 내려진 더 정확한 prediction값을 강조할 것!

         ​	-> 뒷단에서 내린 예측이 더 유효하니까, 거기에 가중치를 주겠다는 뜻으로 이해.

         단순히 뒷단 레이어에 g를 1이나 0으로 주는 거 보다 interpolate 하는 것이 더 빠르고 정확한 예측결과를 뱉음.

    * LSTM inference

      1. last step T에서 prediction return
      2. prediction을 매순간마다 max pooling 하는 경우
      3. 매순간마다 prediction 값을 합하여 max값 리턴
      4. 매순간마다 prediction을 g로 linear하게 weight를 주어서 합한 후 max값 리턴

    모두 비슷한 정확도를 보였으나, 4번이 그 중 제일 나아보임.



   * Optical Flow
        * encodes the pattern of apparent motion of objects in a visual scene
        * 논문에서 제시된 네트워크의 처리속도가 1fps 밖에 나오지 않기 때문에, 이 안에서 apparent한 motion information을 사용하지 않음.
        * 그래서 optical flow image들을 같이 학습 시킴.
        * raw img frame들로 학습한 모델의 가중치로 모델을 초기화하면 optical flow classification에 도움을 줄 수 있음을 발견.
             * 두개의 인접한 frame들로부터 계산. image frame과 같은 방식으로 사용.



## 4. Results

*  Hit K: indicate the fraction of test samples that contained at least one of the ground truth labels in the top k predictions

* Feature pooling 실험: Alexnet 기반 Sports 1M. 
