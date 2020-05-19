# TResNet review

## Abstract

그동안 resnet50보다 작거나 같은 FLOPS 수를 가지고 더 높은 acc 가 아온 모델들이 많이 나왔다. vanila ResNet50이 그래도 acc-time trade off를 고려했을 꽤 괜찮았다. 

문제상황: 1. FLOPS 최적화를 하면서 생기는 bottleneck.

해결: 더 좋은 network design. TResNet 제시.(이전 conv 보다 더 좋고 효율적임) 같은 gpu throughput에서 더 좋았다.

## Introduction

2016년의 ResNet 모델은 학습하기도 쉽고,SGD로 학습해도 수렴이 빠르다. 

ResNet은 작거나 같은 FLOPS에도 좋은 accuracy를 낸다. 비록 FLOPS가 정확한 GPU속도를 표현하지는 않지만, GPU를 위한 sub optimal design에는 유효할 수 있다. 그러나 academy에서는 잘 사용하지 않아왔다. 감춰진 느낌

FLOPS가 최근의 네트워크에서 떡락하는 모습을 볼 수 있는데, 이것은 이 네트워크들이 1*1 conv 와 depthwise한 연산을 많이 사용하기 때문이다. 그러나 GPU들은 메모리 접근에 의해서 더 병목현상이 생기지 연산횟수로는 제약을 많이 받지는  않는다(특히 더 낮은 FLOPS를 가지는 레이어에서는 더더욱). 그럼에도 불구하고 FLOPS감소 이슈는 gpu throughput에 비해서 별로 주목되지 않았다.

또다른 이유로 ResNeXt나 MixNet과 같은 네트워크들은 multipath접근을 굉장히 많이 하는데, 이는 backprogation을 위해서 계속 저장되어 있어야하는 activation map을 많이 생성하게 되므로 batchsize를 크게 잡지 못하는 문제를 야기한다. 즉, gpu throughput이 감소하는 원인이 된다. inplace operation(하나 연산하는 동안 다른 거 연산하는 능력)도 multipath 접근을 많이 하게되면 제약이 많이 생기게된다.



그러므로 TResNet은 최신 딥러닝 모델들의 기법들과 저자들의 novelty를 많이 적용하여 디자인 되었다.

## Design

TResNet: TResNet-M, TResNet-L and TResNet-XL

TResNet architecture contains the following refinements compared to plain ResNet50 design: SpaceToDepth stem, Anti-Alias downsampling, In-Place Activated BatchNorm, Blocks selection and SE layers. Some refinements increase the models’ throughput, and some decrease it. All-in-all, for TResNet-M we choose a mixture of refinements that provide a similar GPU throughput to ResNet50, for a fair comparison of the models’ accuracy.

residual block: resnet의 핵심 아이디어. 그래디언트 정보를 좀 더 잘흐르게 만들기 위해서 일종의 지름길을 만들어주는 것.	LSTM의 forget gate와 비슷한개념.

refinements

stem design: 대부분의 뉴럴넷은 stem unit(최대한 빠르게 input resolution을 줄이기위한 컴포넌트)으로 시작하는데, 	ResNet50 stem 은 stride-2짜리 7*7 conv + max pooling으로 되어있다. (1/4로 줄어듬)

ResNet50-D stem 은 3개의 conv3x3 layer들로 되어있다. 이로 인하여 acc는 높아졌으나, throughput은 감소했다. 

논문의 저자들은 더 빠르고 중간에 끊기지 않고 매끄러운 stem layer(info loss가 적은)를 원했다. 

-> dedicated Space To Depth transformation layer를 사용하여 해결

-> 이것은 spatial data block들을 depth로 rearrange함.

그뒤에 1*1conv 를 붙여서 원하는 채널을 만들도록 해서 사용한다.

#### Anti-Alias Downsampling

AA는 기존의 downscaling기법들을 대제하기 위해 개발했다. shift equivariance를 개선해준다. speed-acc tradeoff를 더 개선 -> stride 2짜리 conv를 stride-1 conv + 3x3 blur kernel filter with stride 2를 결합하여 대체함.



#### In-Place Activated BatchNorm 

BatchNorm + ReLu 레이어들을 Inplace ABN으로 대체해줌. 배치놈을 구현한 것인데, activation을 single, inplace operation으로 구현함 -> 이를통해서 training시 필요한 메모리양을 줄였다.

Inplace ABN의 activation function으로 leaky ReLu를 사용함. (원래는 plain ReLU)

배치놈 레이어는 GPU메모리를 많이 잡아먹는 놈들 중 하나인데, 이를 INplace ABN으로 대체하게 되면 배치사이즈를 두배나 늘릴수 있다 -> gpu throughput을 향상시킬 수 있음

또, TResNet은 Leaky ReLU가 더 좋은 acc를 내는데 도움을 준다. Swish, Mish도 있지만, GPU많이 잡아먹는다.

#### Blocks Selection

ResNet34와 ResNet50은 같은 아키텍쳐를 가지지만, 하나의 차이점이있다. -> 34는 BasicBloack레이어 (2개의 3*3conv를  basic building block으로함)사용하고, 50은 Bottleneck(2개의 1 * 1conv + 한개의 3*3conv를 사용 basic building block으로써)블락 하나를 사용함./

Basic block이 bottleneck 레이어보다 더 적은 gpu를 사용하지만 더 낮은 성능을 가지기도한다.

TResNet에서 저자들은 이 둘을 섞어쓰는것이 speed-acc 에서 베스트 트레이드 오프라는 것을 휴리스틱하게 알아냄.

 Since ’BasicBlock’ layers have larger receptive field, they are usually more effective at the beginning of a network. Hence, we placed ’BasicBlock’ layers at the first two stages of the network, and ’Bottleneck’ layers at the last two stages.

이부분은 좀 더 읽어볼것.



#### SE layers - squeeze and excitation layers

SEblock의 computing cost 를 줄이고 acc는 높이기 위해서, 네트워크의 3단계에만 SE layer를 두었음. standard한 SE와는 다르게, TResNet SE placement and hyper-parameters are also optimized

SENet(Squeeze and excitation networks)-> 논문이 있음.

​	->2017년도 ILSVRC 우승 네트워크. conv 필터 하나하나가 이미지 또는 피쳐맵의 local을 학습하므로, local은 receptive field에 있는 정보들의 조합이다. 이런 local들을 reature recalibration을 함. 그게 바로 Squeeze and excitation.

-> 장점: 1. 각 피쳐맵에 대한 전체 정보를 요약하는 squeeze operation + 각 피쳐맵의 중요도를 스케일 해주는 excitation operation.을 가짐. 즉, 네트워크의 어떤곳에도 붙일 수 있도록 개발 됨

​	2. 이 SEblock을 사용함으로써 증가하는 파라미터의 증가량에 비해서 모델 성능 향상도가 매우크다. 왜냐하면 모델 복잡도와 계산복잡도가 SE block을 사용해도 증가하지 않기 때문이다.

#### code optimization

pytorch 사용. -> implementation detail이란 미명하에 무시할 수 없다. 현대 네트워크 디자인을 위해서 이런 것들도 중요

JIT compilation

-> 파이토치는 디폴트로 동적으로 코드를 실행하게 되어 있다. 그러나 JIT script 컴파일을 하면 네트워크의 일부분을 C++로 pre compile하여 사용할 수 있다. 이는 더 다양한 최적화와 퍼포면스 향상을 할 수 있게 해준다. 학습 할 수 있는 파라미터를 담지 않은 네트워크 모듈만 jit 컴파일함. (AA blur filter, Space to Depth module). 

이러면 GPU cost가 절반으로 줄어든다.

Fixed Global Average Pooling:

TResNet에서 GAP은 굉장히 자주 사용된다. SE 레이어와 fc 앞의 final pooling에서 사용.

AvgPool2d를 돌리는거 보다 torch의 View, Mean을 사용하는 것이 5배는 빨랐다.

Inplace Operations:

토치에서 inplace operation들은 복사 생성자 같은 것들을 사용하지 않고 직접적으로 텐서들을 수정한다.

이는 메모리 접근 비용을  줄이고 필요없는 activation map들을 만드는 것을 하지않는다는 장점이 있다. 

TResNet 은 특히 배치놈에서 이러한 inplace operation 을 많이 쓴다.

residual connection,SE layer, block final activation등등등...

이 것을 사용하여 TResNet은 큰 배치사이즈를 사용할 수 있게 되었다. 



## Results

