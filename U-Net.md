# U-Net: Convolutional Networks for Biomedical Image Segmentation

### Abstract 
* Unet의 핵심 구조, data augmentation, ISBI challenge에서 좋은 성능을 보임.

* training 전략 제시-> data augmentation을 활용하여, 
		* annotated 학습 sample을 더 효율적으로 사용하는 training 전략을 보여줌.

* 네트워크 아키텍쳐-> contracting path에서는 context를 캡쳐하고,
		 * 대칭적인 구조를 이루는 expanding path에선 정교한 localization을 가능하게 하는 구조.


1. Introduction

*지금까지의 cnn -> 분명 visual recognition을 잘하긴 하지만, 
		i. training set의 크기와 네트워크의 크기에 제약을 받게됨.
		ii. 또한, 일반적인 cnn은 단일 클래스 분류문제를 해결하는데 적합함.

* 의료 영역에서 원하는 cnn-> 특히 biomedical precessing에서는, localization이 필요함.
			__그러므로, 각 픽셀마다 클래스 라벨이 할당되어야 함!(bold)__

	* Unet의 핵심 내용:
		1. conv. encoder에 해당하는 contracting path,
		conv. decoder에 해당하는 expanding path가 합쳐진 구조임.
		-> fully conv. + deconv.
		
		2. expanding path에서 upsampling 시에 좀 더 정확한 loclization을 위해
		contracting path의 feature를 copy and crop하여 concat 하는 구조.
		
		3. data augmentation.

* 기존에는 로컬영역에 sliding window를 해서 각 픽셀의 클래스 레이블들을 예측해왔음.
	* 단점:
		1. 네트워크가 각 로컬에 대해(patch) 개별적으로 실행되어야 하고 중복성이 많기 때문에 느림.
		2. localization과 context 사이에는 trade-off가 있는데, 
		    이는 큰 사이즈의 patch들은 많은 max pooling이 필요하게되어 localization의 정확도가 떨어지게 되고,
	  	  반대로 작은 사이즈의 패치들은 협소한 context만 보게 된다.
	
  * 이에 반해  unet은 contracting path되기 전의 feature들을 upsampling시에 Layer와 결합시켜서 high-resolution이미지를 만들 수 있다.
	  또한, downsampling시에는 64채널->1024채널까지 증가되며, upsampling시에는 1024->64채널이 사용됨.
	  그리고 모든레이어는 conv만 사용하고 fc는 없음.

	~~또한, unet은 segmentation을 할 때 overlab-tile 전략을 사용함.	~~
	~~주어진 인풋데이터 사이즈 크기가 매우 큰 경우에는 patch단위로 잘라서 input으로 사용하는데,~~
	~~over lab-tile 이해안됨...;;;~~



2. network 아키텍쳐

	* contracting path: 일반적인 convolution  네트워크
		 ㄴ 두번의 3*3 convolution을 반복 수행.  (unpadded conv.)
	1. activation: ReLU
	2. 2*2 maxpooling, stride = 2
	3. downsampling 시 2배의 feature 채널들을 사용.

	* expanding path: 2*2convolution(up-convolution)사용
	1. feature channel을 반으로 줄여서 사용,
	2. 대신 contracting path에서 matpooling 되기 전의 feature map을 crop하여
	up-convolution에 concat
	3. 그 후 3*3의 conv를 두번 해주는 것을 반복한뒤.
	4. ReLU사용.
	
	* final layer 1*1 convolution을 사용해서 2개의 클래스로 분류
	* -> unet에는 총 23개의 conv. layer가 사용됨.

3. training
	Stochastic gradient descent로 학습.
애매	*** GPU 메모리 사용률을 최대화 시키려면 batch size를 키우기 보다는
	input tile의 size를 크게 주는 것이 더 효과적이다.
	다만, batch size가 작아지면 학습 시간이 길어지는 가장 큰 요소가 되는데,
	이 때 momentum값을 0.99로 하여 과거에 학습 한 값들을 더 많이 반영하게 만들어서 가속 시킨듯??
	
	activation -> softmax
	loss -> cross entropy loss with w(x)
		l(x)는 정답 레이블 k에 해당하는 값을 반환하는 함수 
		-> cross entropy loss는 이 k값을 받아서 정답의 추정 값에 log를 씌운 형태.
		-> 그러나 unet은 여기에 w(x)를 곱한 값을 loss로 사용한다/
		w(x)-> 수학을 배제하고 직관적으로 생각하면, 각 object 사이에 존재하는 픽셀 x에 대하여,
		 object 간 거리가 멀어질 수록 weight(w)가 작아지는 메커니즘으로 이해.
	네트워크 초기화 - He initialization

	 data augmentation: 3by3 elastic transform matrix사용
		세포 세그멘테이션에서 유효한 작용을 하는 요소!(elastic deformation이 굉장히 유효..)
		이건 태스크에 따라 갈릴 수 있겠다.
	
4. experiments
	EM segmentation challenge에서 warping error sota.
	IOU로도 92%/77.5%로 sota

5. conclusion
	annotated image가 없는 상태에서 biomedical segmentation application에서 Unet이 좋은 성능을
	가질 수 있었던 이유는 Elastic 변환을 적용한 data augmentation 이었다.
	학습 시간- NVIDIA Titan GPU 6GB사용시 10시간.
	Unet 아키텍쳐는 다양한 테스크에서 사용이 가능 할 것으로 보임.
	
	**** Unet은 Fully Convolution과 Deconvolution의 구조로 된 U자형 아키텍쳐를 가지고 있으며,
	Localization의 성능을 높이기 위하여 Contracting Path의 Feature를 Expanding path에 concat하여 사용!
			


