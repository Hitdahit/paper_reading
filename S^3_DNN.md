# S^3 DNN: Supervised Streaming and Scheduling for GPU-Accelerated Real-Time DNN workloads

### Abstract
* limited attention has been given to optimizing the execution of GPU accelerated DNN workloads at the system level.
* 즉, GPU로 가속화 한 DNN Workload들의 실행을 시스템 레벨에서 최적화
* S3DNN은 이를 위한 solution.
* S3DNN 구성:
  * governor: 선택적으로 시스템 전체에 퍼져있는 요청을 모아줌. -> smart data fusion을 위함
  * novel supervised streaming and scheduling framework
* S3DNN은 DNN Workload들의 특이점을 찾아냄. -> DNN이 GPU resource를 줄이는 utilization pattern

### I. Introduction
* 임베디드 시스템에서 intelligent decision을 내려야 하는 경우 발생(IoT등)
  * DNN을 이용한 decision making이 효율적 -> 데이터가 작아도 기계가 알아서 학습가능.
  * 그러나 decision making을 위한 layer들을 충분히 쌓을 수 없음 -> 컴퓨팅 자원의 부족. -> 딥러닝이 GPU에 의존하게 된 이유.
* DNN Workload들의 특정 feature를 찾아내어, GPU-accelrated DNN Workload들의 실행을 시스템 레벨에서 최적화 하는 것이 S3DNN의 목표
  * S3DNN이 해결할 문제점
    1. how to guarantee real-time performance
    2. while maximizing system throughput
    3. and resource utilization to mitigate the inherent resource constraint
---
* S3DNN이 위의 문제들을 해결한 방법
  * 1의 해결:
     *  extends a classical deadline-aware real-time scheduler(Least Slack First) (여러개 DNN 인스턴스들에 대해 우선순위와 스케줄링을 정하기 위해.)
  * 2, 3의 해결:
    * schedules workloads in the granularity of GPU kernels and dynamically aggregates kernels that underutilize GPU resources to enable better concurrency.
  * 1과 2의 동시 해결: 단순히 LSF를 합치거나 커널 동시성을 보는 것 외에도 novel supervised streaming and scheduling framework를 개발. -> DNN의 특이한 workload들에서 기인.
  
* DNN-based object detection workload들 -> STAGED GPU resource utilization pattern을 보임. 즉, cost가 큰 layer를 지나고 나면 그다음 layer에서는 점진적으로 GPU resource가 감소하는 모습을 관찰할 수 있었다!

---
임시 정리본
---

s3dnn

1. Introduction.
s3dnn은 multicore cpu, GPU를 가진 환경에서 사용하도록 개발됨.

->s3dnn은 input video들과 GPU를 잇는 middleware로써, GPU위에서의 O.D workload execution을 최적화 함.

frontend-backend framework임.
front: 모든 영상처리 요청들을 백엔드에게 forwarding하는 역할.
back: daemon process로써 request들을 취합하고 computation 실행함.

	이 back은 2개의 컴포넌트로 이뤄짐
	1.governor ->시스템 레벨에서 인풋 영상 프레임들을 선택적으로 fuse(data fuse, 합침)하거나, frame들을 합쳐서 여러개의 DNN 객체로 만듬.
	2. supervised streaming and scheduling module(s3scheduler)
	-> prioritize and schedule the assigned DNN instances to maximize both realtime
	and thourghput perf. (through optimizing execution concurrency)
	real time과 throughput 퍼포먼스를 모두 최적화 하기 위해, 할당된 DNN 인스턴스들을 우선적으로 처리하고 스케줄링함.
	(concurrency 실행을 최적화 함으로써 구현함,)

S3DNN is middleware so that it is easy to be integrated with any existing DNN processing framework.

conducted experiments
-> yolo를 이용해서, real time, throughput performance를 개선하는 실험을함.


throughput improved 18%.
in real time performance, 거의 100% deadline meeting ratio를 보장.
현재 무거운 workload들이 이미 있다는 전제하에, s3dnn은 deadline을 만족하는 선 내에서,
40프로 정도를 개선하였음.

2. Background
Focus on GPGPU to accelerate DNN workloads in a system consisting of multiple discrete GPUs
and a multicore CPU. so use CUDA as a GPU programming model.
여러개의 독립적인 GPU와 멀티코어 CPU를 가진 시스템 내에서 GPGPU로 DNN workload들을 더 빨리 처리하는데 집중하기 위해
CUDA와 GPU 프로그래밍 모델을 사용

2-i. The CUDA programming model
GPGPU application은 아래와 같은 execution flow.<br>
1. init GPU device<br>
2. allocate device memory<br>
3. transferring data from host memory to device memory<br>
4. launching the computation work on GPU<br>
5.copying results back to host memory<br>
6. free device memory and close the device.<br>

이 논문을 읽기 위해 알아야 할 단어
Context: GPU의 가상주소공간 혹은 CUDA application을 위해 GPU 장치에 (GPU 초기화 단계에) 만들어진 공간<br>
	CUDA는 MultiProcess Service이나, linux만 지원하고 그외에도 여러 제약 조건이 많다. (so needs more common scenario)

GPU device에서
SM: Streaming Multiprocessor의 약어. Nvidia GPU 하드웨어 안에 있는 내부 유닛
	low-end gpu got 1 sm
	high-end gpu got 24sm
kernel, Thread, Thread block, Grid:
	kernel - GPU에서 실행되는 코드. 여러 개의 CUDA스레드들로 이뤄져 있으며 병렬처리된다.
	Thread - cuda programming의 기본 실행단위
	Thread block - 스레드로 이뤄진 배열.
	Grid - Thread block이 모인 것.
block, grid의 차원은 programmer가 정할 수 있다.
CUDA stream: CUDA가 메모리 복사의 latency를 숨기고, 서로 다른 독립적인 연산들로부터 kernel launch를 하기 위해 사용하는 기술.

->effect==  concurrent kernel execution을 더 많이 할 수 있게 해줌. 각 커널들이 under utilization 하고 있으면
여러개의 커널이 하나의 GPU에서 동시에 수행되게끔 만들어줌.

2-ii. DNN model
DNN := dataflow graph -> 노드들(layers)은 필수로 array-based computation을 해야한다.
각 layer들은 배열의 집합을 가짐. (feature map)

this paper focus on efficient execution of DNNs for real-time object detection.


3. Motivation
measurement based case studies
target: YOLO  (SOTA, capable of taking continuous frames from a video file
and able to label)
experiment: run multiple YOLO instnaces simultaneously on an NVIDIA QUADRO 6000(14SM)
input: road drive video from KITTI vision benchmark suite.
how to measure throughput of the system:FPS of all processed frames, pMiss: deadline miss rate
pre recorded can be processed at a higher FPS.

once previous img has finished being processed, the next image have to be processed
immediately afterwards without waiting for the release time!!

3-i.GPU Usage Pattern For DNNs
used default yolo setup.  16layers of three types(conv, poolong, region) (see fig2)
measured each layer how big is the output size.(temporarilly stored in CPU flobal memory)
number of thread block, execution time in processing each frame.

and this measurement shows us Fig2.
graph shows staged patteren.after convolutioning.(in terms of big sight)
and final layers shows small input data and light computation -> under utilization of gpu.

dnn generates more fine grained(결이고운)feature maps at early layers.
and prunes details gradually -> depth increase to get a higher abstraction
(추상화 -> more faster processing) (find simple features.)

3-ii. Data Fusion
batch: fetch multiple images to process them on one pass is a common optimizing method.
->batch can reduce I/o wasting time moreover some case can improve GPU utilization

-application레벨의 데이터 data fusion(데이터 융합)
Cafe와 같이 data fusion functionallity를 포함하는 DNN은 이미 많이 구현되어 왔음.
DNN 멀티 태스킹 환경에서, 여러 개의 이미지를 하나의 객체로 만들어 batch하여 throughput을 개선하는 것이 가능.
위의 기법은 특히 자율주행과 같은 환경에서 유리.(여러개의 카메라와 센서들이 자동차에 장착되어 있음)

저자는 YOLO를 변형하여 한번에 4개 이상의 영상을 처리하는 
batch기반의 프로세싱을 지원하도록 만들었다.
4개 이상의 vanilla YOLO프로그램을 바탕으로, 각 프로세스당 하나의 인풋 영상을 처리하도록 하였다.
인풋 영상들은 모두 25로 모두 같은 FPS를 가지는 영상들로 사용하였다.

이 때의 throughput을 조사하기 위해 위해서 말했던 것처럼 평균 FPS를 사용하였다.
performance를 조사할 때엔 pMiss를 조사하였다.
그 결과가 Table 1에 있고, 이표를 보면 바로 알 수 있듯이 인풋 영상이 많아질 수록 data fusion이 더 효율적이게 되어
throughput을 개선하게되었다 -> (per video에 주목.)  (그러나 반대로 under utilization)
그러나 deadline miss율이  백프로가 되었다. (fusing 과정과 processing이 동시에 이뤄지는데다가, 비디오끼리의 FPS가 다르면 더 미스율이 높아짐)


-system level solution
위에서 설명한 application level fusion이 throughput 개선에 효과가 좋다는 것을 확인.
그러나 서로 다른 DNN 프로세스를 고려하지 않기 때문에, 최적이라 할 수 없다.
게다가 모든 상황에서 DNN프로세스가 계속 많다는 보장도 없으므로 throughput도 항시 최적이라 할 수 없다.
무엇보다 데이터 우선순위에 따른 preemption과 같은 것들을 해결할 수 없다.


3-iii. Kernel Scheduling and Concurrency
CUDA stream technique를 이용하여, 질서를 가지고 태스크를 스케줄링 하는 것도 throughput(c처리량)을 높이는 방법이 된다.
한번에 여러개의 커널들을 execute할 수 있기 때문. deadline을 지키면서.
-enabling concurrency using CUDA streams
->각 커널별로 다른 쿠다 스트림에 넣어서 동시에 커널을 execute하는 것이 전체적 perf를 올릴 수 있꼬,
특히 각 커널들이 gpu resource utilization을 못하는 경우에 더 좋다. 그럼에도 불구하고 cuda stream은 
동시성을 올리기 위해 사용하기에는 각 커널의 타이밍 예측을 방해하여 예측 정확도를 저하시킨다.(자세히 읽기 필)

-supervised CUDA streams with scheduling.
다시 읽기.

insight i: DNN의 GPU사용패턴은 staged. 레이어가 앞에 있는 것은 high resource를 요구하나, 뒤로갈 수록 그럴필요가 없어져 under utilization이 됨

insight ii:지금까지 application레벨에서의 data fusion, 즉 batch를 이용하여 throughput을 개선해왔다.
자율주행자동차와 같은 많은 instance들이 DNNinstance를 계속 사용하는 경우나 input이 많은 경우 정말 throughput 개선에는 효과적이나,
정작 중요한 deadline miss율이 높아지는 사태가 일어남.

insight iii:각 커널별로 다른 쿠다 스트림에 넣어서 동시에 커널을 execute하는 것이 전체적 perf를 올릴 수 있꼬,
특히 각 커널들이 gpu resource utilization을 못하는 경우에 더 좋다. 그럼에도 불구하고 cuda stream은 
동시성을 올리기 위해 사용하기에는 각 커널의 타이밍 예측을 방해하여 예측 정확도를 저하시킨다.

insight iv: staged 특징을 보이는 DNN workload concurrency(동시성)을 최적화 하기 위해, 디폴트 CUDA streaming을 쓰는 것 대신, 
supervised streming을 수행하는 것이 더 나을 수 있다. 최대의 동시성을 내기 위해 DNN의 각 layer들이 요구하는,(그리고 계속 변화하는)
 컴퓨팅 리소스 요구량을 고려해서 커널들을 스케줄링하므로,

4. Design and implementation
4-i.design overview
4-ii.system-level data fusion
4-iii.supervised streaming and scheduling
5. evaluation and appendix.

