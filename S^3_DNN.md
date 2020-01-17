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

