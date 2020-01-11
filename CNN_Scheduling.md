# Optimally Scheduling CNN Conolutions for Efficient Memory Access

### I. 문제 상황
---
  internal accelerator memory로는 CNN 알고리즘의 input feature map, output feature map, filter weight들을 동시에 버퍼링하기 힘들다.

### II. 이전의 문제 해결방안
---
1. 더 큰 SRAM 버퍼를 사용하는 것으로 문제를 해결하는 방식을 사용하였으나, SRAM은 매우 비싸다..   
   
2. dataflow 스케줄링에 의해 정해지는 data reuse(전통적인 캐시메모리에 의한 메모리 계층 구조)를 이용! (data reuse는 스케줄링에 따라 바뀜)  (dataflow = 어떤 work를 컴퓨터가 수행하기 위해 실행되는 각각의 프로세스들 사이에서 자료가 입출력되는 모습.)
(dataflow 스케줄링은 CNN의 종류와 크기에 따라 굉장히 변화가 커지게 됨. => CNN 종류에 따라 accelerator(메모리)가 적응하는 것이 어려운 이유.)

* etc. 2가 잘 이해 안됨.
CNN알고리즘의 가장 내부 루프의 working set을 그 시점에 가용한 internal storage와 fit시켜서 사용해 왔음.
???? -> 여기서 fit이 무슨 의미인가?
    internal과 external memory들이 최소화 되는 동안??
그러나 현존하는 CNN 모델들은 application managed buffer에 완벽하게 붙지 않음
CNN accelerator들이 가장 많이 쓰는 메모리 아키텍쳐 템플릿을 구성 -> internal storage requirement를 쓸데없이 크게 잡아서 sub optimal한 스케줄링을 하게 된다
---> 정확히 뭔소린지 다 이해 못함.

* etc. 근데 2를 발전시킨 것이 이 논문의 주제
정확힌 2번의 아이디어를 채택하되, 입벌리고 감 떨어질 때까지 기다리는 방식이 아닌, 좀더 능동적인 스케줄러를 만들어서 문제를 해결해보자!

### III. 이 논문의 contribution
---
1. dataflow 스케줄을 평가하기 위한 분석적인 메모리 퍼포먼스 모델 제시
   local memory requirement, overall external memory traffic면에서 평가
2. 최고의 dataflow scheduling을 만듬
3. 그 스케줄링을 임베디드 시스템을 위한 CNN accelerator 디자인(여러 CNN에 적용가능한)의 case study에 적용.

논문에서 다룰 문제
제한된 local buffer 용량이 주어진 환경에서 off-accelerator로의 memory access 최소화

### IV. Background
---
1. convolution에서 reuse하는 요소
   input feature map, ouput feature map, weight
2. 버퍼 재사용 기법.
   Cache
   application managed spm -> 실행 타임이 아닌 컴파일 타임에 데이터의 위치를 고정해야하는 메모리. 
    partitioning the local reuse buffer in a set of application-managed buffers
   spm 이용할거임
3. 데이터 지역성 최적화 기법
   reordering, titling

### V. Method
--- 
* dX(L): distance루프 L에서, 요소 X를 접근한 후 다시 접근하는데 필요한 iteration 수

* FX(Li): 루프 Li에서 배열 X의 footprint. 루프 L에서 사용된 X내부의 서로 다른 요소 갯수
  FX(Li) = FX(Li-1)* n(Li) / dX(Li)

    *n(Li): 루프Li의 iteration 횟수

* application managed buffer의 요구되는 크기(BX(Li))는 아래와 같이 계산된다.
Li가 X를 캐리한 경우: FX(Li-1)
캐리하지 않은 경우: BX(Li-1)

* Tx: memory traffic. 배열X에 접근하기 위해 off-accelerator memory로 접근한 바이트 수
TX = PX * FX(Li)*(ㅠn(Lj) 
 //ㅠn(Lj) = j가 i에서 N-1까지 n(Lj)를 모두 곱한값.
 TX는 Dataflow 스케줄링에만 영향을 받는다. local buffer와는 관계x (왜?)

  * PX: X 배열이 스토리지에서 사용한 크기를 바이트로 표현한 상수.
i: TX의 공식에서 i는 버퍼의 최대크기보다 작은 B(Li)의 최대를 만드는 i이다.

  * 그러므로 전체 Memory Traffic은 
    T = T input + T weight + T output, acc + T output, final

  즉, T를 이용하여 dataflow schedule에 대한 평가가 가능하다.

* Dataflow schedule selection procedure
  1. 각 배열의 참조마다 local buffering requirement 계산.
  2. 1에서 계산한 requirement들을 이용하여 특정 CNN layer를 분석해서 최상의 버퍼링 레벨 조합을 찾는다. (단 CNN의 3가지 배열들이 가지는 local buffer capacities들을 고려해야.)

* 이전 연구보다 뛰어난 점.
  * footprint calculation -> data reuse에 대하여 더 잘 고려할 수 있다.
    I, W, O 배열들에 대하여 독립적인 버퍼링을 고려할 수 있으므로.

  * 3개 배열에 대해 독립적인 버퍼링을 수행하므로, 전체 memory traffic의 관점에서  memory transfer를 항상 optimal하게 수행한다고 말 할 수 있다.
  * 