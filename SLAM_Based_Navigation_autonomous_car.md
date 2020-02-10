# SLAM
1. Introduction
    * 사용한 것들.
        * ROS
        * urg_node: LiDAR센서를 위한 ROS의 driver 라이브러리. 
       * hector_slam: hector_mapping 노드를 가지고 있는 패키지. 
	        (hector_mapping노드는 LiDAR 센서를 이용하여 2Dmap을 생성)
       * hector_slam_example: launch, configuration파일들을 모아둔것.(LiDAR센서 configuration에 쓰임)
       * navigation_stack: 로봇들을 2Dmap상에서 navigate하기 위해 필요한 함수들이 담겨있는 라이브러리.
        * global_planner(최단거리 탐색함수), move_base(로봇의 모터에게 속도값을 쏴주는 패키지. amcl)
       * teb_local_planner: base_local_planner(마찬가지로 navigation_stack에 있음)의 플러그인을 구현한 패키지.
	현재 route를 LiDAR센서를 이용해서 보여줌.

  * target problem: LiDAR센서를 이용한 SLAM맵을 기반으로한 자율주행 자동차의 구현.
	-> map 만들기/ 최단경로 찾기/ local obstacle 피하기/ management_unit과 연결하기의 문제를 구현해야함.

2. Background

  * SLAM: Simultaneous Localization And Mapping.
	로봇이 처음보는 지역(environment)에서 map을 생성하고 navigating을 할 수 있도록 해주는 알고리즘.<br>
	Landmark extraction<br>
	data association<br>
	state_estimation<br>
	state_update<br>
	Landmark update   의 순서로 문제를 해결.<br> 
    -> 각 단계들은 로봇이 처한 environment별로 여러가지 방법으로 구현될 수 있다.

	ㄴ6pg SLAM프로세스 보면 이해가 빠름.
    	pose estimation을 EKF SLAM으로 해결하는 과정

    R: robot위치
    M: set of the landmark states
    state vector의 평균: x=[R, M]^T<br>
    " 의 공분산 행렬: P = [[P_RR, P_RM],     		[P_MR, P_MM]]

* EKF-> 위의 state vector의 평균 x, 공분산 행렬P를 Gaussian Variable을 이용하여 모델링. 
어케하는지는 https://www.youtube.com/watch?v=4OerJmPpkRg 참고.

이후 robot이 움직이면서 x <- f(x, u, n)  (단, u=control vector, n= perturbation vector)로,
odometry가 바뀌고, 이를
y=h(x)+v의 식으로 (y: noisy measurement, h: observe function, x: full state, v: measurement noise)
position을 업데이트 후 EKF에 저장하는 방식으로 pose estimation을 하는 듯 하다.

3. implementation

   1. Mapping(+pose_estimation)
	hector_slam라이브러리를 이용하여 구현.<br>
	마지막으로 센서가 스캔한 데이터(endpoint z)가 point cloud로 변환됨(platform orientation을 사용함.)<br>
	    -> endpoint z 에 기반한 filtering을 사용.(scan matching)
	    pose estimation:
		1. endpoint를 map에 사영(현재의 pose estimation 기준)
		2. endpoint에서 map의 사용확률 그래디언트 측정.
		3. pose estimation 정제를 위한 Gauss-Newton iteration.<br>
   2. navigation<br>
	global planner: 차가 예측된 시작점에서 목적지까지 가는데에 유효한 global plan을 탐색함<br>
		navigation_stack 라이브러리의 global_planner를 사용.<br>
		Dijkstra(O(n^2))    (음수간선 쓰지 말것.)<br>
	local planner: local environment에 적절한 반응을 하게함.<br>
		uncharted 장애물들이 global route위에 있을 수 있음.<br> 이를 애커만 조형장치와 센싱값을 가지고 teb_local_planner라이브러리를 이용한
		local path를 찾아냄.
    3. localization: <br>
    지도상에 자신이 어디에 있는지 알아내는 알고리즘
	amcl방법. 차후 업데이트.

