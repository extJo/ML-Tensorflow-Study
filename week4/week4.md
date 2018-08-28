# ML Study Week 4

## Section 9.1 - XOR 문제 딥러닝으로 풀기

이전에 XOR문제를 어떻게 해결하느냐가 핵심이었음.

Neural Net 을 통해서 XOR 문제를 해결 할 수 있다.

왜 Logistic Regression을 쓰는가? => XOR 로 결국 0 또는 1로 표현만 가능하면 되기때문

## Section 9.2 - 10분만에 미분 정리하기

https://www.youtube.com/watch?v=oZyvmtqLmLo

## Section 9.3 - 딥넷트웍 학습 시키기 (backpropagation)

여러개의 레이어에 대해서 학습을 어떻게 시킬것인가?

1. 일단 우리가 임의의 weight와 bias를 각각의 가설에 설정할 것이다.
2. 그를 통해서 우리가 forward 즉 다음 layer에 들어갈 값을 준다.
3. 그 이후에 정해진 값들을 통해서 backward 즉 뒤로 상수값들을 다시 전달하면서 우리는 weight 와 bias 값들을 재 조정 할 수 있다.


> 미분을 통해서 상수들을 통해 우리가 원하는 값을 구할 수 있다.

왜냐면 각각의 퍼셉트론에 대해서 가설이 무엇인지 알고, 해당하는 가설을 그래프 형태로 풀어내어서 미분으로 결과값들을 도출 할 수 있다.

그 이후, 다음 layer에서 어느 값에 영향을 미치는지 알기떄문이다.

결국 수식을 어떻게 그래프로 잘 나타내느냐의 문제.

[Back Propagation Chanin Rule Reference][back_propagation_chain_rule]

[back_propagation_chain_rule]: http://aikorea.org/cs231n/optimization-2/