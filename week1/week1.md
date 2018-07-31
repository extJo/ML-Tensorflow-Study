# ML Study Week 1

## Section 1.1 - 기본적인 ML 용어와 기초 설명

**기초 용어**

```
label - 예측하는 실제 항목 (y) => 기본 선형회귀의 y 변수

feature - 데이터를 설명하는 입력변수 (x_i) => 기본 선형 회귀의 {x_1, x_2 .... x_n} 변수

example == instance - 데이터 (x)의 특정 인스턴스
        ㄴ 라벨이 있는 예 : 학습 시에 사용
        ㄴ 라벨이 없는 예 : 새 데이터를 예측하는데 사용

model - example을 예측된 Label(y')에 매핑하는 함수 => 학습되는 내부 매개변수에 정의 됨

bias - 편향

weight - 가중치
```

**Learning 은 두가지로 구분할 수 있다**

1. Supervised learning
    - 결과가 미리 정해져있는 데이터를 보고 학습 하는 경우이다 (learning with labeled example or training set)
    - data set을 통해서 ML을 학습 시키면, 어떤 model 이 생성될 것이다.
    - 그 이후에 label 값을 알지 못하는 feature를 통해서 Label을 구하려고 하면 ML내부에서 정의된 model을 이용해서 결과값을 도출 해 준다.
    - Type of Supervised Learning
        - predicting final exam score based on time spent => regression (회귀 분석)
        - pass/non-pass based on time spent => binary classification  (이진 분류)
        - letter grade (A, B, C, D, E, and F) based on time spent => multi-label classification (멀티 라벨 분류)

2. Unsupervised learning
    - 결과가 정해져 있지 않은 data set (un-labeled data)을 통해서 스스로 학습을 하는 경우 
    - 예시
        - google news grouping
        - word clustering

## Section 2.1 - Linear Regression 의 Hypothesis 와 cost

### Linear Regression 이란?

https://medium.com/qandastudy/mathpresso-%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EC%8A%A4%ED%84%B0%EB%94%94-4-%ED%9A%8C%EA%B7%80-%EB%B6%84%EC%84%9D-regression-1-6d6cc0aaa483

### (Linear) Hypothesis

1. 학습한 데이터에 대해서 H(x) = Wx + b 형태로 모델의 가설을 세운다.
2. 이후에 여러 model의 가설에 대해서 학습한 데이터와 오차가 가장 적은 model의 가설을 채택한다.

그러면 어떻게 오차가 가장 적음을 찾아내는가? => **Cost function (Loss function)** 을 통해 찾아낸다.

## Section 3.1 - Linear Regression의 Cost 최소화 알고리즘의 원리

### Minimized Cost function (Loss function)

- Cost Function
    
    cost(W, b) = 1/m * (i = 1 => m)SUM((H(x_i) - y(y_i))^2)

    결국 Cost function 은 (예측 label - 실제 label)^2 의 총 합 에서 총 개수를 나눈 것 이다.

- Gradient descent algorithm (경사 하강법 알고리즘)
    - 많은 최소 문제법에서 사용됨
    - cost function이 주어지면, W(weight)와 b(bias)가 가장 작을때를 찾아낸다
    - 다항 cost function 에서도 사용 가능하다
    - 작동원리
        - 아무 지점에서 시작한다. 그 이후에 Weight와 bias 를 변경시키면서 cost(W,b)를 경감시킨다
        - 계속 반복하다보면, Minimized Cost에 도달한다
    - 미분을 이용해서 구하자!

- [Convex function][convex_function]
    - convex function의 모양을 가진다면, 경사 하강법 알고리즘을 통해서 항상 Minimized Cost를 찾아낼 수 있다.

## Section 4.1 - Multi-variable Linear Regression

여러개의 input, 즉 여러개의 feature 에 대해서는 어떻게 모델을 만들어 낼 것인가?

Hypothesis => (x_1, x_2, x_3) = w_1 * x_1 + w_2 * x_2 + w_3 * x_3 +b

Cost function => cost(W, b) => 길어서 생략

**그러면 길어질때마다 계속 길게 쓸것인가? => 놉, matrix 를 이용하자!**

선형대수학에서 이용하는 [matrix multiplication][linear_algebra_matrix_multi] 만 이용한다!

### 왜 강의에서는 bias 이야기를 하지 않을까요? - [다른 문서][why_omitted_bias]의 4p를 참고하였습니다.

> bias 가 생략되는 경우는 만약 Y의 행렬식이고, 적어도 한 회귀에 대해서 상관관계가 있기때문에 생략합니다.

> 만약 다중 회귀가 통제 변수를 포함하면, 우리는 적절한 통제가 이루지지 않은 생략된 요소가 있는지 질문 할 필요가 있다. 즉, 제어 변수를 포함시킨 후에도 오류 항이 관심변수와 상관되는지에 대한 여부를 물어볼 필요가 있다.


## 볼만한 참고 자료

1. 오차 추정법 관련 잘쓴 글

    https://medium.com/qandastudy/mathpresso-%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EC%8A%A4%ED%84%B0%EB%94%94-3-%EC%98%A4%EC%B0%A8%EB%A5%BC-%EB%8B%A4%EB%A3%A8%EB%8A%94-%EB%B0%A9%EB%B2%95-7d1fb64ea0cf

    https://medium.com/qandastudy/mathpresso-%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EC%8A%A4%ED%84%B0%EB%94%94-3-5-%EC%98%A4%EC%B0%A8%EB%A5%BC-%EB%8B%A4%EB%A3%A8%EB%8A%94-%EB%B0%A9%EB%B2%95-2-e23e08d95cc3

2. 이 강의에서 다중회귀에서 bias를 생략한 이유를 좀 더 알고싶을떄.

    http://blog.naver.com/PostView.nhn?blogId=astiminjjang&logNo=220733648602

    http://m.blog.daum.net/dataminer9/172

[convex_function]: https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221148661854&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F
[linear_algebra_matrix_multi]: http://twlab.tistory.com/10
[why_omitted_bias]: https://www.sas.upenn.edu/~fdiebold/Teaching104/Ch9_slides.pdf