# ML Study Week 3

## Section 7.1 - 학습 rate, Overfitting, 그리고 일반화 (Regularization)

### learning rate를 잘 정해야 하는것이 핵심

large learning rate - overshooting - learning rate 가 너무 커서 값이 극저점으로 다가가지 않고, 결과가 쓰레기값이 나오는 경우

small learning rate - 극저점에 도달하기 까지 너무 오래 걸린다

> 여러개의 learning rate 를 시도해봐야 한다. 기본적으로 0.01로 시작한다.

### gradient descent를 위해서 가끔은 Data를 가공(preprocessing)해야 한다

normalize 할 필요가 있다. => 값 전체의 범위가 항상 범위에 들어가도록 작업 하는 것.

#### Standardization

x'_j = (x_j - u_j) / a_j

```python
X_std[:,0] = (X[:0] - X[:0].mean()) / X[:0].std()
```

### Overfitting 

TF가 학습 데이터에 너무 딱맞는 모델을 생성해내는 것.

#### 해결방법

- more training data
- reduce the number of feature
- Regularization

#### Regularization 이란?

weight에서 너무 큰 값들을 가지지 않게 해야함

해결법 - [Regularization strength][regularization_strength]

[regularization_strength]: https://www.kdnuggets.com/2016/06/regularization-logistic-regression.html

## Section 7.2 - Training/Testing 데이타 셋

준비한 데이터 셋에서 training data set 과 test set을 나누어야 한다

만약 learning rate 또는 Regularization strength를 튜닝해야한다면, training data, validation data, testing data 로 나누어서 하는것이 일반적이다

### 데이터가 너무 큰 경우 => Online learning

학습된 모델에 추가적인 data set을 넣어서 학습을 시키는 것


## Section 8.1 - 딥러닝의 기본 개념: 시작과 XOR 문제

## Section 8.2 - 딥러닝의 기본 개념2: Back-propagation 과 2006/2007 ‘딥’의 출현

### Activation Function


