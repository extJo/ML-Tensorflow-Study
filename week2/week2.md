# ML Study Week 2

## Section 5.1 - Logistic (regression) classification의 가설 함수 정의

**Binary Classification**

ex) Spam detection, facebook feed, credit card fraudulent transaction detection

Binary 형태로 적용시키려면, 0, 1로 인코딩을 시킨다.


Binary Classification 에서는 Linear regression 을 적용 시키기 어렵다.

=> 가설이 1보다 커지거나 0보다 작아 질 수 있기 때문이다.

**Logitics Hypothesis**

sigmoid == logistic function (S 자가 누운듯한 그래프 모양)

g(z) = 1 / (1 + e^-z )

z = WX

H(x) = g(z) = 1 / ( 1 + e^-( W^T X ))

여기서 T는 W벡터의 형태에 따라서 달라지는 것.

## Section 5.2 - Logitics Regression의 cost 함수

이전에 사용했던 (Linear Regression) Cost function 을 사용해서 Gradient Decent 를 적용하면, 시작지점에 따라서, minimized 된 weight를 구할 수가 없다. (그래프 형태가 구불구불 해지기 때문)

**New Cost function for logistic**

```
cost(W) = 1/m * Sigma ( c(H(x), y) )

c(h(X), y)
    ㄴ -log(H(x))      : y = 1
    ㄴ -log(1 - H(x))  : y = 0

c(h(X), y) = -y*log(H(x)) - (1 - y)*log(1 - H(x))

Minimize cose = cost function 미분!
```

## Section 6.1 - Softmax Regression (Multinomial Logitics Regression) multinomial 개념 소개

[참고 링크][multinomial_classification]

**Multinomial classification**

많은 변수값을 기준으로, 많은 종류의 결과값들 중에서 선택하는 분류법.

각각의 변수는 당연히 독립적이라는 점.

## Section 6.2 - Softmax Regression (Multinomial Logitics Regression) cost 함수 소개

어떤 변수에 대해서 label값이 나오는데 label 값의 합은 1이다.

그럼 SOFTMAX 라는 함수는 뭔가?

1. 출력 label 각각의 값이 0~1 이라는점
2. 출력된 label의 합이 1이라는점

그럼 결국 출력된 label을 값을 확률로 보자!

'one-hot' encoding 을 통해서 확률이 가장 큰 녀석만 1로 바꾸고 나머지는 0으로 바꾼다.

그러면 cost function은 어떻게 하냐?

Cross-Entropy => D(S, L) = - sigma_i L_i log(S_i)

**Logistic cost VS corss entropy**

사실상 logistic cost 랑 cross entropy는 같습니당..

cross entropy가 logistic cost 를 일반화 한 것이다.



**참고 링크**

1. http://pythonkim.tistory.com/20
2. https://taeoh-kim.github.io/blog/bayes-theorem%EA%B3%BC-sigmoid%EC%99%80-softmax%EC%82%AC%EC%9D%B4%EC%9D%98-%EA%B4%80%EA%B3%84/


[multinomial_classification]: https://frontalnh.github.io/2018/01/18/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EA%B0%95%EC%9D%98%204%EA%B0%95%20-%20multinomial-classification/