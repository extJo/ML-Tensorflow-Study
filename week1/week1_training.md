# ML Study Week 1

## Section 1 - 2 TensorFlow의 설치 및 기본적인 operations

**TensorFlow 란?**

data flow graphs 를 이용해서 numerical computation 을 위한 오픈소스 라이브러리

**Data Flow Graphs 란?**

node 와 edge로 이루어진 그래프.

node == mathematical operation, edges == data array == tensors

**Hello World 출력하기**
```python
import tensorflow as tf

# 문자열이 든 하나의 constant 노드 생성
hello = tf.constant("hello, TensorFlow!")

# data flow graph를 실행하기 위해서는 session을 생성해야함
sess = tf.Session()

# session을 실행시키면 실행이 된다!
print(sess.run(hello))
```
> Output =>  b'hello, TensorFlow!', 여기서 b는 Bytes literals 를 뜻한다.

### TensorFlow Mechanics
1. Build graph using TensorFlow operations
2. feed data and run graph (operation) **sess.run(op)**
3. update variables in the graph (and return values)

**Computaional Graph**
```python
import tensorflow as tf

# 1. Build graph using TensorFlow operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # 똑같이 tf.float32 이다.
node3 = tf.add(node1, node2)

print("node1: ", node1, "node2: ", node2)
print("node3: ", node3)

# 2. feed data and run graph (operation) **sess.run(op)**
# 3. update variables in the graph (and return values)

sess = tf.Session()
print("sess.run([node1, node2]): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))
```

**Placeholder** => 실행단계에서 값을 던저주고싶을때 사용
```python
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

sess = tf.Session()
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
```

### 모든것이 Tensor 라고 인지해야함!
tensor 는 기본적으로 array 를 뜻한다

Tensor 에서의 Rank == 차원, 즉 dimension

Rank | Math entity                       | python example
---- | --------------------------------- | --------------
0    | Scalar (magnitude only)           | s = 432
1    | Vector (magnitude and direction)  | v = [1, 4, 3]
2    | Matrix (table of numbers)         | m = [[1, 3], [4,9]]
3    | 3-Tensor (cube of numbers)        | ....
n    | n-Tensor (you get the idea)       | ....

Tensor 에서의 Shapes == 각각의 element에 몇개씩 들었는가?

None 으로 정의하면 n개 가 들어갈 수 있다는 말임.

Rank | Shape              | Dimension number | Example
---- | ------------------ | ---------------- | --------------------------------------
0    | []                 | 0-D              | A 0-D tensor. A scalar
1    | [D0]               | 1-D	             | A 1-D tensor with shape [5]
2    | [D0, D1]           | 2-D	             | A 2-D tensor with shape [3, 4]
3    | [D0, D1, D2]       | 3-D              | A 3-D tensor with shape [1, 4, 3]
n    | [D0, D1, ... Dn-1] | n-D	             | A tensor with shape [D0, D1, ... Dn-1]

Tensor 에서의 Types. 보통의 경우에 tf.float32, tf.int32 를 사용한다.

## Section 2 - 2 Tensorflow로 간단한 linear regression을 구현

1. build graph using TF operation
```python
import tensorflow as tf

# X, Y 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# weight 와 bias 정의. Variable의 노드로 설정. 
# 일반적인 Variable 이랑 다르다. tensorflow 내에서 사용되는 변수.
# 즉 tensorflow 가 학습하는 동안에 변경되는 variable 이라고 생각.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

# reduce_mean 은 평균을 내어주는 function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
```

2. Run/Update graph and get results
```python
sess = tf.Session()

# 그래프 안의 전역변수를 초기화 시켜줌
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
```

### Placeholders 를 사용해서 구현 해보기

```python
import tensorflow as tf

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X: [1, 2, 3, 4, 5], Y: [1.1, 2.1, 3.1, 4.1, 5.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
```

## Section 3 - 2 Linear Regression의 cost 최소화의 TensorFlow 구현

**[matplotlib 설치][matplotlib]**

pip3 즉 python3를 사용하는 경우 : pip3 install -U matplotlib

```python
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()
```
그래프를 눈으로 확인가능합니다 !!!!!!!!!!!

### Gradient descent
```python
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(update, feed_dict={X: x_data, Y: y_data}), sess.run(W))
```
> 왜 전에 예제와 같이 reduce_mean을 안 쓰고 reduce_sum을 쓴 이유

> 1/m 이 붙냐 차이기 때문에 기능적으로는 동일한 결과를 나타낼 것이다.

말이 안되는 W 값을 주고 GD 이용해서 minimized cost 찾기.
```python
import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.0)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
```

### Optional: compute_gradient and apply_gradient

TF 에서 기본적으로 제공하는 GD를 조금더 건드리고 싶을때, compute_gradient를 가지고 optimized 된 cost 를 return 받은 후에, 원하는대로 조작 후, apply_gradients로 적용시킨다.

## Section 4 - 2 multi-variable linear regression을 TensorFlow에서 구현하기

### Multi Variable hypothesis

```python
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

# placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
print(hypothesis)

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
```

### Matrix
```python
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
# tf.matual 은 matrix 곱 연산이다. 여기서는 X * W
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

```

## Section 4 - 3 TensorFlow로 파일에서 데이타 읽어오기

### loading data from file
```python
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# delimiter는 구분자 이다.
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
print("Your score will be ", sess.run(
    hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
```

**file이 너무 큰 경우**
[TensorFlow의 Queue Runners 를 이용해서 loading 하자!][queue_runner]

### loading file many and big file

```python
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# shuffle은 섞을껀지 말껀지 정함.
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in. batch는 펌프같은 역활로 데이터를 끌어오는 역활이다.
# xy[0:-1] == x data, xy[-1:] == y data
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
# 조정자(Coordinator) 클래스는 멀티 쓰레드들이 같이 정지되도록 한다.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

```

[matplotlib]: https://matplotlib.org/2.2.2/users/installing.html
[queue_runner]: https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/how_tos/threading_and_queues/