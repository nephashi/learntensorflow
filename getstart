#Getting Start with Tensorflow
tf提供多层的API，低层API Core提供更完全的系统控制，但是高层的API允许你更容易的构建模型，例如tf.contrib.learn。
##Tensor（张量）
张量是tf中核心的数据单元，简单来说就是多位数组，张量的阶就是数组的维度。例如：
```
3	#零阶张量,也就是一个标量.shape[]
[1., 2., 3.]	#一个1阶张量.shape[3]
[[1., 2., 3.], [4., 5., 6.]]	#一个2阶张量.shape[2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]]	#一个3阶张量.shape[2, 1, 3]
```

##构建简单的线性回归模型
###使用TensorFlow-Core建模
TF-Core是低层API，允许我们更完整的控制程序功能。可以认为这种程序包括两部分
1. 构建一个计算图
2. 执行一个计算图
一个计算图（computational graph）是被封装到一个图中的一系列TensorFlow操作的集合。图中的顶点（node）将0个或更多张量作为输入，并输出一个张量。
####常量
我们可以定义一些常量顶点：输入0个张量，输出一个值。
```
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```
这段程序会输出
```
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```
程序只是输出了一些状态信息，没有输出常量的具体值。只有当放在session里执行，才会真正输出值。一个session封装了运行时的控制和状态。
```
sess = tf.Session()
print(sess.run([node1, node2]))
```
将会输出
```
[3.0, 4.0]
```
可以在这两个常量上定义一些操作，例如加法：
```
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
```
结果是
```
node3:  Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3):  7.0
```
TensorFlow还提供了一种有效的可视化工具：TensorBoard，它可以得到下面的结果。
![tensorboard-add](images/tensorboard_add.png)
####占位符（placeholder）
上面的常量不太有用，我们还需要一些东西来接受可变的输入：placeholder就是这样的东西。正如其名，placeholder可以占领一个位置，等待之后的输入。可以在placeholder上定义操作，之后使用session的run方法的feed_dict参数将数据喂给占位符。
```
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # shortcut for tf.add(a, b)
add_and_triple = adder_node * 3.

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
print(sess.run(add_and_triple, {a: 3, b:4.5}))
```
结果是
```
7.5
[ 3.  7.]
22.5
```
####变量
机器学习任务中，我们需要改变参数达到训练目的。这时需要引入变量，声明变量时需要声明类型和一个初始值。
```
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```
声明后，变量不会被初始化，直到在session中显示的执行初始化。在下面的代码中，init只是个句柄，直到run(init)之后变量才被初始化。
```
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))
```
上面的代码初始化变量，之后计算linear_model，并且得到了结果。
```
[ 0.          0.30000001  0.60000002  0.90000004]
```
####优化模型
接下来我们会使用损失函数优化模型。给定一组x和y，我们采用均方误差度量模型。
```
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```
损失值是
```
23.66
```
接下来我们需要使用基于梯度的优化方法优化模型，简单的方法是使用提督下降法，TensorFlow封装了很多优化器。
```
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([W, b]))
```
结果是最优的模型参数
```
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```
完整的线性回归代码是：
```
import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```
这张图也可以被tensorBoard可视化
![](images/tensorboard_final.png)
###使用tf.contrib.learn构造模型
这是一种高级API，省去了很多细节
```
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
                                              batch_size=4,
                                              num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
```
结果是
```
train loss: {'global_step': 1000, 'loss': 4.3049088e-08}
eval loss: {'global_step': 1000, 'loss': 0.0025487561}
```
同时还可以基于已有的框架封装自己的算法，我们使用tf.contrib.learn封装自己的线性回归模型，注意下面代码的结构很类似我们使用core API写的线性回归。
```
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did. 
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
```