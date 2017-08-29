# Variables

Variable适合保存共享的，持久的状态。
Variable的存在方式不依赖于运行时，特定的操作可以读取或改变它的值。这些改变对于多个Session都是可见的。

## 创建Variable

创建一个Variable最佳的方式是调用tf.get_variable。这个函数需要传入一个名字参数和一个shape参数。名字参数在重新获取该变量，或保存，加载模型时都有用。使得复用模型变得简单。
tf.get_variable还接受数据类型参数和初始化参数，默认分别是```tf.float32```和```tf.glorot_uniform_initializer```。下面有多种调用的方式。
```
my_variable = tf.get_variable("my_variable", [1, 2, 3])

my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)

other_variable = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant([23, 42]))
```

## 变量集合

在某些断开的代码段中，我们想用同一个变量。这时可以使用collections。
默认情况下所有变量都放在两个collection中：```tf.GraphKeys.GLOBAL_VARIABLES```——跨设备访问的变量，以及```tf.GraphKeys.TRAINABLE_VARIABLES```——tensorflow会为这里的变量计算梯度。
你也可以自定义collection。下面的代码存取已经定义的my_local变量。
```
tf.add_to_collection("my_collection_name", my_local)

tf.get_collection("my_collection_name")
```
## 设备设置
tensorflow允许把特定的变量放在特定的设备上允许，例如
```
with tf.device("/gpu:1"):
    v = tf.get_variable("v", [1])
```
在分布式环境下，正确的分发变量是很重要的。加入弄错了变量服务器（parameter server）和工作者（worker），后果很严重。因此tf提供工具自动分发变量。
```
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  v = tf.get_variable("v", shape=[20, 20])  # this variable is placed in the parameter server by the replica_device_setter
```

## 初始化变量

任何变量在使用前都需要初始化。高级API会自动设置，使用低级API时则需要手动配置。在session中run```tf.global_variables_initializer()```一次性初始化所有变量。也可以一次初始化一个变量。
```
session.run(myvariable.initializer)
```
注意```tf.global_variable_initializer()```方法不考虑顺序，因此如果某一变量的初始值依赖另一变量，可能会引发错误。因此在引用变量初始值时使用```variable.initialized_value()```会更保险。
```
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
```

## 使用变量
使用variable和使用tensor是一样的。如果想改变变量的值，可以调用assign等一系列操作。同时，Optimizer提供方法自动优化变量。

## 共享变量

共享变量有两种方式

- 直接传递Variable对象
- 把Variable对象包裹在tf.variable_scope中

variable scope允许你在调用内部创建变量的函数时控制变量复用，同时允许你创建层次性的变量名体系。例如下面的代码建立一个卷积层。
```
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
```
但如果你想调用这个函数建立多个卷积层，则不会work。
```
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 1, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
```
因为代码的含义不清楚。是要建立新的卷积层，还是要复用已有的变量？把代码包裹在scope中则可以解决这个问题。
```
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 1, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```
如果你想让函数间复用变量，可以把scope的reuse置为True，这又两种方式。
```
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)
```
```
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)
```
像上面一样使用同样的字符串引用scope是很危险的。通过把已有的scope传入构造器，可以引用同样的scope。
```
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)
```
[这里](./sharing_variable.md)对变量共享有进一步的说明。