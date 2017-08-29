#变量域（Variable Scope）
关于变量的基本操作，可以看[这里](https://www.tensorflow.org/programmers_guide/variables)

## 提出问题

假设我们定义了一个函数，用于构造一个卷积模型。
```
def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_biases)

    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    conv2 = tf.nn.conv2d(relu1, conv2_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2 + conv2_biases)
```
我们希望复用这个模型，使得多张图片被同样的模型处理，这时我们会遇到问题，每次调用都会创建新的变量。我们可以在函数外创建变量并作为参数传入，但这样会破坏封装性。此时可以使用VairableScope。
要使用变量域，关键要用两个方法。

- ```tf.get_variable(<name>, <shape>, <initializer>)```创建一个指定名字的变量
- ```tf.variable_scope(<scope_name>)```管理变量域

```tf.get_variable(<name>, <shape>, <initializer>)```需要传入初始化器来指定变量的初始值。这里有几个常用的选择。可以顾名知其意。

- tf.constant_initializer(value)
- tf.random_uniform_initializer(a, b)
- tf.random_normal_initializer(mean, stddev)

尝试解决上面的问题
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
    
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
        
result1 = my_image_filter(image1)
result2 = my_image_filter(image2)
# Raises ValueError(... conv1/weights already exists ...)
```
我们期望可以复用conv1和conv2域中的变量，但是代码抛出了异常：con1/weights已经存在了。这和tf.get_variable的作用方式有关。下面的代码可以完美解决问题。
```
with tf.variable_scope("image_filters") as scope:
    result1 = my_image_filter(image1)
    scope.reuse_variables()
    result2 = my_image_filter(image2)
```

## 变量域作用机理
每个变量域都会绑定一个布尔型的reuse标记。我们调用tf.get_variable时，有两种情况。

#### 1.tf.get_variable_scope().reuse == False
这是默认情况，此时get_variable会创建新的变量，加入指定的变量名在域中已经存在，就抛出异常。例如：
```
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
assert v.name == "foo/v:0"
```
#### 2.tf.get_variable_scope().reuse == True
这种情况下，get_variable会取出已经存在的变量。若变量不存在，就会抛出异常。例如：
```
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 is v
```

可以调用scope.reuse_variable()把reuse置为True，但不能把reuse置为False。因为一旦置为True，人们会期待其中的变量能够复用。尽管如此，你可以进入一个reuse为True的新变量域，之后再退出，这样就可以任意的设置reuse标志了。注意scope嵌套时，reuse是会继承的。
```
with tf.variable_scope("root"):
    # At start, the scope is not reusing.
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo"):
        # Opened a sub-scope, still not reusing.
        assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo", reuse=True):
        # Explicitly opened a reusing scope.
        assert tf.get_variable_scope().reuse == True
        with tf.variable_scope("bar"):
            # Now sub-scope inherits the reuse flag.
            assert tf.get_variable_scope().reuse == True
    # Exited the reusing scope, back to a non-reusing one.
    assert tf.get_variable_scope().reuse == False
```
## 捕获变量域
我们也可以把变量域存成对象，这样用起来更方便。
```
with tf.variable_scope("foo") as foo_scope:
    v = tf.get_variable("v", [1])
with tf.variable_scope(foo_scope):
    w = tf.get_variable("w", [1])
with tf.variable_scope(foo_scope, reuse=True):
    v1 = tf.get_variable("v", [1])
    w1 = tf.get_variable("w", [1])
assert v1 is v
assert w1 is w
```
当我们在嵌套中使用变量域对象，则会跳出嵌套。
```
with tf.variable_scope("foo") as foo_scope:
    assert foo_scope.name == "foo"
with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo"  # Not changed.
```