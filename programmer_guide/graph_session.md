# Graph and Session

tensorflow使用数据流图（dataflow graph）来表示计算之间的依赖。一般先建立一个计算图，之后执行这张图或者它的一部分。

## 为什么使用数据流图

数据流图是一种通用的编程模型，其中节点（tf.Operation）代表操作，边（tf.Tensor)代表数据。这种模型有几点好处。

- 并行化
- 分布式执行
- 汇编：tf在内部会重新汇编优化代码
- 移植：模型保存之后可以跨语言使用

## 什么是tf.Graph

tf.Graph包含两方面的信息：

- 图结构：图的节点和边指定了它们是如何组织的，但没有指定他们应该怎么被使用。
- 图集合：```tf.add_to_collection```可以将一系列图中的数据和一个字符串键关联（tf.GraphKeys定义了一些标准键）。```tf.get_collection```则允许你使用键找到这些数据。当你创建一个Variable时，它会被自动加入"global_variable"和"trainable_variable"集合，当你保存模型或者使用优化算法时，这两个集合会被使用。

tensorflow提供一个默认图。调用API将在这个图上进行操作。

## 命名操作

tf为所有操作提供默认命名，但自定义名字可以增强可读性。有两种方式自定义操作名：

- 大部分创建操作的API接受name参数。例如```tf.constant(42.0, name="answer")```创建一个名为"answer"的操作并返回一个名为"answer:0"的Tensor。如果图中已经有answer了，tensorflow会append"_1"，"_2"， etc。
- tf.name_scope允许为操作增加前缀，如果名字已经存在，那么tensorfow会append"_1"，"_2"， etc。
```
c_0 = tf.constant(0, name="c")  # => operation named "c"

# Already-used names will be "uniquified".
c_1 = tf.constant(2, name="c")  # => operation named "c_1"

# Name scopes add a prefix to all operations created in the same context.
with tf.name_scope("outer"):
  c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

  # Name scopes nest like paths in a hierarchical file system.
  with tf.name_scope("inner"):
    c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

  # Exiting a name scope context will return to the previous prefix.
  c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

  # Already-used name scopes will be "uniquified".
  with tf.name_scope("inner"):
    c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"
```
注意被返回的Tensor是在操作之后被命名的。如果返回Tensor的名字是```"<OP_NAME>:<i>"```，这意味着这个张量是由OP_NAME操作创建的，i是这个操作返回的Tensor的索引。

## 执行图

tensorflow使用session连接客户程序和C++运行时。有两种创建session的方式。注意由于session持有一些物理资源（GPU和网络），可以使用with来自动关闭session。或者也可以手动调用```session.close()```
```
# Create a default in-process session.
with tf.Session() as sess:
  # ...

# Create a remote session.
with tf.Session("grpc://example.org:2222"):
  # ...
```
tf.Session()接受三个可选参数，具体查文档。

### 使用tf.Session.run()

run方法需要传入一些fetches。fetches可以是tf.Operation，tf.Tensor或者是类Tensor类型（Tensor，Variable，numpy.array，list，python一般类型：bool，float，int，str）。fetches决定了哪一部分子图将被执行，这个子图会包含所有指定的fetches和它们依赖的操作。下面是一个例子。
```
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
  # Run the initializer on `w`.
  sess.run(init_op)

  # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
  # the result of the computation.
  print(sess.run(output))

  # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
  # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
  # op. Both `y_val` and `output_val` will be NumPy arrays.
  y_val, output_val = sess.run([y, output])
```
可以传入一个feed_dict参数，指定一些Tensor的值。注意这种方式不只可以用在placeholder上。
使用可选参数options可以追踪计算细节。
```
y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)
```