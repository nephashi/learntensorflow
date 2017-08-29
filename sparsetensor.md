# SparseTensor
tensorflow提供了关于稀疏张量的功能，但张量的一般操作并不支持稀疏张量，非常的恶心。
## 一.SparseTensor
稀疏张量以三个成员数组存储稀疏的张量。分别是非零元素的索引，非零元素的值，张量的shape。索引和值张量第零维长度相等，值互相对应。以矩阵为例：
```
SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
```
等于下面的矩阵
[[1, 0, 0, 0]
[0, 0, 2, 0]
 [0, 0, 0, 0]]
  
## 二.稀疏乘法
一般的语法并不支持稀疏张量。对于稀疏矩阵的乘法，有以下的几种方式：

- 如果一个稀疏矩阵和一个普通矩阵相乘，可以使用```tf.sparse_tensor_dense_matmul()```。这个方法设定第一个参数是稀疏矩阵，如果需要调换顺序，可以转置因子矩阵，再转置结果。这个方法的adjoint参数或许可以用作此法，但我不确定。
- 还可以使用```tf.nn.embedding_lookup_sparse```。这个方法十分恶心，我详细研究了它的用法，会在下面具体说。
- 对于不是特别稀疏的矩阵，可以转换成普通矩阵做乘法。
```
a = tf.SparseTensor(...)
b = tf.SparseTensor(...)

c = tf.matmul(tf.sparse_tensor_to_dense(a, 0.0),
              tf.sparse_tensor_to_dense(b, 0.0),
              a_is_sparse=True, b_is_sparse=True)
```
- 对于稀疏向量乘矩阵，可以使用```tf.nn.embedding_lookup```。具体情况看[这里](https://www.tensorflow.org/tutorials/word2vec#building_the_graph)

Note：[文档](https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/sparse_tensor_dense_matmul)中指出，上述两种方法接受不同的参数表示稀疏矩阵。比如你有一个[3, 5]的矩阵：
[[ 0 a 0 0 0 ]
 [ b 0 0 0  c ]
 [ 0 0 d 0 0 ]]
 那么```tf.sparse_tensor_dense_matmul()```需要一个SparseTensor作为参数sp_a (indices, values)，这下面的表示是SparseTensor的两个成员：索引和值：
\[0, 1]: a
\[1, 0]: b
\[1, 4]: c
\[2, 2]: d
```tf.nn.embedding_lookup_sparse```需要两个SparseTensor参数sp_ids和sp_weights。这里索引张量已经不是索引了。它的第二个值被放到了sp_ids的值中。而第二个值本身只是记录了当前的值在当前行中的count，已经没有用了。
[0, 0]: 1                [0, 0]: a
[1, 0]: 0                [1, 0]: b
[1, 1]: 4                [1, 1]: c
[2, 0]: 2                [2, 0]: d

## 三.使用```embedding_lookup_sparse```
这个方法本身意在处理embedding，不是用做乘法的。我会讨论两种情况，我认为这两种情况是不同的。
#### 密向量乘稀疏矩阵
```embedding_lookup_sparse```接受三个主要参数, params, sp_ids, sp_weights。在目前的case中，第一个是密向量，这个参数直接传入就好了，后两个是SparseTensor用来表示一个稀疏矩阵，主要需要讨论这两个值。这里我们假设param是一个行向量。
sp_ids和sp_weights，二者的indices和shape参数一致。indices中元素的第一个值，sp_ids的value值和sp_weights的value值一一对应（这三个数组长度相等），分别表示非零元素的列号，行号和值。
**可以认为，函数根据indices中每个元素的第一个值把sp_ids和sp_weights分组，不同的组对应结果中不同的列。在每组中，根据sp_ids的value（行号）从params中取出切片，乘以对应weight相加后作为对于列的值。这符合矩阵乘法的定义。**
注意得到的矩阵的列数会由indices中元素的第一个值的最大值决定，也就是列号的最大值。SparseTensor的shape成员没什么用。如果稀疏矩阵的最后一列完全为零，那么该函数将会返回错误的shape。
可以看一段代码:
```
import tensorflow as tf
import numpy as np

X = tf.placeholder("float", [10])
x = np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.float32)

sp_indices = tf.placeholder(tf.int64)
sp_shape = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)
sp_weights_val = tf.placeholder(tf.float32)
sp_ids = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
sp_weights = tf.SparseTensor(sp_indices, sp_weights_val, sp_shape)
y = tf.nn.embedding_lookup_sparse(X, sp_ids, sp_weights, combiner="sum")

sess = tf.Session()
sess.run(tf.initialize_all_variables())

y_values = sess.run(y, feed_dict={
X: x,
sp_indices: [[0, 0], [0, 1], [1, 0], [1, 1], [3, 0]], 
sp_shape: [5, 5], # 乱写的，并没有用
sp_ids_val: [2, 5, 3, 4, 7],
sp_weights_val: [1.0, 1.5, 3.5, 4.5, 2]
})

print (y_values)
```
这段代码会输出
```
[  9.5  28.5   0.   14. ]
```

#### 矩阵乘稀疏矩阵
如果上面的示例中的x变成两行，那么该矩阵的shape必须是[10, 2]。我怀疑是内部切片时总是从第一维切片。这样params就变成列向量了。得到的矩阵的shape也会相应变化，我们可以transpose回来。
```
import tensorflow as tf
import numpy as np

X = tf.placeholder("float", [10,2])
x = np.array([[0,1,2,3,4,5,6,7,8,9],[0,.1,.2,.3,.4,.5,.6,.7,.8,.9]], dtype=np.float32)
x = x.transpose()

sp_indices = tf.placeholder(tf.int64)
sp_shape = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)
sp_weights_val = tf.placeholder(tf.float32)
sp_ids = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
sp_weights = tf.SparseTensor(sp_indices, sp_weights_val, sp_shape)
y = tf.nn.embedding_lookup_sparse(X, sp_ids, sp_weights, combiner="sum")

sess = tf.Session()
sess.run(tf.initialize_all_variables())

y_values = sess.run(y, feed_dict={
X: x,
sp_indices: [[0, 0], [0, 1], [1, 0], [1, 1], [3, 0]], 
sp_shape: [5, 5],  #没用
sp_ids_val: [2, 5, 3, 4, 7],
sp_weights_val: [1.0, 1.5, 3.5, 4.5, 2]
})

print (y_values.transpose())
```
上面的代码会输出
```
[[  9.5         28.5          0.          14.        ]
 [  0.94999999   2.85000014   0.           1.39999998]]
```