# 使用TensorBoard可视化embedding

## 一.什么是embedding

大家都听说过word-embedding。什么是embedding呢，其实就是从高位空间到低维空间的映射，可以形象的理解为把高维空间的点嵌入了指定维的空间。

在这里embedding也引申为空间中的数据点。任何数据集中的样本都可以认为是多维空间中的一个点。比如iris数据集嵌入到三维空间之后就是这样的。

![](images/iris_embedding.png)

## 二.python API
tensorboard会从配置目录中读取需要到文件。可视化embedding可以分为几步

1.建立一个二维的张量作为被可视化对象，这里需要用name参数给张量起名。
```
sess = tf.InteractiveSession()
training_set = pd.read_csv("iris_training.csv", skipinitialspace=True,skiprows=1)
embedding = tf.Variable(training_set, trainable=False, name="embedding")
tf.global_variables_initializer().run()
```
2.用一个saver保存当前的模型状态。
```
saver = tf.train.Saver()
saver.save(sess, "/tmp/tensorflow/iris/log/model.ckpt")
```
3.构造一个ProjectorConfig对象，并把第一步的embedding加到里面。之后调用```projector.visualize_embeddings```保存这个config对象，这需要一个FileWrite作为参数。
```
writer = tf.summary.FileWriter("/tmp/tensorflow/iris/log", sess.graph)
config = projector.ProjectorConfig()
embad = config.embeddings.add()
embad.tensor_name = embedding.name
projector.visualize_embeddings(writer, config)
```
之后在控制台执行```tensorboard --logdir=$LOGDIR```，打开对应页面中的embedding选项卡，就可以看到结果。
## 三.数据点额外信息
我们可以使用tsv文件给数据点配置类标，图像等信息。只需要在上述代码上修改。见[原文](https://www.tensorflow.org/get_started/embedding_viz)和这个[示例代码](https://github.com/normanheckscher/mnist-tensorboard-embeddings/blob/master/mnist_t-sne.py)。