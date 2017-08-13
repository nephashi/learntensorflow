# 检测和日志
## 一.默认日志层级
tf有默认的日志系统，这些日志严重度从低到高分为五级：DEBUG, INFO, WARN, ERROR, FATAL。当你配置某一层级之后，该层级和更高层级的信息将输出。当你调用以下代码，则设置日志等级为INFO：
```
tf.logging.set_verbosity(tf.logging.INFO)
```
那么当你训练某一模型时，会得到以下输出，默认100步输出一次。
```
INFO:tensorflow:loss = 1.18812, step = 1
INFO:tensorflow:loss = 0.210323, step = 101
INFO:tensorflow:loss = 0.109025, step = 201
```
## 二.validation_monitor
validation_monitor可以在训练中途在测试集上做测试，以得到更多信息。在实例化分类器之前构造一个validation_monitor。由于monitor依赖模型存储的checkpoint做测试，因此需要config参数，设定多少秒存储一次。该值不宜过大，如果monitor两次测试之间checkpoint没有更新，那么monitor不会做测试（我根据代码表现推测）。
```
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50)
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=3,
    model_dir="/tmp/iris_model",
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
```
在训练时传入monitor。
```
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor])
```
会得到类似这样的结果。
```
INFO:tensorflow:Validation (step 50): loss = 1.71139, global_step = 0, accuracy = 0.266667
...
INFO:tensorflow:Validation (step 300): loss = 0.0714158, global_step = 268, accuracy = 0.966667
...
INFO:tensorflow:Validation (step 1750): loss = 0.0574449, global_step = 1729, accuracy = 0.966667
```
## 三.自定义评价指标
默认状况下，使用loss和accuracy，但我们也可以自定义指标。我没有看懂参数，但大致如下调用：
```
validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics)
```
## 四.提早停止训练
当模型拟合到一定程度，继续训练只是在浪费时间，还会增加过拟合风险。当某一指标在一定迭代次数没有增加/减少，我们可以停止训练。
```
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics,
    #评价指标
    early_stopping_metric="loss",
    #若期望指标减小则置为True，否则False
    early_stopping_metric_minimize=True,
    #200轮不变则停止训练
    early_stopping_rounds=200)
```
可以得到以下的结果
```
...
INFO:tensorflow:Validation (step 1150): recall = 1.0, loss = 0.056436, global_step = 1119, precision = 1.0, accuracy = 0.966667
INFO:tensorflow:Stopping. Best step: 800 with loss = 0.048313818872.
```