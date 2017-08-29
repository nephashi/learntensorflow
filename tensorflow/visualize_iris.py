import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd

sess = tf.InteractiveSession()

training_set = pd.read_csv("iris_training.csv", skipinitialspace=True,
                           skiprows=1)

embedding = tf.Variable(training_set, trainable=False, name="embedding")

tf.global_variables_initializer().run()

saver = tf.train.Saver()
saver.save(sess, "/tmp/tensorflow/iris/log/model.ckpt")

writer = tf.summary.FileWriter("/tmp/tensorflow/iris/log", sess.graph)


config = projector.ProjectorConfig()
embad = config.embeddings.add()
embad.tensor_name = embedding.name

projector.visualize_embeddings(writer, config)
