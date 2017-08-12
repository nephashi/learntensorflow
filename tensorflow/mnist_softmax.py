import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
print("downloading")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("finished")

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#[None, 10]
logit = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(1000):
    if(i % 100 == 0):
        print("iteration:" + str(i))
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x : batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(logit,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))