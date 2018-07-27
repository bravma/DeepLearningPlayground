import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 784 = 28x28 images
# None: We don't know how many items will be in this dimension
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

sess = tf.InteractiveSession()

# Weights and bias
# variables: we need to change this values when the model learns
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# loss optimization
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5) \
    .minimize(cross_entropy)

# Init all variables
init = tf.global_variables_initializer()

sess.run(init)

# What is correct
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# How accurate is it
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Trian the model
num_steps = 3000
batch_size = 50
display_every = 100

for i in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys
        })
        print("Step {0}, training accuracy {1:.3f}%"
              .format(i, train_accuracy * 100))

# accuracy on test data
validation_accuracy = accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels
})

print("Validation Accuracy: {0:.3f}%".format(validation_accuracy))

sess.close()
