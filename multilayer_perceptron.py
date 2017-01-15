# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.', one_hot=True)

import tensorflow as tf

# Parameters
training_epochs = 16
batch_size = 100

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
features = tf.placeholder("float", [None, n_input])
labels = tf.placeholder("float", [None, n_classes])

# Layers, weight, and bias
layer_widths = {
    'h1': 256,
    'h2': 256,
    'out': n_classes}
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, layer_widths['h1']])),
    'h2': tf.Variable(tf.random_normal([layer_widths['h1'], layer_widths['h2']])),
    'out': tf.Variable(tf.random_normal([layer_widths['h2'], layer_widths['out']]))}
biases = {
    'b1': tf.Variable(tf.random_normal([layer_widths['h1']])),
    'b2': tf.Variable(tf.random_normal([layer_widths['h2']])),
    'out': tf.Variable(tf.random_normal([layer_widths['out']]))}

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(features, weights['h1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)
# Hidden layer with RELU activation
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)
# Output layer with linear activation
logits = tf.matmul(layer_2, weights['out']) + biases['out']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for _ in range(mnist.train._num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([optimizer, cost], feed_dict={features: batch_x, labels: batch_y})

        # Print Loss
        print('Epoch {:>2} - Loss: {}'.format(epoch, loss))

    # Print accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({features: mnist.test.images, labels: mnist.test.labels}))
