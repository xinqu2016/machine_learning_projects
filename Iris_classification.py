"""
This program classifies three different types of Iris based on their four features, using a neural network model.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf

# download the training data
import os
training_data_url = 'http://download.tensorflow.org/data/iris_training.csv'
training_data_ftp = tf.keras.utils.get_file(os.path.basename(training_data_url),origin = training_data_url)
!head -n10 {training_data_ftp}

# read in training data
import csv
import numpy as np
with open(training_data_ftp,'r') as f:
    reader = csv.reader(f)
    features = np.ones([120, 4],dtype=np.float32)
    labels    = np.ones([120],dtype=np.int32)
    for i, line in enumerate(reader):
        if i>0:
          features[i-1,:]  = np.array(line[0:4])
          labels[i-1]    = np.array(line[-1])

# read in test data
testing_data_url = 'http://download.tensorflow.org/data/iris_test.csv'
testing_data_ftp = tf.keras.utils.get_file(os.path.basename(training_data_url), origin=testing_data_url)
!head - n10
{testing_data_ftp}
with open(testing_data_ftp, 'r') as f:
    reader = csv.reader(f)
    test_features = np.ones([30, 4], dtype=np.float32)
    test_labels = np.ones([30], dtype=np.int32)
    for i, line in enumerate(reader):
        if i > 0 and i < 31:
            test_features[i - 1, :] = np.array(line[0:4])
            test_labels[i - 1] = np.array(line[-1])

# build a neutral network model
model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)), \
                             tf.keras.layers.Dense(10, activation='relu'), \
                             tf.keras.layers.Dense(10, activation='relu'), \
                             tf.keras.layers.Dense(3)])
x = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='x')
y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
y_ = model(x)
loss = tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_)
opitimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

# train and assess the model
num_epochs = 500
training_accuracy = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        loss_val, _, y_pred  = sess.run([loss, opitimizer, y_], feed_dict={x: features, y: labels})
        if (epoch % 10 == 0):
            epoch_accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.convert_to_tensor(y_pred),axis=1, output_type=tf.int32), \
                                                         tf.convert_to_tensor(labels))
            print(sess.run(epoch_accuracy))

    loss_val, _, y_pred  = sess.run([loss, opitimizer, y_], feed_dict={x: test_features, y: test_labels})
    epoch_accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.convert_to_tensor(y_pred), axis=1, output_type=tf.int32), \
                                                 tf.convert_to_tensor(test_labels))
    print(sess.run(epoch_accuracy))