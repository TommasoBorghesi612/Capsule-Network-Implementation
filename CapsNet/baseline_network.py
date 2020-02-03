import numpy as np
import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


X = tf.placeholder(tf.float32,shape=[None,784])
labels = tf.placeholder(tf.float32,shape=[None,10])
X_image = tf.reshape(X,[-1,28,28,1])

epochs = 50
batch_size = 100
learning_rate = 0.0001

conv1_filters = 512
conv1_kernel = 9
conv1_stride = 1
pool1_kernel = 2
pool1_stride = 2
conv2_filters = 256
conv2_kernel = 5
conv2_stride = 1
pool2_kernel = 2
pool2_stride = 2
fc1_units = 1024
nlabels = 10



conv1 = tf.contrib.layers.conv2d(X_image, conv1_filters, conv1_kernel, 
                                            conv1_stride, padding="VALID", activation_fn=tf.nn.relu)

pool1 = tf.contrib.layers.max_pool2d(conv1, pool1_kernel, stride=pool1_stride)

conv2 = tf.contrib.layers.conv2d(pool1, conv2_filters, conv2_kernel, 
                                            conv2_stride, padding="VALID", activation_fn=tf.nn.relu)

pool2 = tf.contrib.layers.max_pool2d(conv2, pool2_kernel, stride=pool2_stride)

pool2_flat = tf.reshape(pool2, [-1, 2304])

fc1 = tf.contrib.layers.fully_connected(pool2_flat, num_outputs=fc1_units, activation_fn=tf.nn.relu)

dropout =  tf.contrib.layers.dropout(fc1, keep_prob=0.5)

logits = tf.contrib.layers.fully_connected(dropout, num_outputs=nlabels, activation_fn=tf.nn.sigmoid)

predictions = tf.contrib.layers.softmax(logits)


loss = tf.contrib.losses.sigmoid_cross_entropy(predictions, labels)


optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


print(tf.trainable_variables())

with tf.Session() as sess:

    print('Session start')
    
    sess.run(init)
    # saver.restore(sess, "./save_3ri_nobias")

    
    for k in range(epochs):
                    
        num_batches = mnist.train.num_examples // batch_size
        
        for i in range(num_batches): #num_batches

            batch_x = mnist.train.images[i*batch_size:(i+1)*batch_size]
            batch_y = mnist.train.labels[i*batch_size:(i+1)*batch_size]

            sess.run(train,feed_dict={X:batch_x,labels:batch_y})

        print('Currently on step {}'.format(k))
        print('Accuracy is:')
        # Test the Train Model
        matches = tf.equal(tf.argmax(predictions,axis=1),tf.argmax(labels,axis=1))

        acc = tf.reduce_mean(tf.cast(matches,tf.float32))
        total_acc = 0
        total_recon_error = 0
        main_loss = 0

        for j in range (100):

            # batch_size = 100
            batch_acc = sess.run(acc,feed_dict={X:mnist.test.images[100*j:100*(j+1)],labels:mnist.test.labels[100*j:100*(j+1)]})
            total_acc = total_acc + batch_acc

        glob_acc = total_acc/100
        print(glob_acc)

        # asd = sess.run(logits,feed_dict={X:mnist.train.images[0:100],labels:mnist.train.labels[0:100]})
        # print(asd.eval)

        saver.save(sess, "./baseline_wpool")