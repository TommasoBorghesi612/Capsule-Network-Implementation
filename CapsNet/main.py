import numpy as np
import tensorflow as tf
import random

from CapsNet import CapsNetwork
from CapsNet import get_predicting_units

import matplotlib.pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')

tf.compat.v1.disable_eager_execution()

batch_size = 100
learning_rate = 0.0001
epochs = 50

X = tf.compat.v1.placeholder(tf.float32, shape=[None,784])
y_true = tf.compat.v1.placeholder(tf.float32,shape=[None,10])
X_image = tf.reshape(X,[-1,28,28,1])

conv1_filters = 256
conv1_kernel = 9
conv1_stride = 1
# input_shape = tf.shape(X_image)
convcaps_num = 32
caps1_filters = 8
caps1_kernel = 9
caps1_stride = 2
caps2_neurons = 16      
digitcaps_num = 10
predicting_units = int(get_predicting_units([28,28], conv1_kernel, conv1_stride, caps1_kernel, caps1_stride, convcaps_num))
# print(predicting_units)
routing_it = 3
fc1_units = 512
fc2_units = 1024
image_dims = 784
T_c = y_true


model = CapsNetwork(conv1_filters, conv1_kernel, conv1_stride,
                    convcaps_num, caps1_filters, caps1_kernel, caps1_stride, caps2_neurons, 
                    digitcaps_num, predicting_units, routing_it,
                    fc1_units, fc2_units, image_dims, T_c)


v_softmax, total_loss, reconstruction_err, reconstructed_image = model.compute(X_image, batch_size, T_c, X)
average_error, reconstructed_fake_image = model.compute(X_image, batch_size, T_c, X, input_kind="fake")
detail_error, decoder_loss, difference_error = model.compute(X_image, batch_size, T_c, X, true_input=False)

tvars = tf.compat.v1.trainable_variables()

encoder_vars = [var for var in tvars if 'Encoder' in var.name]
decoder_vars = [var for var in tvars if 'Decoder' in var.name]

print([v.name for v in encoder_vars])
print([v.name for v in decoder_vars])

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
train_global = optimizer.minimize(total_loss)
train_average = optimizer.minimize(average_error, var_list = decoder_vars)
train_details = optimizer.minimize(detail_error, var_list = encoder_vars)

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)

print(tf.compat.v1.trainable_variables())

with tf.compat.v1.Session() as sess:

    print('Session start')
    
    sess.run(init)
    # saver.restore(sess, "./CapsNet_fixed_decoder_loss")
    
    for k in range(1):
                    
        num_batches = mnist.train.num_examples // batch_size
        
        for i in range(num_batches): #num_batches

            batch_x = mnist.train.images[i*batch_size:(i+1)*batch_size]
            batch_y = mnist.train.labels[i*batch_size:(i+1)*batch_size]

            # batch_x , batch_y = mnist.train.next_batch(batch_size)        
            sess.run(train_global,feed_dict={X:batch_x,y_true:batch_y})     
            sess.run(train_average,feed_dict={X:batch_x,y_true:batch_y})     
            # sess.run(train_details,feed_dict={X:batch_x,y_true:batch_y})
            # affffff = sess.run(decoder_loss,feed_dict={X:batch_x,y_true:batch_y})
            # print(affffff)
            # baffffff = sess.run(difference_error,feed_dict={X:batch_x,y_true:batch_y})
            # print(baffffff)
            # print('-----------------')

            # shap = sess.run(asd,feed_dict={X:batch_x,y_true:batch_y})
            # print(shap)

            # if 10*(i/num_batches)%1 == 0:
            #     print('Current epoch % progress = {}'.format(i/num_batches))

        print('Currently on step {}'.format(k))
        print('Accuracy is:')
        # Test the Train Model
        matches = tf.equal(tf.argmax(tf.squeeze(v_softmax), axis=1), tf.argmax(y_true, axis=1))

        acc = tf.reduce_mean(tf.cast(matches,tf.float32))
        total_acc = 0
        total_recon_error = 0
        main_loss = 0

        for j in range (100):

            # batch_size = 100
            batch_acc = sess.run(acc,feed_dict={X:mnist.test.images[100*j:100*(j+1)],y_true:mnist.test.labels[100*j:100*(j+1)]})
            total_acc = total_acc + batch_acc
            # batch_recon = sess.run(reconstruction_err,feed_dict={X:mnist.test.images[100*j:100*(j+1)],y_true:mnist.test.labels[100*j:100*(j+1)]})
            # total_recon_error = total_recon_error + batch_recon
            # batch_loss = sess.run(reconstruction_err_fake,feed_dict={X:mnist.test.images[100*j:100*(j+1)],y_true:mnist.test.labels[100*j:100*(j+1)]})
            # main_loss = main_loss + batch_loss   
            # filtered = sess.run(filtered_out,feed_dict={X:mnist.test.images[10:20],y_true:mnist.test.labels[10:20]})

        glob_acc = total_acc/100
        print(glob_acc)
        # print('Reconstrucion Loss is:')
        # print(total_recon_error)
        # main_loss = main_loss/100
        # print('fake error is Loss is:')
        # print(main_loss)
        # # print(filtered)
        # print('\n')

        saver.save(sess, "./trywithv2")

with tf.compat.v1.Session() as sess:

    saver.restore(sess, "./trywithv2")
    images = sess.run(reconstructed_fake_image,feed_dict={X:mnist.test.images[0:100],y_true:mnist.test.labels[0:100]})
    orig = sess.run(X_image, feed_dict={X: mnist.test.images[0:100],
                                        y_true: mnist.test.labels[0:100]})
    count = 10
    f, a = plt.subplots(2, 10, figsize=(20, 4))
           
    j = 8
    for i in range(count):
        a[0][i].imshow(np.reshape(orig[i+j*10], (28, 28)), cmap ='gray')
        a[1][i].imshow(np.reshape(orig[i+j*10], (28, 28)), cmap ='gray')
    
    plt.show()
