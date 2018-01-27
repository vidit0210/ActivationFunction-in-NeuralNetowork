import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

h=np.linspace(-10,10,50)
out=tf.nn.relu(h)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y=sess.run(out)
    plt.xlabel('Activity of Neuron')
    plt.ylabel('Output of Neuron')
    plt.title('Relu Activation Function')
    plt.plot(h,y)
    plt.show()
