import tensorflow as tf
import  matplotlib.pyplot as plt
import numpy as np
h=np.linspace(-10,10,50)
out=tf.nn.softmax(h)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y= sess.run(out)
plt.xlabel('Activity of Neuron')
plt.ylabel('Output of Neuron')
plt.title('Sigmoid Activation Function')
plt.plot(h,y)
plt.show()
