import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#tf.logging.set_verbosity(tf.logging.WARN)

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) 
                            ## onwe_hot=True > alternative way to represent number (binary-> digit)
                            ##                                                  5 = 101 -> 1000000

print("\n\n\n\n\n***********************************************************\n\n")    
print(tf.__version__)                     
print(mnist.train.images)

#######################################
## First method : Interactive Session
sess = tf.InteractiveSession()

x  = tf.placeholder(tf.float32, shape=[None, 784])  ## any number of 784(28x28) pixel image
y_ = tf.placeholder(tf.float32, shape=[None, 10])   ## any number of sized 10 outcome label
# Weight tensor
W = tf.Variable(tf.zeros([784,10],tf.float32))      ## (10 classifier weight)
# Bias tensor
b = tf.Variable(tf.zeros([10],tf.float32))

sess.run(tf.global_variables_initializer())
# activations
y = tf.nn.softmax(tf.matmul(x,W)+b)                 ## softmax generate probabilities for the output

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) ## red_ind=[1] -> row sum /[0] -> col sum
                                    ## unseen cost function. focus only on reducing y=1 index
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Load 50 training examples for each training iteration   
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})   # minibatch gradient descent

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

sess.close()

######################################
## Second method : Building graph and run session

#(Input) -> [batch_size, 28, 28, 1] >> Apply 32 filter of [5x5]
#(Convolutional layer 1) -> [batch_size, 28, 28, 32]
#(ReLU 1) -> [?, 28, 28, 32]
#(Max pooling 1) -> [?, 14, 14, 32]
#(Convolutional layer 2) -> [?, 14, 14, 64]
#(ReLU 2) -> [?, 14, 14, 64]
#(Max pooling 2) -> [?, 7, 7, 64]
#[fully connected layer 3] -> [1x1024]
#[ReLU 3] -> [1x1024]
#[Drop out] -> [1x1024]
#[fully connected layer 4] -> [1x10]

sess.close()  ## finish possible remaining session

sess = tf.InteractiveSession()

# Load data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# Initial Parameters
width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

# Input and output
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

# Converting images of dataset to tensor
x_image = tf.reshape(x, [-1,28,28,1])  
print(x_image)
####################################################################
## Building Deep Neural Network
###########     Convolutional Layer1    28x28x1 -> 14x14x32
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))   #[H,W,InChan,OutChan]
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
print(conv1)        ## find max in each 2x2 pixels (compressing)

###########     Convolutional Layer2    14x14x32 -> 7x7x64
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2
h_conv2 = tf.nn.relu(convolve2)
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
print(conv2)

###########     Fully connected Layer3  7x7x64 -> 1024
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs
fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fcl)
print(h_fc1)
# Drop out (Optional phase for reducing Overfitting)
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
print(layer_drop)

###########     Readout Layer (Softmax Layer)   1024 -> 10
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]
fc=tf.matmul(layer_drop, W_fc2) + b_fc2
y_CNN= tf.nn.softmax(fc)
print(y_CNN)

#########################################################
## Define cost function & Training setup
# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))
# optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# prediction
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))
# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Start training
# init
sess.run(tf.global_variables_initializer())

for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Evaluate the model
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# Visualization
#kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32,-1]))
