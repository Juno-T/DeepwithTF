import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  ## for 3d plot (add_subplot,plot_surface)

#matplotlib inline

def plot_act(i=1.0, actfunc=lambda x: x):       ## setting with default arg
    #ws = np.arange(-0.5,0.5,0.05)              ## range by step size
    ws = np.linspace(-1,1,20)                   ## range by number of step
    #bs = np.arange(-0.5,0.5,0.05)
    bs = np.linspace(-1,1,20)
    X,Y = np.meshgrid(ws,bs)
    
    os = np.array([actfunc(tf.constant(w*i+b)).eval(session=sess) \
                  for w,b in zip(np.ravel(X), np.ravel(Y))])   ## np.ravel flatten to 1D (can set difference style)
    Z = os.reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')           ## creating 3d axis == ax=Axes3D(fig)
    ax.plot_surface(X ,Y ,Z ,rstride=1, cstride=1,cmap=cm.coolwarm)      ## plot surface(line,scatter,wire,etc)
                                                        ## rstride - Array row stride(step size)
                                                        ## X,Y,Z is 2D array
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
#start a session
sess = tf.Session();
#create a simple input of 3 real values
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
#create a matrix of weights
w = tf.random_normal(shape=[3, 3])
#create a vector of biases
b = tf.random_normal(shape=[1, 3])
#dummy activation function
def func(x): return x
#tf.matmul will multiply the input(i) tensor and the weight(w) tensor then sum the result with the bias(b) tensor.
act = func(tf.matmul(i, w) + b)
#Evaluate the tensor to a numpy array
print(act.eval(session=sess))   ## 3 inputs and 3 output layer(can be hidden layer)
plot_act(1.0, func)

## Sigmoid
plot_act(1, tf.sigmoid)
act = tf.sigmoid(tf.matmul(i, w) + b)
act.eval(session=sess)

## tanh
plot_act(1, tf.tanh)
act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess)

## Rectified Linear Unit
plot_act(1, tf.nn.relu)
act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess)