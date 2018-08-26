import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#matplotlib inline
plt.rcParams['figure.figsize'] = (10,6)
X = np.arange(0.0, 5.0, 0.1)
print(X)
a=-2
b=5
Y= a*X + b 

#plt.figure(1)
plt.subplot(211)
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
#plt.show()

##  Linear Regression
#       create and visualize training set
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 2
#making lamda(function), parameter => last (y_data)
y_data = np.vectorize(lambda y,sc: y + np.random.normal(loc=0.0, scale=sc))(y_data,0.1)
#plt.figure(1)
plt.subplot(212)
plt.plot(x_data,y_data,'ro')
plt.show()

#       Initialize params and cost function
a = tf.Variable(1.0)
b = tf.Variable(0.2)            ## Init randomly
y = a*x_data + b                ## first prediction

loss = tf.reduce_mean(tf.square(y-y_data))      ## define cost function

alpha = 0.5                                             ## Learning rate
optimizer = tf.train.GradientDescentOptimizer(alpha)    
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

##  Training
train_data = []
costbyit = []
for step in range(100):
    evals = sess.run([train])[0:]       ## {array}[a:b] -> return array[a] through array[b-1]
                                            ## space mean very first or very last
    dmp = sess.run([a,b])
    costbyit.append(sess.run(loss))
    if step % 2 == 0:
        print(step,dmp)
        train_data.append(dmp)

## visualize Learning

plt.figure(1)
converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)         ## changing line color for each iteration
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0               ## limit the color value ([0.0,1.0])
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(x_data)   ## prediction line of each iteration params
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr,cg,cb))    ## setp = set property

plt.plot(x_data, y_data, 'ro')

# create legend
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line,green_line])

##  Learning curve
plt.figure(2)
plt.plot(np.arange(1,len(costbyit)+1,1),costbyit)       ##plot cost to iteration
plt.xlabel('No. of iterations')
plt.ylabel('cost')
plt.show()
