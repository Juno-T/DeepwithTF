import numpy as np
from scipy import signal as sg
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

h = [2,1,0]     ## image input
x = [3,4,5]     ## kernel

y = np.convolve(x,h)    ## slide invertof kernel on [0 h 0] (full padding)
#print(y)

print("Compare with the following values from Python: y[0] = {0} ; y[1] = {1}; y[2] = {2}; y[3] = {3}; y[4] = {4}".format(y[0],y[1],y[2],y[3],y[4])) 

x = [6,2]
h = [1,2,5,4]

y = np.convolve(x,h,"full")  #now, because of the zero padding, the final dimension of the array is bigger
print(y)                     ## np.convolve arg = 1d array

y = np.convolve(x,h,"same") #it is same as zero padding, but withgenerates same
                            ## zero padding only on left side
print(y)

y = np.convolve(x,h,"valid") # we will understand why we used the argument valid in the next example 
                             # no zero padding
print(y)

I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]

g= [[-1,1]]                 ## matrix cast

print ('Without zero padding \n')
print ('{0} \n'.format(sg.convolve( I, g, 'valid')))    ## any dimension
# The 'valid' argument states that the output consists only of those elements 
# that do not rely on the zero-padding.

print ('With zero padding \n')
print (sg.convolve( I, g))

g= [[-1,  1],
    [ 2,  3],]
    
## invert of g= [[ 3,  1],
##               [ 2, -1],]

print ('With zero padding \n')
print ('{0} \n'.format(sg.convolve( I, g, 'full')))
# The output is the full discrete linear convolution of the inputs. 
# It will use zero to complete the input matrix

print ('With zero padding_same_ \n')
print ('{0} \n'.format(sg.convolve( I, g, 'same')))
# The output is the full discrete linear convolution of the inputs. 
# It will use zero to complete the input matrix


print ('Without zero padding \n')
print (sg.convolve( I, g, 'valid'))
# The 'valid' argument states that the output consists only of those elements 
#that do not rely on the zero-padding.

#############################################
##  Using tensorflow

# Building graph

input = tf.Variable(tf.random_normal([1,10,10,1]))      ## [batch size, width, height, number of channels] format to be passed in tf.nn.conv2d
filter = tf.Variable(tf.random_normal([3,3,1,1]))       ## [width, height, channels, number of filters] format to be passed in tf.nn.conv2d 
op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

#initialization and session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n{0} \n".format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result)

##############################################
## Image

im = Image.open('bird.jpg')

#uses the ITU-R 601-2 Luma transform (ther are several ways to convert an image to grey scale)

image_gr = im.convert("L")
print("\n Original type: %r \n\n"%image_gr)

# convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)          ## convert input to array
print("After conversion to numerical representation : \n\n%r"%arr)
#### Activating matplotlib for Ipython
#%matplotlib inline

### Plot image

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray') # (greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

##############################################
## Edge Detector on image

kernel = np.array([
                        [ 0, 1, 0],
                        [ 1,-4, 1],
                        [ 0, 1, 0],
                                     ]) 

grad = sg.convolve2d(arr, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
plt.show()

###########################################
print(type(grad))

grad_biases = np.absolute(grad) + 150

grad_biases[grad_biases > 255] = 255    ## == assigning != print(grad[grad>2]) < become array

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')
plt.show()