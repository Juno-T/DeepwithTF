import numpy as np
import tensorflow as tf

sess = tf.Session()

LSTM_CELL_SIZE = 4  # output size(dimension), which is same as hidden size in the cell

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([3,LSTM_CELL_SIZE]),)*2  ## (always) 2 element, h=prv_output and c=prv_state
                                ## (array(),)*3 = (array(),array(),array())
print(sess.run(state))

sample_input = tf.constant([[1,2,3,4,3,2,7],[3,2,2,2,2,2,1],[7,6,5,4,3,2,1]],dtype=tf.float32)
print (sess.run(sample_input))  ## input of LSTM cell ( basically compose of 
                                ## input data and RNN Output (batch = 2)?)

with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
print (sess.run(state_new))
print (sess.run(output))