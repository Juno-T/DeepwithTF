import tensorflow as tf

## Helloworld
a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a,b)
session = tf.Session()
result = session.run(c)
print(result)
session.close()

with tf.Session() as ss :
    result = ss.run(c)
    print(result)
    
##  Multi dimensional array
Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
with tf.Session() as session:
    result = session.run(Scalar)
    print ("Scalar (1 entry):\n %s \n" % result)
    result = session.run(Vector)
    print ("Vector (3 entries) :\n %s \n" % result)
    result = session.run(Matrix)
    print ("Matrix (3x3 entries):\n %s \n" % result)
    result = session.run(Tensor)
    print ("Tensor (3x3x3 entries) :\n %s \n" % result)
    
    
##  Operation 
#       add,subtraction
Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

first_operation = tf.add(Matrix_one, Matrix_two)
second_operation = Matrix_one + Matrix_two

#first_operation = tf.subtract(Matrix_one, Matrix_two)
#second_operation = Matrix_one - Matrix_two

with tf.Session() as session:
    result = session.run(first_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = session.run(second_operation)
    print ("Defined using normal expressions :")
    print(result)
    
#       multiplication
Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

#first_operation = tf.matmul(Matrix_one, Matrix_two) ## Mathematically
first_operation = Matrix_one * Matrix_two  ## element-wise


with tf.Session() as session:
    result = session.run(first_operation)
    print ("Defined using tensorflow function :")
    print(result)
    
##  Variable
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.global_variables_initializer()
with tf.Session() as session:
  session.run(init_op)              # Very important to initiate Var
  print(session.run(state))
  for i in range(3):
    session.run(update)
    print(session.run(state))
    
##  Placeholder
a = tf.placeholder(tf.float32)
b = a*[[2],[1],[1],[2]]
with tf.Session() as sess:
    result = sess.run(b,feed_dict={a:[[1],[1],[1],[1]]})
    print (result)

dictionary={a: [ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }
with tf.Session() as sess:
    result = sess.run(b,feed_dict=dictionary)
    print (result)
    
