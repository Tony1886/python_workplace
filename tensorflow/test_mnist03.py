# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 18:39:51 2018
来自一本tensorflow的书 
有点问题 没找到原因
@author: Tan Zhijie
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# layers setting
LAYER1_NODE = 500                 # 节点数目
BATCH_SIZE = 100                  # batch大小
LEARNING_RATE_BASE = 0.8          # 基础学习率
LEARNING_RATE_DECAY = 0.99        # 学习率衰减系数
REGULARIZATION_RATE = 0.0001      # 正则化系数 
TRAINING_STEPS = 30000            # 训练次数
MOVING_AVERAGE_DECAY = 0.99       # 滑动平均衰减率

def inference(input_tensor , avg_class, weights1,biases1,
              weights2,biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用平滑平均衰减
        layer1 = tf.nn.relu(
                tf.matmul(input_tensor, avg_class.average(weights1)) + 
                avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
    
def train(mnist):
    x = tf.placeholder(tf.float32,[None, INPUT_NODE], name = 'x_input')
    y_ = tf.placeholder(tf.float32,[None, OUTPUT_NODE], name = 'y_input')
    
    # 生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE], stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE], stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))
    
    y = inference(x,None,weights1, biases1, weights2, biases2)
    # 定义训练轮数的不可训练的变量
    global_step = tf.Variable(0, trainable = False)
    
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    # 计算滑动平均的操作
    variable_averages_op = variable_average.apply(tf.trainable_variables())
    
    average_y = inference(x, variable_average, weights1, biases1, weights2, biases2)
    
    # loss function
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_,1))
    cross_entropy = tf.reduce_sum(-y_*tf.log(y),reduction_indices = [1])
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # L2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1)+regularizer(weights2)
    
    loss = cross_entropy_mean + regularization
    
    # 学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step, 
                                               mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    
    train_op = tf.group(train_step,variable_averages_op)
    
    correct_prediction = tf.equal(tf.argmax(average_y, 1),tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())
        validate_feed = {x:mnist.validation.images,
                         y:mnist.validation.labels}
        
        test_feed ={x:mnist.test.images,y:mnist.test.labels}
        
        for i in range(TRAINING_STEPS):
#            if i%100 == 0:
#                validate_acc = sess.run(accuracy, feed_dict = validate_feed)
#                print('After %d trainning step(s), validation accuacy using average model is %g' 
#                      %(i, validate_acc))
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict = {x:xs,y:ys})
            
        test_acc = sess.run(accuracy, feed_dict = test_feed)
        print('After %d trainning step(s), validation accuacy using average model is %g' 
                      %(i, test_acc))
def main(argv = None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)
    
if __name__ == '__main__':
    tf.app.run()
