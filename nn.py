import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入mnist数据,'MNIST_data':判断'MNIST_data'文件夹中是否有数据，没有则下载；one_hot:特征提取的一种编码形式
mnist = input_data.read_data_sets('./MNIST_data', one_hot = True)

#占位符
x = tf.placeholder(dtype=tf.float32, shape=(None, 28*28*1), name='x')#input_layer
y = tf.placeholder(dtype=tf.float32, shape=(None,10), name='y')#label

batch_size = 200 #分批次训练，每次的训练数据量

#建立神经网络中的层结构
def add_layer(input_data, input_num, output_num, activation_function = None):
    #output = input_data * weight + bias
    #tf.random_normal:用于从服从指定正太分布的数值中取出指定个数的值
    w = tf.Variable(initial_value = tf.random_normal(shape = [input_num, output_num]))#初始化系数w
    b = tf.Variable(initial_value = tf.random_normal(shape = [1, output_num]))#初始化bias
    output = tf.add(tf.matmul(input_data, w), b) #等同于 y = wx + b
    if activation_function:
        output = activation_function(output)
    return output

#建立神经网络全连接,两个隐藏层，一个输出层
def build_nn(data):
    hidden_layer1 = add_layer(data, 784, 100, activation_function=tf.nn.sigmoid)
    hidden_layer2 = add_layer(hidden_layer1, 100, 50, activation_function=tf.nn.sigmoid)
    output_layer = add_layer(hidden_layer2, 50, 10)
    return output_layer

#训练神经网络
def train_nn(data):
    #out of NN
    output = build_nn(data)
    #loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    liter_num = 500#训练次数

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(liter_num):
            #减轻内存负担，分批次进行训练
            epoch_cost = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                #
                x_data, y_data = mnist.train.next_batch(batch_size)
                cost, _ = sess.run([loss,optimizer], feed_dict={x: x_data, y: y_data})
                epoch_cost += cost
            print('Epoch', i, ':', epoch_cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(output, 1)),tf.float32))
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print(acc)
train_nn(x)
