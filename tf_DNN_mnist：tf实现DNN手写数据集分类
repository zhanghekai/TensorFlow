import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib.layers import fully_connected


# 构建图阶段
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


# 构建神经网络层，我们这里两个隐藏层，基本一样，除了输入inputs到每个神经元的连接不同
# 和神经元个数不同
# 输出层也非常相似，只是激活函数从ReLU变成了Softmax而已
'''def neuron_layer(X, n_neurons, name, activation=None):
    # 包含所有计算节点对于这一层，name_scope可写可不写
    with tf.name_scope(name):
        # 取输入矩阵的维度作为层的输入连接个数
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)# np.sqrt()返回平方根
        
        # 这层里面的w可以看成是二维数组，每个神经元对于一组w参数
        # truncated normal distribution 比 regular normal distribution的值小
        # 不会出现任何大的权重值，确保慢慢的稳健的训练
        # 使用这种标准方差会让收敛快
        # w参数需要随机，不能为0，否则输出为0，最后调整都是一个幅度没意义
        #   tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
        #      这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，
        #      就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
        
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        w = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        # 向量表达的使用比一条一条加和要高效
        z = tf.matmul(X, w) + b
        if activation == "relu":
            return tf.nn.relu(z) # tf.nn.relu()函数是将大于0的数保持不变，小于0的数置为0
        else:
            return z


with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    # 进入到softmax之前的结果
    logits = neuron_layer(hidden2, n_outputs, "outputs")
'''

# fully_connected函数是tf已经定义好的，其底层代码为上述两个注释掉的函数
# 建立的是整个深度神经网络的拓扑图
with tf.name_scope("dnn"):
    # tensorflow使用这个函数帮助我们使用合适的初始化w和b的策略，默认使用ReLU激活函数
    #即fully_connected实现了三个操作：1.初始化 W 和 b；2.Z=Wx+b的线性求和；3.使用激活函数
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    # 输出层不适用ReLU函数，因为我么希望在输出层使用softmax函数
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

# 根据真实值计算损失函数
with tf.name_scope("loss"):
    # 定义交叉熵损失函数，并且求样本平均
    # 函数等价于先使用softmax损失函数，再接着计算交叉熵，并且更有效率
    # 类似的softmax_cross_entropy_with_logits只会给one-hot编码
    # sparse_softmax_cross_entropy_with_logits会给0-9分类号,先将logits进行softmax归一化，然后与label表示的onehot向量比较，计算交叉熵。 
    # (具有logits的稀疏softmax交叉熵)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

# 梯度下降最小化损失函数
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    # tf.nn.in_top_k()获取logits里面最大的那1位和y比较类别是否相同，返回True或者False一组值
    '''
    predictions: 你的预测结果（一般也就是你的网络输出值）大小是预测样本的数量乘以输出的维度
    target:      实际样本类别的标签，大小是样本数量的个数
    k:           每个样本中前K个最大的数里面（序号）是否包含对应target中的值  
    '''
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 计算图阶段
mnist = input_data.read_data_sets("MNIST_data_bak/")
n_epochs = 400 # 迭代次数
batch_size = 50 # 每一次训练所使用的数据集大小

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) # 循环执行的是梯度下降最小值函数
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        # 每迭代一次，分别用训练数据和测试数据计算准确率
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_dnn_model_final.ckpt")# 保存模型，即保存 W 和 b 的值
    
'''
# 使用模型预测
with tf.Session as sess:
    saver.restore(sess, "./my_dnn_model_final.ckpt")# 载入模型
    X_new_scaled = [...]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)  # 查看最大的类别是哪个
