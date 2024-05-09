#通过python中的类（class）创建三层ANN模型，并实现其训练和测试
#neural network class definition
import numpy as np
import math
import scipy
import matplotlib.pyplot
import time
import pandas as pd

#ANN的python模型从结构上主要分为三大块：网络初始化模块、网络训练模块和网络测试（询问或预测）模块
class neuralnetwork:

    #网络初始化模块
    #initialise the neural network
    #初始化网格时需要设置输入层、隐含层、输出层神经元个数以及学习率等参数
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.i_nodes = inputnodes #输入层神经元个数input_layer_nodes通过初始化参数inputnodes赋值
        self.h_nodes = hiddennodes #隐含层神经元个数hidden_layer_nodes通过初始化参数hiddennodes赋值
        self.o_nodes = outputnodes #输出层神经元个数output_layer_nodes通过初始化参数outputnodes赋值
        self.l_r = learningrate #学习率learning_rate通过初始化参数learningrate赋值

        #initialise the weights
        #初始化权重, w_i_h定义为为输入层数隐含层之间的权重矩阵，w_h_o定义为隐含层和输出层之间的权重矩阵
        #numpy.random.normal(loc=0.0, scale=1.0, size=None)函数用于产生元素按正态分布的数组，第一个参数用于设置正态分布的均值，
        #第二个参数用于设置正态分布的标准差，第三个参数用于设置数组的维度和大小（形状）。
        #math.pow(x,y)函数返回x的y次方
        #权重矩阵的大小（形状）：其行数为后一层神经元的个数，其列数为前一层神经元的个数
        self.w_i_h = np.random.normal(0.0, math.pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_h_o = np.random.normal(0.0, math.pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        #define activation function
        #对于wavelet神经网络，其激活函数为某种小波基函数
        #lambda是定义函数的命令，定义的函数接收x并返回scipy.special.expit(x)，也就是sigmoid（x）的值
        #lambda定义的函数是匿名的，但可以通过幅值分配一个名称，如此处的activation_func
        #self.activation_func = lambda x: np.cos(1.75*x)*np.exp(-0.5*x**2)
        #self.activation_func = lambda x: scipy.special.expit(x)

        pass

    #定义sigmoid激活函数
    def func_sigmoid(self, x):
        y = scipy.special.expit(x)
        return y

    #sigmoid激活函数的导数
    def de_func_sigmoid(self, x):
        y = scipy.special.expit(x) * (1-scipy.special.expit(x))
        return y

    #定义morlet小波激活函数
    def func_morlet(self, x):
        y = np.cos(1.75*x)*np.exp(-0.5*x**2)
        return y

    #morlet小波激活函数的导数
    def de_func_morlet(self, x):
        y = -1.75*np.sin(1.75*x)*np.exp(-0.5*x**2)-x*np.cos(1.75*x)*np.exp(-0.5*x**2)
        return y

    #网络训练模块
    #网络隐含层激活函数采用Morlet小波函数
    #网络输出层激活函数采用sigmoid函数
    #train the neural network
    def train(self, input_list, target_list):
        #网络训练可以分为两部分：
        #第一部分，根据训练样本的输入计算输出。
        #第二部分，将计算得到的输出与真实输出进行对比，然后根据差值（误差）指导网络权重的更新
        #Calculate outputs emerging from neural network model
        #Convert input_list and target_list to 2D arrays
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        #calculate signals into hidden layer
        hidden_layer_inputs = np.dot(self.w_i_h, inputs)
        #Calculate signals emerging from hidden layer
        hidden_layer_outputs = self.func_morlet(hidden_layer_inputs)

        #Calculate signals into final output layer
        final_layer_inputs = np.dot(self.w_h_o, hidden_layer_outputs)
        #calculate signals emerging from final output layer
        final_layer_outputs = self.func_sigmoid(final_layer_inputs)

        #Calculate errors
        #计算误差，包括输出层误差，即模型预测误差，及根据误差逆传播得到的隐含层误差
        #Calculate final output layer errors
        #计算输出层的误差，亦即模型的预测误差，等于目标值-模型输出值
        final_layer_errors = targets - final_layer_outputs
        #Calculate hidden layer errors
        #根据误差逆传播算法计算隐含层误差
        hidden_layer_errors = np.dot(self.w_h_o.T, final_layer_errors)

        #Update weights
        #更新权重，包括更新隐含层与输出层之间的权重矩阵以及输入层与隐含层之间的权重矩阵
        #Update the weights for the links between the hidden and output layers
        self.w_h_o = self.w_h_o + self.l_r*np.dot((final_layer_errors*self.de_func_sigmoid(final_layer_inputs)), np.transpose(hidden_layer_outputs))
        #update the weights for the links between the input and hidden layers
        self.w_i_h = self.w_i_h + self.l_r*np.dot((hidden_layer_errors*self.de_func_morlet(hidden_layer_inputs)), np.transpose(inputs))

        pass

    #网络测试模块
    #query the neural network
    #网络查询或测试模块用于将测试样本输入训练后的网络，然后输出预测结果。
    #功能上可以具体分为三部分：接收输入、计算隐含层输出、计算输出层输出。
    def query(self, input_list):
        #convert inputs list to 2D array
        #将输入数据库或文件中的数据转换为可用于模型输入的格式及形式，例如，将文本型数据转换为数值型数据。
        inputs = np.array(input_list, ndmin=2).T
        #calculate signals into hidden layer
        #计算隐含层的输入信号
        hidden_layer_inputs = np.dot(self.w_i_h, inputs)
        #calculate the signals emerging from hidden layer
        #计算隐含层的输出信号
        hidden_layer_outputs = self.func_morlet(hidden_layer_inputs)

        #calculate signals into final output layer
        #计算输出层的输入信号
        final_layer_inputs = np.dot(self.w_h_o, hidden_layer_outputs)
        #calculate the signals emerging from final output layer
        #计算输出层的输出信号
        final_layer_outputs = self.func_sigmoid(final_layer_inputs)

        #返回测试样本的网络预测输出
        return final_layer_outputs

        pass


#设置要创建的神经网络模型的各参数值
input_layer_nodes = 3 #input_layer_nodes参数用于设置网络输入层神经元个数
hidden_layer_nodes = 120 #hidden_layer_nodes参数用于设置网络隐含层神经元的个数
output_layer_nodes = 3 #output_layer_nodes参数用于设置网络输出层神经元的个数
learning_rate = 0.004 #learning_rate参数用于设置学习率的值
#基于上述设置的参数值采用ANN.py中定义的neuralnetwork类创建一个神经网络模型
nn=neuralnetwork(input_layer_nodes,hidden_layer_nodes,output_layer_nodes,learning_rate)

#Load the training data CSV file into a list
training_data_file = open("D:/Student/", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neural network
#go through all the records in the training data set
epochs = 210
for e in range(epochs):
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = np.asfarray(all_values[:3])
        # create the target output values
        targets = np.asfarray(all_values[3:])
        #train the network
        nn.train(inputs, targets)
        pass
    pass


#query the neural network
test_data_file = open("D:/Student/", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

#定义并初始化一个测试输出矩阵
test_output_matrix = np.zeros((len(test_data_list), 3))

#go through all the records in the test data set
for i in range(len(test_data_list)):
    #split the record by the ',' commas
    record = test_data_list[i]
    all_values = record.split(',')
    #read in inputs for test
    inputs_test = np.asfarray(all_values[:3])
    #query the neural network,output the predications
    output_test = nn.query(inputs_test)

    test_output_matrix[i][0] = output_test[0][0]
    test_output_matrix[i][1] = output_test[1][0]
    test_output_matrix[i][2] = output_test[2][0]

    pass

#print(test_output_matrix)
data = pd.read_csv('D:/Student/', header=None)
column_4 = data[3]
column_5 = data[4]
column_6 = data[5]
targets_matrix = np.zeros((len(column_4), 3)) #目标矩阵
for i in range(len(column_4)):
    targets_matrix[i][0] = column_4[i]
    targets_matrix[i][1] = column_5[i]
    targets_matrix[i][2] = column_6[i]
#print(targets_matrix)
loss_matrix = targets_matrix - test_output_matrix
#print(loss_matrix)
loss = np.sum(loss_matrix**2)
print(loss)

df_w_i_h = pd.DataFrame(nn.w_i_h)
df_w_h_o = pd.DataFrame(nn.w_h_o)
path = 'D:/Student/'
Time = time.strftime("%Y%m%d%H%M%S", time.localtime())
df_w_i_h.to_excel(path+'loss-%.3f-' % loss + 'hidden_nodes-%d-' % hidden_layer_nodes +'learning_rate-%.3f-' % learning_rate +'epochs-%d-' % epochs + '%s-matrix-w_i_h.xlsx' % Time, index = False, header = None)
df_w_h_o.to_excel(path+'loss-%.3f-' % loss + 'hidden_nodes-%d-' % hidden_layer_nodes +'learning_rate-%.3f-' % learning_rate +'epochs-%d-' % epochs + '%s-matrix-w_h_o.xlsx' % Time, index = False, header = None)







