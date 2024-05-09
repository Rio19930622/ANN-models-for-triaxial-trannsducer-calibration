import numpy as np
import pandas as pd
import time

#基于Python中的类定义和构建一个极限学习机（ELM，extreme learning machine）模型
class Extreme_learning_machine():
    #定义初始化函数，接收三个参数：输入层神经元个数num_input_nodes，隐含层神经元个数num_hidden_nodes，输出层神经元个数num_output_nodes
    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes):
        self._num_input_nodes = num_input_nodes
        self._num_hidden_nodes = num_hidden_nodes
        self._num_output_nodes = num_output_nodes

        #初始化输入层与隐含层之间的连接权重矩阵_w_input_hidden，偏执向量_bias，隐含层与输出层之间的连接权重矩阵_beta_hidden_output
        #模型输入按行向量排布，因此_w_input_hidden行数等于输入层节点数，列数等于隐含层神经元个数
        #连接权重矩阵_w_input_hidden，_beta_hidden_output的初始化采用均匀分布
        #偏执向量置零，因此，偏执向量在此处不发挥任何作用
        self._w_input_hidden = np.random.uniform(-1., 1., size=(self._num_input_nodes, self._num_hidden_nodes))
        self._bias = np.zeros(shape=(self._num_hidden_nodes,))
        self._beta_hidden_output = np.random.uniform(-1., 1., size=(self._num_hidden_nodes, self._num_output_nodes))

    #定义隐含层激活函数，ELM中只有隐含层的输出需要激活函数，而输出层并不需要激活函数
    def _activation(self, x):
        #激活函数为Sigmoid函数
        return 1. / (1. + np.exp(-x))

    #定义损失函数
    def _loss(self, x, y):
        #损失函数为均方误差函数
        diff = x - y
        return np.sum(diff ** 2)

    #定义训练函数，ELM的训练指的是训练或计算隐含层与输出层之间的权重矩阵_beta_hidden_output
    #输入层与隐含层之间的权重矩阵及偏置向量初始化后无需再进行训练
    #权重矩阵_beta_hidden_output采用矩阵运算进行求解，矩阵运算涉及求广义逆矩阵（Moore-Penrose generalized inverse matrix）
    def train(self, x, y):
        #计算隐含层输出
        #x.dot(y)相当于numpy.dot(x, y)
        hidden_outputs = self._activation(x.dot(self._w_input_hidden)+self._bias)
        #y = hidden_outputs.dot(_beta_hidden_output)
        #_beta_hidden_output = pseudo inverse(hidden_outputs).dot(y)
        #先求隐含层输出的伪逆（广义逆），采用pinv()函数
        hidden_outputs_pinv = np.linalg.pinv(hidden_outputs)
        self._beta_hidden_output = hidden_outputs_pinv.dot(y)

    #方法__call__可以使得类的实例像函数一样被调用
    #用于计算ELM的输出
    def __call__(self, x):
        hidden_outputs = self._activation(x.dot(self._w_input_hidden)+self._bias)
        return hidden_outputs.dot(self._beta_hidden_output)

    def _test(self, x, y):
        #self(x)可以执行__call__()函数
        #self代表了类的实例本身
        prediction = self(x)
        print(prediction)
        print(len(prediction))
        loss = self._loss(prediction, y)
        return loss

num_input_nodes = 3
num_hidden_nodes = 115
num_output_nodes = 3

#read train dataset
data = pd.read_csv('D:/Student/', header=None)
column_1 = data[0]
column_2 = data[1]
column_3 = data[2]
column_4 = data[3]
column_5 = data[4]
column_6 = data[5]
inputs_matrix_train = np.zeros((len(column_1), 3)) #目标矩阵
targets_matrix_train = np.zeros((len(column_4), 3)) #目标矩阵
for i in range(len(column_1)):
    inputs_matrix_train[i][0] = column_1[i]
    inputs_matrix_train[i][1] = column_2[i]
    inputs_matrix_train[i][2] = column_3[i]
    targets_matrix_train[i][0] = column_4[i]
    targets_matrix_train[i][1] = column_5[i]
    targets_matrix_train[i][2] = column_6[i]
# print(inputs_matrix_train)
# print(len(inputs_matrix_train))
# print(targets_matrix_train)
# print(len(targets_matrix_train))


#定义极限学习机模型
ELM_Model = Extreme_learning_machine(num_input_nodes, num_hidden_nodes, num_output_nodes)
#模型训练
ELM_Model.train(inputs_matrix_train, targets_matrix_train)

#read test dataset
data_test = pd.read_csv('D:/Student/', header=None)
column_1_test = data_test[0]
column_2_test = data_test[1]
column_3_test = data_test[2]
column_4_test = data_test[3]
column_5_test = data_test[4]
column_6_test = data_test[5]
inputs_matrix_test = np.zeros((len(column_1_test), 3)) #目标矩阵
targets_matrix_test = np.zeros((len(column_4_test), 3)) #目标矩阵
for i in range(len(column_1_test)):
    inputs_matrix_test[i][0] = column_1_test[i]
    inputs_matrix_test[i][1] = column_2_test[i]
    inputs_matrix_test[i][2] = column_3_test[i]
    targets_matrix_test[i][0] = column_4_test[i]
    targets_matrix_test[i][1] = column_5_test[i]
    targets_matrix_test[i][2] = column_6_test[i]
# print(inputs_matrix_test)
# print(len(inputs_matrix_test))
# print(targets_matrix_test)
# print(len(targets_matrix_test))

#模型测试（验证集上）
test_loss = ELM_Model._test(inputs_matrix_test, targets_matrix_test)

print("value of test_loss is", test_loss)

df_w_i_h = pd.DataFrame(ELM_Model._w_input_hidden)
df_w_h_o = pd.DataFrame(ELM_Model._beta_hidden_output)
path = 'D:/Student/'
Time = time.strftime("%Y%m%d%H%M%S", time.localtime())
df_w_i_h.to_excel(path+'loss-%.6f-' % test_loss + 'hidden_nodes-%d-' % ELM_Model._num_hidden_nodes  + '%s-matrix-w_i_h.xlsx' % Time, index = False, header = None)
df_w_h_o.to_excel(path+'loss-%.6f-' % test_loss + 'hidden_nodes-%d-' % ELM_Model._num_hidden_nodes  + '%s-matrix-w_h_o.xlsx' % Time, index = False, header = None)




