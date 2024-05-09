import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import norm, pinv
import pandas as pd
import time

#定义一个RBF神经网络类
class RBF:
    #定义初始化函数，当声明一个RBF类的时候，自动执行这一函数
    #初始化函数接收的变量有三个，这也就意味着声明一个RBF类的时候需要出入三个参数：
    # RBF网络输入层神经元个数、隐含层神经元的个数、输出层神经元的个数
    def __init__(self, input_layer_num, hidden_layer_num, output_layer_num):
        self.input_layer_num = input_layer_num
        self.hidden_layer_num = hidden_layer_num
        self.output_layer_num = output_layer_num
        #径向基函数形状因子，有点类似于高斯分布中的参数sigma，其大小可以决定径向基函数的宽度或“胖瘦”，
        #该参数此处直接指定，可调。
        self.gamma = 0.5
        #初始化径向基函数的中心，中心的个数为RBF网络中间层神经元的个数，每个中心的维度与输入层神经元的个数一致。
        self.centers = [np.random.uniform(-1, 1, input_layer_num) for i in range(hidden_layer_num)]
        #初始化隐含层与输出层之间的权重矩阵，权重矩阵维度中的行数为隐含层神经元个数，列数为输出层神经元的个数
        self.W = np.random.random((self.hidden_layer_num, self.output_layer_num))

    #定义基函数
    def _basisfunc(self, x, c):
        pass
        #接受变量x和c，返回基函数计算值，基函数为：exp{-gamma||x-c|| ** 2}
        #norm()返回各元素平方和的平方根
        return np.exp(-self.gamma*norm(x-c) ** 2)

    #定义隐含层输出矩阵计算函数
    def _calcAct(self, X):
        #初始化隐含层输出矩阵，矩阵维度：行数为输入样本集X中样本的个数，列数为中心点个数，也就是隐含层神经元的个数
        hidden_output = np.zeros((X.shape[0], self.hidden_layer_num), dtype=float)
        #enumerate()函数返回self.centers变量中的元素索引号赋值给c_i_index，元素值赋值给c_i
        for c_i_index, c_i in enumerate(self.centers):
            for x_i_index, x_i in enumerate(X):
                hidden_output[x_i_index, c_i_index] = self._basisfunc(c_i, x_i)
        return hidden_output

    #定义训练函数，RBF的训练，主要需要得到两个量：第一个是中心点，第二个是隐含层与输出层之间的权重矩阵。
    #中心点的训练本例程来用的是样本中随机选取的方式。
    def train(self, X, Y):
        #permutation()函数返回索引号数组，也就是从样本中随机确定中心点
        rnd_idx = np.random.permutation(X.shape[0])[:self.hidden_layer_num]
        #中心点处赋值
        self.centers = [X[i, :] for i in rnd_idx]
        #计算RBF激活函数的值，也就是隐含层输出矩阵
        hidden_outputs = self._calcAct(X)
        #计算权重矩阵
        self.W = np.dot(pinv(hidden_outputs), Y)
        pass

    #定义网络测试函数
    def predict(self, X):
        #计算隐含层输出
        hidden_outputs = self._calcAct(X)
        #计算网络输出
        Y = np.dot(hidden_outputs, self.W)
        #返回预测值
        return Y

#读入训练数据（训练集中的输入矩阵和输出矩阵）
data = pd.read_csv('D:/Student/', header=None)
column_1 = data[0]
column_2 = data[1]
column_3 = data[2]
column_4 = data[3]
column_5 = data[4]
column_6 = data[5]
inputs_matrix = np.zeros((len(column_1), 3))  #输入矩阵
targets_matrix = np.zeros((len(column_4), 3)) #目标矩阵
for i in range(len(column_1)):
    inputs_matrix[i][0] = column_1[i]
    inputs_matrix[i][1] = column_2[i]
    inputs_matrix[i][2] = column_3[i]
for i in range(len(column_3)):
    targets_matrix[i][0] = column_4[i]
    targets_matrix[i][1] = column_5[i]
    targets_matrix[i][2] = column_6[i]
# print(inputs_matrix)
# print(len(inputs_matrix))
# print(targets_matrix)
# print(len(targets_matrix))


#声明一个输入层、隐含层、输出层分别有1,20,1个神经元的RBF网络
rbf = RBF(3,160,3)
#进行网络训练
rbf.train(inputs_matrix, targets_matrix)

#读入测试数据（测试集中的输入矩阵和输出矩阵）
data_test = pd.read_csv('D:/Student/', header=None)
column_1_test = data_test[0]
column_2_test = data_test[1]
column_3_test = data_test[2]
column_4_test = data_test[3]
column_5_test = data_test[4]
column_6_test = data_test[5]
inputs_matrix_test = np.zeros((len(column_1_test), 3))  #输入矩阵
targets_matrix_test = np.zeros((len(column_4_test), 3)) #目标矩阵
for i in range(len(column_1_test)):
    inputs_matrix_test[i][0] = column_1_test[i]
    inputs_matrix_test[i][1] = column_2_test[i]
    inputs_matrix_test[i][2] = column_3_test[i]
for i in range(len(column_4_test)):
    targets_matrix_test[i][0] = column_4_test[i]
    targets_matrix_test[i][1] = column_5_test[i]
    targets_matrix_test[i][2] = column_6_test[i]
# print(inputs_matrix_test)
# print(len(inputs_matrix_test))
# print(targets_matrix_test)
# print(len(targets_matrix_test))

#基于训练后的网络输出预测值
Outputs_matrix_test = rbf.predict(inputs_matrix_test)
loss_matrix = targets_matrix_test - Outputs_matrix_test
loss = np.sum(loss_matrix ** 2)
print(Outputs_matrix_test)
print(len(Outputs_matrix_test))
print(loss)

df_centers = pd.DataFrame(rbf.centers)
df_W = pd.DataFrame(rbf.W)
df_Outputs_matrix_test = pd.DataFrame(Outputs_matrix_test)
path = 'D:/Student/'
Time = time.strftime("%Y%m%d%H%M%S", time.localtime())
df_centers.to_excel(path+'loss-%.8f-' % loss + 'centers-%d-' % rbf.hidden_layer_num  +'gamma-%.2f-' % rbf.gamma + '%s-centers.xlsx' % Time, index = False, header = None)
df_W.to_excel(path+'loss-%.8f-' % loss + 'centers-%d-' % rbf.hidden_layer_num  +'gamma-%.2f-' % rbf.gamma + '%s-matrix-W.xlsx' % Time, index = False, header = None)
df_Outputs_matrix_test.to_excel(path+'loss-%.8f-' % loss + 'centers-%d-' % rbf.hidden_layer_num  +'gamma-%.2f-' % rbf.gamma + '%s-matrix-output.xlsx' % Time, index = False, header = None)












