import numpy as np
import pandas as pd
import time
#使用定义的神经网络类（neuralnetwork类）创建特定各层节点个数及学习率的具体神经网络模型
#导入定义了neuralnetwork类的ANN.py文件
import ANN

#设置要创建的神经网络模型的各参数值
input_layer_nodes = 3 #input_layer_nodes参数用于设置网络输入层神经元个数
hidden_layer_nodes = 160#hidden_layer_nodes参数用于设置网络隐含层神经元的个数
output_layer_nodes = 3 #output_layer_nodes参数用于设置网络输出层神经元的个数
learning_rate = 0.003 #learning_rate参数用于设置学习率的值
#基于上述设置的参数值采用ANN.py中定义的neuralnetwork类创建一个神经网络模型
nn=ANN.neuralnetwork(input_layer_nodes,hidden_layer_nodes,output_layer_nodes,learning_rate)

#Load the training data CSV file into a list
training_data_file = open("D:/Student/", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neural network
#go through all the records in the training data set
epochs = 220
for e in range(epochs):
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # read in inputs
        inputs = np.asfarray(all_values[:3])
        # read in targets
        targets = np.asfarray(all_values[3:])
        # train the network
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
# print(targets_matrix)
# print(len(targets_matrix))
loss_matrix = targets_matrix - test_output_matrix
#print(loss_matrix)
loss = np.sum(loss_matrix ** 2)
# print(test_output_matrix)
# print(len(test_output_matrix))
print(loss)

df_w_i_h = pd.DataFrame(nn.w_i_h)
df_w_h_o = pd.DataFrame(nn.w_h_o)
path = 'D:/Student/'
Time = time.strftime("%Y%m%d%H%M%S", time.localtime())
df_w_i_h.to_excel(path+'loss-%.3f-' % loss + 'hidden_nodes-%d-' % hidden_layer_nodes +'learning_rate-%.3f-' % learning_rate +'epochs-%d-' % epochs + '%s-matrix-w_i_h.xlsx' % Time, index = False, header = None)
df_w_h_o.to_excel(path+'loss-%.3f-' % loss + 'hidden_nodes-%d-' % hidden_layer_nodes +'learning_rate-%.3f-' % learning_rate +'epochs-%d-' % epochs + '%s-matrix-w_h_o.xlsx' % Time, index = False, header = None)









