import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.cluster import KMeans
import Loaddata
from numpy import random
import time
from datetime import date
import matplotlib.pyplot as plt
import os
from pandas import DataFrame, concat
import multiprocessing as mp


class LSTM_double:
    # 定义常量
    def __init__(self, data):
        self.rnn_unit = 300
        self.input_size = 100
        self.output_size = 1
        self.lr = 0.00006
        self.time_step = 1
        self.batch_size = 1
        self.data = self.series_to_supervised(data, 100)
        self.train_begin = 0
        self.train_end = len(self.data)
        self.test_begin = len(self.data)-1
        self.weights = {
            'in': tf.Variable(tf.random_normal([self.input_size, self.rnn_unit])),
            'out': tf.Variable(tf.random_normal([self.rnn_unit, self.output_size]))
        }
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.rnn_unit, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }

# 定义分割函数
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()

        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        agg = concat(cols, axis=1)
        agg.columns = names

        if dropnan:
            agg.dropna(inplace=True)
        return agg.values

    # 获取训练集
    def get_train_data(self):
        batch_index = []
        data_train = self.data[self.train_begin:self.train_end]
        normalized_train_data = data_train/1e8
        train_x, train_y = [], []  # 训练集
        for i in range(len(normalized_train_data)-self.time_step):
            if i % self.batch_size == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i+self.time_step, :100]
            y = normalized_train_data[i:i+self.time_step, 100:]

            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data)-self.time_step))
        return batch_index, train_x, train_y

    # 获取测试集
    def get_test_data(self):
        data_test = self.data[self.test_begin:]
        normalized_test_data = data_test/1e8
        size = (len(normalized_test_data) +
                self.time_step)//self.time_step  # 有size个sample
        test_x, test_y = [], []
        for i in range(size-1):
            x = normalized_test_data[i *
                                     self.time_step:(i+1)*self.time_step, :100]
            y = normalized_test_data[i *
                                     self.time_step:(i+1)*self.time_step, 100]
            test_x.append(x.tolist())
            test_y.extend(y)
        test_x.append(
            (normalized_test_data[(i+1)*self.time_step:, :100]).tolist())
        test_y.extend(
            (normalized_test_data[(i+1)*self.time_step:, 100]).tolist())
        return test_x, test_y

    # ——————————————————定义神经网络变量——————————————————
    def lstm(self, X):

        self.batch_size = tf.shape(X)[0]
        self.time_step = tf.shape(X)[1]

        w_in = self.weights['in']
        b_in = self.biases['in']

        # 将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input = tf.reshape(X, [-1, self.input_size])
        input_rnn = tf.matmul(input, w_in)+b_in
        # 将tensor转成3维，作为lstm cell的输入
        input_rnn = tf.reshape(input_rnn, [-1, self.time_step, self.rnn_unit])

        cell = tf.nn.rnn_cell.LSTMCell(self.rnn_unit)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output_rnn, final_states = tf.nn.dynamic_rnn(
            cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, self.rnn_unit])  # 作为输出层的输入

        w_out = self.weights['out']
        b_out = self.biases['out']
        pred = tf.matmul(output, w_out)+b_out

        pred = tf.reshape(pred, [-1, self.output_size])
        return pred, final_states

    # ——————————————————训练模型——————————————————
    def train_lstm(self, num_epochs=40, numb_sub=1,numb_class=1,continue_train=False,class_people='purchase'):
        X = tf.placeholder(tf.float32, shape=[None, 1, 100])
        Y = tf.placeholder(tf.float32, shape=[None, 1, 1])
        batch_index, train_x, train_y = self.get_train_data()

        with tf.variable_scope("sec_lstm"):
            pred, _ = self.lstm(X)
    # 损失函数
        loss = tf.reduce_mean(
            tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1])))

        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        if continue_train==True:
            module_file = tf.train.latest_checkpoint('model_save_'+class_people+'_'+
                                              str(numb_sub)+'_'+str(numb_class))    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if continue_train==True:
                saver.restore(sess, module_file)
            # 重复训练
            for i in range(num_epochs):
                for step in range(len(batch_index)-1):
                    _, loss_ = sess.run([train_op, loss], feed_dict={
                                        X: train_x[batch_index[step]:batch_index[step+1]], Y: train_y[batch_index[step]:batch_index[step+1]]})
                print(i+1, loss_)
                if ((i+1) % num_epochs) == 0:
                    print("保存模型：", saver.save(sess, 'model_save_'+class_people+'_' +
                                              str(numb_sub)+'_'+str(numb_class)+'/modle.ckpt', global_step=i))

    # ————————————————预测模型————————————————————
    def prediction(self, numb_sub=1,numb_class=1,class_people='purchase'):
        self.time_step = 1
        self.input_size = 100
        self.output_size = 1
        X = tf.placeholder(tf.float32, shape=[
                           None, self.time_step, self.input_size])
        Y = tf.placeholder(tf.float32, shape=[
                           None, self.time_step, self.output_size])
        test_x, test_y = self.get_test_data()

        with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
            pred, _ = self.lstm(X)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint(
                'model_save_'+class_people+'_'+str(numb_sub)+'_'+str(numb_class))
            saver.restore(sess, module_file)

            test_x = test_x[:1]
            test_x = [a[0] for a in test_x]
            test_x = np.array(test_x)
            test_x[:, :99] = test_x[:, 1:]
            test_x[:, 99:] = test_y[-1]
            test_predict = []

            for step in range(30):
                prob = sess.run(pred, feed_dict={X: [test_x]})
                predict = prob.reshape(-1)
                test_predict.extend(prob)
                test_x[:, :99] = test_x[:, 1:]
                test_x[:, 99:] = prob[-1]

            test_predict = np.array(test_predict)
            test_predict = test_predict[:, 0]
            test_predict = test_predict.flatten()

            test_predict = np.array(test_predict)*1e8
            print(test_predict)
        return test_predict


class k_mean(object):
    def __init__(self, data):
        self.x_train = data

    def k_mean_divide(self, cluster_num):
        kmeans = KMeans(n_clusters=cluster_num,
                        random_state=0).fit(self.x_train)
        divide_labels = kmeans.labels_
        divide_class = {}
        for i in range(cluster_num):
            divide_answer = (divide_labels == i)
            divide = []
            for j in range(len(divide_labels)):
                if divide_answer[j] == True:
                    divide.append(j)
            divide_class['cluster'+str(i)] = np.array(divide)+1
        return divide_class


class genetic(object):
    def getEncoding(self, popSize, chromLength):  # 生成种群
        pop = random.randint(0, 2, size=(popSize, chromLength))
        return pop

    def binary2decimal(self, pop, chromLength_type, chromLength):
        row = pop.shape[0]
        chromLength_length = len(chromLength_type) - 1
        tempfinal = np.zeros((row, chromLength_length))
        position_sum = np.cumsum(chromLength_type)

        for i in range(row):
            for j in range(chromLength_length):
                t = 0
                for k in range(position_sum[j], position_sum[j+1]):
                    t += pop[i, k]*(math.pow(2, k - position_sum[j]))
                tempfinal[i, j] = t
        tempfinal[:, 0] = tempfinal[:, 0]+1
        tempfinal[:, 1:] = tempfinal[:, 1:]/(math.pow(2, 8)-1)*5
        return tempfinal

    def multiprocess_fitness_purchase(self, j):# 并行计算
        multiple_time = np.hstack((self.tempfinal[j, 1], np.tile(
            self.tempfinal[j, 2], 7), np.tile(self.tempfinal[j, 3], 12)))  # 拼接倍数
        for k in range(4, self.tempfinal.shape[1]):
            multiple_time = np.hstack((multiple_time, self.tempfinal[j, k]))
        user_profile_onehot = self.user_profile_onehot * multiple_time  # 将部分向量的权重扩大
        model_kmean = k_mean(user_profile_onehot)  # 聚类
        divide_class = model_kmean.k_mean_divide(int(self.tempfinal[j, 0]))
        user_balance = Loaddata.UserBalance()
        purchase_predict_class = []
        purchase_test_class = []
        for i in range(len(divide_class)):  # 将这几种分类分别带入网络识别
            print('第'+str(j+1)+'个种群 第'+str(i+1)+'个类')
            user_balance.CalculateDayPurchaseList(
                divide_class['cluster'+str(i)])
            user_balance.CalculateDayRedeemList(
                divide_class['cluster'+str(i)])
            purchase_train, redeem_train = user_balance.GetdataUsedInPredict()
            purchase_test, redeem_test = user_balance.GetTestData()
            purchase_model = LSTM_double(purchase_train.reshape((-1, 1)))
            purchase_model.train_lstm(numb_sub=j+1,numb_class=i+1)
            purchase_predict = purchase_model.prediction(numb_sub=j+1,numb_class=i+1)
            tf.reset_default_graph()
            plt.plot(purchase_predict, 'b')
            plt.plot(purchase_test, 'g')
            if not os.path.exists('out_lstm_double/'):
                os.makedirs('out_lstm_double/')
            plt.savefig('out_lstm_double/purchase_the_{}_times_the_{}_gene_the_{}_class.png'.format(
                str(self.times_calc), str(j+1), str(i+1)))
            plt.close()
            purchase_predict_class.append(purchase_predict)
            purchase_test_class.append(purchase_test)
        purchase_loss_value = np.mean(abs(np.array(purchase_predict_class).sum(
            axis=0) - np.array(purchase_test_class).sum(axis=0))/(np.array(purchase_test_class).sum(axis=0)))
        return 1/purchase_loss_value

    def fitness_purchase(self, tempfinal, user_profile_onehot, times_calc):  # 适应度
        self.user_profile_onehot = user_profile_onehot
        self.tempfinal = tempfinal
        self.times_calc = times_calc
        pool = mp.Pool(processes=tempfinal.shape[0])
        purchase_loss_value = pool.map(
            self.multiprocess_fitness_purchase, range(tempfinal.shape[0]))
        pool.close()
        pool.join()
        return np.squeeze(purchase_loss_value)

    def fitness_predict_purchase(self,length_best, tempfinal, user_profile_onehot, user_balance):
        multiple_time = np.hstack((tempfinal[0, 1], np.tile(
            tempfinal[0, 2], 7), np.tile(tempfinal[0, 3], 12)))  # 拼接倍数
        for k in range(4, tempfinal.shape[1]):
            multiple_time = np.hstack((multiple_time, tempfinal[0, k]))
        user_profile_onehot = user_profile_onehot * multiple_time  # 将部分向量的权重扩大
        model_kmean = k_mean(user_profile_onehot)  # 聚类
        divide_class = model_kmean.k_mean_divide(int(tempfinal[0, 0]))
        purchase_predict_class = []
        for i in range(len(divide_class)):  # 将这几种分类分别带入网络识别
            user_balance.CalculateDayPurchaseList(
                divide_class['cluster'+str(i)])
            user_balance.CalculateDayRedeemList(divide_class['cluster'+str(i)])
            purchase_train, redeem_train = user_balance.GetdataAll()

            purchase_model = LSTM_double(purchase_train.reshape((-1, 1)))
            purchase_model.train_lstm(num_epochs = 10,numb_sub = length_best,numb_class=i+1,continue_train=True)
            purchase_predict = purchase_model.prediction(numb_sub=length_best,numb_class=i+1)
            tf.reset_default_graph()

            purchase_predict_class.append(purchase_predict)
        purchase_predict_return = np.array(purchase_predict_class).sum(axis=0)
        return purchase_predict_return

    def multiprocess_fitness_redeem(self, j):
        multiple_time = np.hstack((self.tempfinal[j, 1], np.tile(
            self.tempfinal[j, 2], 7), np.tile(self.tempfinal[j, 3], 12)))  # 拼接倍数
        for k in range(4, self.tempfinal.shape[1]):
            multiple_time = np.hstack((multiple_time, self.tempfinal[j, k]))
        user_profile_onehot = self.user_profile_onehot * multiple_time  # 将部分向量的权重扩大
        model_kmean = k_mean(user_profile_onehot)  # 聚类
        divide_class = model_kmean.k_mean_divide(int(self.tempfinal[j, 0]))
        user_balance = Loaddata.UserBalance()
        redeem_predict_class = []
        redeem_test_class = []
        for i in range(len(divide_class)):  # 将这几种分类分别带入网络识别
            print('第'+str(j+1)+'个种群 第'+str(i+1)+'个类')
            user_balance.CalculateDayPurchaseList(
                divide_class['cluster'+str(i)])  # 主要时间花在这里！！！！
            user_balance.CalculateDayRedeemList(
                divide_class['cluster'+str(i)])
            purchase_train, redeem_train = user_balance.GetdataUsedInPredict()
            purchase_test, redeem_test = user_balance.GetTestData()
            redeem_model = LSTM_double(redeem_train.reshape((-1, 1)))
            redeem_model.lr = 0.0001
            redeem_model.train_lstm(num_epochs=60, numb_sub=j+1,numb_class=i+1,class_people='redeem')
            redeem_predict = redeem_model.prediction(numb_sub=j+1,numb_class=i+1,class_people='redeem')
            tf.reset_default_graph()
            plt.plot(redeem_predict, 'b')
            plt.plot(redeem_test, 'g')
            plt.savefig('out_lstm_double/redeem_the_{}_times_the_{}_gene_the_{}_class.png'.format(
                str(self.times_calc), str(j+1), str(i+1)))
            plt.close()
            redeem_predict_class.append(redeem_predict)
            redeem_test_class.append(redeem_test)
        redeem_loss_value = np.mean(abs(np.array(redeem_predict_class).sum(
            axis=0) - np.array(redeem_test_class).sum(axis=0))/(np.array(redeem_test_class).sum(axis=0)))
        return 1/redeem_loss_value

    def fitness_redeem(self, tempfinal, user_profile_onehot, times_calc):  # 适应度
        self.user_profile_onehot = user_profile_onehot
        self.tempfinal = tempfinal
        self.times_calc = times_calc
        pool = mp.Pool(processes=tempfinal.shape[0])
        redeem_loss_value = pool.map(
            self.multiprocess_fitness_redeem, range(tempfinal.shape[0]))
        pool.close()
        pool.join()
        return np.squeeze(redeem_loss_value)

    def fitness_predict_redeem(self,length_best, tempfinal, user_profile_onehot, user_balance):
        multiple_time = np.hstack((tempfinal[0, 1], np.tile(
            tempfinal[0, 2], 7), np.tile(tempfinal[0, 3], 12)))  # 拼接倍数
        for k in range(4, tempfinal.shape[1]):
            multiple_time = np.hstack((multiple_time, tempfinal[0, k]))
        user_profile_onehot = user_profile_onehot * multiple_time  # 将部分向量的权重扩大
        model_kmean = k_mean(user_profile_onehot)  # 聚类
        divide_class = model_kmean.k_mean_divide(int(tempfinal[0, 0]))
        redeem_predict_class = []
        for i in range(len(divide_class)):  # 将这几种分类分别带入网络识别
            user_balance.CalculateDayPurchaseList(
                divide_class['cluster'+str(i)])
            user_balance.CalculateDayRedeemList(divide_class['cluster'+str(i)])
            purchase_train, redeem_train = user_balance.GetdataAll()
            # LSTM_double
            redeem_model = LSTM_double(redeem_train.reshape((-1, 1)))
            redeem_model.lr = 0.0001
            redeem_model.train_lstm(num_epochs=10,numb_sub = length_best,numb_class=i+1,continue_train=True,class_people='redeem')
            redeem_predict = redeem_model.prediction(numb_sub = length_best,numb_class=i+1,class_people='redeem')
            tf.reset_default_graph()

            redeem_predict_class.append(redeem_predict)
        redeem_predict_return = np.array(redeem_predict_class).sum(axis=0)
        return redeem_predict_return

    def calfitValue(self, value):  # 保证损失大于等于0 好像没什么必要的样子
        for i in range(value.shape[0]):
            if value[i] < 0:
                value[i] = 0
        return value

    def selection(self, pop, value):  # 选择
        newfitvalue = np.zeros((value.shape[0], 1))
        totalValue = sum(value)
        accumalator = 0
        j = 0
        for i in value:  # 轮盘赌
            newValue = (i*1.0/totalValue)
            accumalator += newValue
            newfitvalue[j] = (accumalator)
            j = j+1
        newfitvalue[j-1] = 1
        ms = []
        for i in range(value.shape[0]):
            ms.append(random.random())
        ms.sort()
        fitin = 0
        newin = 0
        newpop = pop
        while newin < value.shape[0]:
            if(ms[newin] < newfitvalue[fitin]):
                newpop[newin] = pop[fitin]
                newin = newin+1
            else:
                fitin = fitin+1
        return newpop

    def crossover(self, pop, crossrate, chromLength):  # 交叉
        row = pop.shape[0]-1  # 确保有两个基因能够对位交叉
        pop = pop.tolist()
        for i in range(0, row, 2):
            if(random.random() < crossrate):  # 对基因块的不同部分进行交叉部位生成
                singpoint = random.randint(chromLength)
                temp1 = []
                temp2 = []
                temp1.extend(pop[i][0:singpoint])
                temp1.extend(pop[i + 1][singpoint:chromLength])
                temp2.extend(pop[i + 1][0:singpoint])
                temp2.extend(pop[i][singpoint:chromLength])
                pop[i] = temp1  # 生成新子群
                pop[i + 1] = temp2
        pop = np.array(pop)
        return pop

    def mutation(self, pop, mutationrate, chromLength):  # 变异
        row = pop.shape[0]
        for i in range(row):
            if (random.random() < mutationrate):
                mpoint = random.randint(0, chromLength)  # 变异部位
                if(pop[i, mpoint] == 1):
                    pop[i, mpoint] = 0
                else:
                    pop[i, mpoint] = 1
        return pop

    def best(self, pop, value, chromLength):
        bestvalue = value.max()
        find_best = np.argmax(value)
        temp = pop[find_best, :].reshape((-1, chromLength))
        return temp, bestvalue, find_best+1
