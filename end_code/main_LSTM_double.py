import time
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Loaddata
import Model_LSTM_double
import tensorflow as tf
np.set_printoptions(suppress=True)  # 不以科学计数法输出

if __name__ == "__main__":
    # 整体模型测试
    print('start test')
    print(time.ctime())
    user_profile = Loaddata.UserProfile()
    user_balance = Loaddata.UserBalance()
    user_profile_onehot = user_profile.GetMoreUserFeatureFromBalance(
        user_balance)
    print('get user_profile_onehot')
    print(time.ctime())
    crossrate = 0.8  # 交叉概率
    mutationrate = 0.001  # 变异概率
    popSize = 30  # 种群个数
    # 0用来计数累计，剩下分别是簇的个数 城市 星座 性别 purchase_mean redeem_mean purchase_std redeem_std 交易次数 交易频率 的编码个数
    chromLength_type = [0, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    chromLength = sum(chromLength_type)

    # -------------purchase
    print('start purchase')
    print(time.ctime())
    GA = Model_LSTM_double.genetic()
    pop = GA.getEncoding(popSize, chromLength)
    tempfinal = GA.binary2decimal(pop, chromLength_type, chromLength)
    value = GA.fitness_purchase(tempfinal, user_profile_onehot, 0)
    newpop = pop
    for i in range(10):
        print('第'+str(i+1)+'次循环')
        print(time.ctime())
        value = GA.calfitValue(value)
        newpop = GA.selection(newpop, value)
        newpop = GA.crossover(newpop, crossrate,  chromLength)
        newpop = GA.mutation(newpop, mutationrate, chromLength)
        tempfinal = GA.binary2decimal(newpop, chromLength_type, chromLength)
        value = GA.fitness_purchase(
            tempfinal, user_profile_onehot, i+1)
    temp_purchase, bestvalue_purchase, find_best_purchase = GA.best(newpop, value, chromLength)
    temp_answer_purchase = GA.binary2decimal(temp_purchase, chromLength_type, chromLength)
    print(temp_answer_purchase)
    purchase_predict_return = GA.fitness_predict_purchase(find_best_purchase,
        temp_answer_purchase, user_profile_onehot, user_balance)
    purchase_predict_return = purchase_predict_return.astype('int32')
    print(purchase_predict_return)
    print(time.ctime())
    with open('puchase_predict_lstm_double.txt', 'w') as f:
        for i in range(purchase_predict_return.shape[0]):
            f.write(str(purchase_predict_return[i]))
            f.write('\n')
        f.write(str(temp_answer_purchase))
        f.write('\n')
        f.write(str(find_best_purchase))

    # -------------redeem
    print('start redeem')
    print(time.ctime())
    GA = Model_LSTM_double.genetic()
    pop = GA.getEncoding(popSize, chromLength)
    tempfinal = GA.binary2decimal(pop, chromLength_type, chromLength)
    value = GA.fitness_redeem(tempfinal, user_profile_onehot, 0)
    newpop = pop
    for i in range(10):
        print('第'+str(i+1)+'次循环')
        print(time.ctime())
        value = GA.calfitValue(value)
        newpop = GA.selection(newpop, value)
        newpop = GA.crossover(newpop, crossrate,  chromLength)
        newpop = GA.mutation(newpop, mutationrate, chromLength)
        tempfinal = GA.binary2decimal(newpop, chromLength_type, chromLength)
        value = GA.fitness_redeem(tempfinal, user_profile_onehot, i+1)
    temp_redeem, bestvalue_redeem, find_best_redeem = GA.best(
        newpop, value, chromLength)
    temp_answer_redeem = GA.binary2decimal(
        temp_redeem, chromLength_type, chromLength)
    print(temp_answer_redeem)
    redeem_predict_return = GA.fitness_predict_redeem(find_best_redeem,
        temp_answer_redeem, user_profile_onehot, user_balance)
    redeem_predict_return = redeem_predict_return.astype('int32')
    print(redeem_predict_return)
    print(time.ctime())
    with open('redeem_predict_lstm_double.txt', 'w') as f:
        for i in range(redeem_predict_return.shape[0]):
            f.write(str(redeem_predict_return[i]))
            f.write('\n')
        f.write(str(temp_answer_redeem))
        f.write('\n')
        f.write(str(find_best_redeem))
    # 导出csv
    print('import csv')
    print(time.ctime())
    pre = {}
    pre['time'] = np.array([20140901,
                            20140902,
                            20140903,
                            20140904,
                            20140905,
                            20140906,
                            20140907,
                            20140908,
                            20140909,
                            20140910,
                            20140911,
                            20140912,
                            20140913,
                            20140914,
                            20140915,
                            20140916,
                            20140917,
                            20140918,
                            20140919,
                            20140920,
                            20140921,
                            20140922,
                            20140923,
                            20140924,
                            20140925,
                            20140926,
                            20140927,
                            20140928,
                            20140929,
                            20140930,
                            ])
    pre['total_purchase_amt'] = purchase_predict_return
    pre['total_redeem_amt'] = redeem_predict_return
    df = pd.DataFrame(pre)
    df.to_csv('outfile_lstm_double.csv', header=False, index=False, columns=[
        'time', 'total_purchase_amt', 'total_redeem_amt'])
    print('end')
    print(time.ctime())
