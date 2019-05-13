import time
from datetime import date
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class UserProfile:
    path_user_profile_table = '../data/user_profile_table.csv'

    data = None
    shape = None

    def __init__(self):
        self.data = pd.read_csv(self.path_user_profile_table)
        print("the shape of {} is {}".format(
            self.path_user_profile_table, self.data.shape))

    def GetMoreUserFeatureFromBalance(self, Balance):
        # this function get more feature of user in Balance table
        # the Balance is the '../data/user_balance_table.csv'
        self.data['city'] = self.data['city'].astype('str')
        self.data = pd.get_dummies(self.data)
        self.data = self.data.set_index('user_id')
        dataid = Balance.data['user_id']
        # you can use this method get more features
        self.data['mean_total_purchase_amt'] = Balance.data[[
            'user_id', 'total_purchase_amt']].groupby(['user_id']).mean()
        self.data['std_total_purchase_amt'] = Balance.data[[
            'user_id', 'total_purchase_amt']].groupby(['user_id']).std()
        self.data['mean_total_redeem_amt'] = Balance.data[[
            'user_id', 'total_redeem_amt']].groupby(['user_id']).mean()
        self.data['std_total_redeem_amt'] = Balance.data[[
            'user_id', 'total_redeem_amt']].groupby(['user_id']).std()
        Balance.data['tBalance_minus_yBalance'] = Balance.data['tBalance'] - Balance.data['yBalance'] - Balance.data['share_amt']
        Balance.data['tBalance_minus_yBalance'] = Balance.data[Balance.data['tBalance_minus_yBalance'] != 0]
        self.data['num_of_effective_operation'] = Balance.data['tBalance_minus_yBalance'].value_counts()
        self.data['tBalance_first_use'] = Balance.data[[
            'user_id', 'report_date']].groupby(['user_id']).min()
        self.data['tBalance_first_use'] = (pd.to_datetime(
            '01/09/2014', dayfirst=True)-self.data['tBalance_first_use']).dt.days
        self.data = self.data.fillna(0)
        self.data['tBalance_first_use'] = self.data['num_of_effective_operation'] / self.data['tBalance_first_use']
        self.data = self.data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        return self.data.astype('float')

class UserBalance:
    path_user_balance_table = '../data/user_balance_table.csv'
    day_purchase = None
    day_redeem = None
    data = None

    def __init__(self):
        self.data = pd.read_csv(self.path_user_balance_table, parse_dates=[1])
        self.shape = self.data.shape
        print("the shape of {} is {}".format(
            self.path_user_balance_table, self.data.shape))

    def CalculateDayPurchaseList(self, clusterID):
        # statistices the purchase for each day and save it in
        DataIndexByUserid = self.data.set_index('user_id')
        DataOfThiscluster = DataIndexByUserid.loc[clusterID]
        date_index_as_date = DataOfThiscluster.set_index('report_date')
        date_sta_as_day = date_index_as_date.resample('D').sum()

        dates = pd.date_range('20130701',periods=427)
        dict_sta_as_zeors = pd.DataFrame(np.zeros((427,1)),index=dates,columns=['total_purchase_amt']) 
        dict_sta_as_one = dict_sta_as_zeors.add(date_sta_as_day['total_purchase_amt'],axis=0).fillna(0)
        self.day_purchase = dict_sta_as_one['total_purchase_amt'].values


    def CalculateDayRedeemList(self, clusterID):
        DataIndexByUserid = self.data.set_index('user_id')
        DataOfThiscluster = DataIndexByUserid.loc[clusterID]
        date_index_as_date = DataOfThiscluster.set_index('report_date')
        date_sta_as_day = date_index_as_date.resample('D').sum()

        dates = pd.date_range('20130701',periods=427)
        dict_sta_as_zeors = pd.DataFrame(np.zeros((427,1)),index=dates,columns=['total_redeem_amt']) 
        dict_sta_as_one = dict_sta_as_zeors.add(date_sta_as_day['total_redeem_amt'],axis=0).fillna(0)
        self.day_redeem = dict_sta_as_one['total_redeem_amt'].values


    def GetTestData(self):
        return self.day_purchase[-30:], self.day_redeem[-30:]

    def GetdataUsedInPredict(self):
        return self.day_purchase[:-30], self.day_redeem[:-30]

    def GetdataAll(self):
        return self.day_purchase,self.day_redeem

class Dayinterest:
    path_mfd_day_share_interest = '../data/mfd_day_share_interest.csv'
    def __init__(self):
        self.data = pd.read_csv(self.path_mfd_day_share_interest)
        print("the shape of {} is {}".format(
            self.path_mfd_day_share_interest, self.data.shape))
    def getdata(self):
        return self.data['mfd_7daily_yield']

class BankShibor:
    path_mfd_bank_shibor = '../data/mfd_bank_shibor.csv'
    def __init__(self):
        self.data = pd.read_csv(self.path_mfd_bank_shibor)
        print("the shape of {} is {}".format(
            self.path_mfd_bank_shibor, self.data.shape))
    def getdata(self):
        return self.data['Interest_1_W']

