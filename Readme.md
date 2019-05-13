
蚂蚁金服拥有上亿会员并且业务场景中每天都涉及大量的资金流入和流出，面对如此庞大的用户群，资金管理压力会非常大。在既保证资金流动性风险最小，又满足日常业务运转的情况下，精准地预测资金的流入流出情况变得尤为重要。通过对例如余额宝用户的申购赎回数据的把握，精准预测未来每日的资金流入流出情况。对货币基金而言，资金流入意味着申购行为，资金流出为赎回行为。命题中使用的数据主要包含四个部分，分别为用户基本信息数据user_profile_table、用户申购赎回数据user_balance_table、收益率表mfd_day_share_interest 和银行间拆借利率表mfd_bank_shibor。<br>
题目来源于https://tianchi.aliyun.com/competition/entrance/231573/introduction?spm=5176.12281925.0.0.739c7137KbwTLG <br>


* 整个框架<br>
-------------
![structure](https://github.com/cgq45120/Tianchi-small-demo-solve-big-data-of-purchase-and-redeem-for-YuEbao/blob/master/end_code/picture/structure_of_tianchi_data.png)
<br>
整个框架分为三部分，分别为遗传算法部分，人员聚类部分和资金预测部分，三部分形成一个进化迭代的整体。<br>

遗传算法部分接受资金预测部分提供的预测损失作为种群的适应度，并且维护一个金融特征缩放比例和聚类参数的基因种群（该种群将被用于对交易人员进行聚类）。通过对种群的不断优化，不断的产生适应度更高的金融特征缩放比例和聚类参数。<br>

聚类算法部在每个循环从遗传算法部分获取金融特征缩放比例和聚类参数用于人员聚类，将行为特征相近的客户看成一个整体。将分散的客户看成一个整体进行预测可以在有效的减少客户行为随机性带来的误差的同时，减少预测的时间成本和运算成本。<br>

资金预测部分将在聚类结果的基础上对每个类别的资金流入流出量进行预测。并计算预测的损失值，在遗传方法的优化过程，损失值将用于指导遗传算法种群的进化，在实际的预测过程，损失值将用于对总体模型效果进行评估。<br>

* 总的模型<br>
-------------
随机产生种群数，得到特征缩放基因编码和聚类个数的基因编码，将其与数据预处理后得到的9个聚类特征放进模型中进行训练与测试，即子流程开始。其次，根据基因编码得到弹性缩放程度，并进行数据缩放，根据体现用户消费习惯的9个特征采用k-means聚类算法进行聚类，并将每一类用户在2013年7月1日-2014年8月1日的购买赎回数据放进双层LSTM神经网络中训练，以此预测用户在2014年8月的购买总量和赎回总量，将测试结果与真实数据作对比，得到测试误差，从而得到适应度。最后，经过子流程，基因编码得到进化，并依据适应度选择父辈染色体，适应度高的个体被选中的概率高，适应度低的个体被淘汰，用父母的染色体按照一定的方法进行交叉，生成子代，并对子代染色体进行变异；由交叉和变异产生新一代种群，不断迭代，直到满足预测要求。<br>

* GA<br>
-------------
![GA](https://github.com/cgq45120/Tianchi-small-demo-solve-big-data-of-purchase-and-redeem-for-YuEbao/blob/master/end_code/picture/GA_structure.png)
遗传算法及求解适应度流程图<br>

在本解决方案中，将遗传算法用于搜寻最优特征缩放比率与最优分类簇个数，其中每一个聚类簇放入双隐层LSTM神经网络中进行训练并不断调整反馈，从而预测出8月份购买量和赎回量数据，并将所有聚类簇的预测误差累加得到单次总模型测试误差，即适应度值。然后根据适应度对基因进行选择、交叉和变异，不断迭代，直到测试误差达到所定阈值即可结束。在得到最优特征缩放比率与最优聚类簇个数的同时，保存最优训练模型。在此基础上，加入8月份实际数据对模型参数略微调整后，进行9月份实际预测。其中本文设定遗传算法种群为30个，交叉概率0.8，变异概率0.01，特征缩放比率范围为0~5，聚类个数范围为0~4，双隐层LSTM训练次数为40次，学习率为0.00006。采用2013.7-2014.7进行训练，2014.8用于测试，保存最优模型。<br>

* Double lstm<br>
-----------------
![double_lstm](https://github.com/cgq45120/Tianchi-small-demo-solve-big-data-of-purchase-and-redeem-for-YuEbao/blob/master/end_code/picture/double_lstm.png)
<br>
双层LSTM的“双”体现在两层隐藏层，在LSTM网络中遗忘门是关键，数据通过遗忘门可以排除对预测效果不利或者效果甚微的数据，从而使预测效果达到最优，但单一的遗忘门排除数据的效果较为低下，且隐层数的增加可以降低网络误差，提高精度。因此将数据通过输入门得到数据后利用双遗忘门处理数据再进行数据的输出不失为一种较为优异的方法<br>

* Answer<br>
----------------

结果只是稍微训练做了下微调，如下所示<br>
![purchase](https://github.com/cgq45120/Tianchi-small-demo-solve-big-data-of-purchase-and-redeem-for-YuEbao/blob/master/end_code/picture/purchase.png)
![redeem](https://github.com/cgq45120/Tianchi-small-demo-solve-big-data-of-purchase-and-redeem-for-YuEbao/blob/master/end_code/picture/redeem.png)

<br>
其中购买量预测总和与实际相比，平均误差为22.1%,赎回量预测总和与实际相比，平均误差为12.3%,模型预测的8月份赎回量趋势与实际趋势大致相同，但随着天数的增加，误差也会越来越大，预测的趋势也会有相应较大程度的偏离。这是符合模型的。最后通过筛选的模型进行最后的9月份的预测<br>

![purchase](https://github.com/cgq45120/Tianchi-small-demo-solve-big-data-of-purchase-and-redeem-for-YuEbao/blob/master/end_code/picture/purchase_for_predict.png)
![redeem](https://github.com/cgq45120/Tianchi-small-demo-solve-big-data-of-purchase-and-redeem-for-YuEbao/blob/master/end_code/picture/redeem_for_predict.png)

<br>

主要程序采用并行计算<br>
python endcode/main_LSTM_double.py

* Environment:<br>
---------------
pip install tensorflow<br>
pip install matplotlib<br>
pip install numpy <br>
pip install pandas <br>
pip install sklearn<br>

* Structure:<br>
---------------
data 来源于https://tianchi.aliyun.com/competition/entrance/231573/information 里的数据，因为太大所以没有上传<br>


+--CHUANGFU-2019<br>
|      +--end_code<br>
|      |      +--environment.sh<br>
|      |      +--Loaddata.py<br>
|      |      +--main_LSTM_double.py<br>
|      |      +--Model_LSTM_double.py<br>
|      |      +--__pycache__<br>
|      |      |      +--cgq_Loaddata.cpython-35.pyc<br>
|      |      |      +--Model_LSTM_double.cpython-35.pyc<br>
+--data<br>
|      +--comp_predict_table.csv<br>
|      +--mfd_bank_shibor.csv<br>
|      +--mfd_day_share_interest.csv<br>
|      +--user_balance_table.csv<br>
|      +--user_profile_table.csv<br>
|      +--项目说明.docx<br>
