# Audio Conformer

使用wenet框架，conformer网络
使用MSDM数据集的audio部分数据，在不同严重等级中根据人级别划分测试训练集

测试集：
严重(FDA 29-57):  S_F_00010 S_M_00009
中度(FDA 58-91):  S_M_00047 S_M_00005 S_M_00061
轻度(FDA 82-115): S_M_00059 S_M_00060 S_M_00032 S_M_00024 S_M_00072 S_M_00057
正常(FDA 116):  N_F_10008 N_F_10001 N_M_10012 N_M_10015 S_F_00019

结果CER：
attention_rescoring:  14.52 % N=25372 C=21849 S=3161 D=362 I=162
CTC_greedy: 15.04 % N=25372 C=21772 S=3242 D=358 I=216