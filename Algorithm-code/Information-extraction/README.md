# 中文命名实体识别的两种典型算法实现
（IDCNN-CRF、Bi-LSTM-CRF)实现对文本中人名、机构名、地名的识别
## 数据
#### 训练集：example.train
#### 验证集：example.dev
#### 测试集：example.test
#### 中文字符嵌入数据：vec.txt
## 代码相关说明
```Bash
都在main.py文件中进行修改
1. 如果直接使用本代码进行NER，修改代码行：19~20左右的“clean"与“train”两个标记为False，否则都修改为True（表示重新训练模型）。
2. 选择训练使用的模型：代码行52~53左右的model_type标记。
3. 训练完后建议及时执行1步骤，以免重复训练。
```
## 实验结果
【IDCNN-CRF】训练模型速度比较快，从测试集结果来看，对于机构名（ORG）的识别准确率不错。
```Bash
2019-03-12 22:07:17,068 - log/train.log - INFO - evaluate:test
2019-03-12 22:07:17,125 - log/train.log - INFO - processed 1478 tokens with 69 phrases; found: 65 phrases; correct: 57.

2019-03-12 22:07:17,126 - log/train.log - INFO - accuracy:  97.97%; precision:  87.69%; recall:  82.61%; FB1:  85.07

2019-03-12 22:07:17,126 - log/train.log - INFO -               LOC: precision:  86.67%; recall:  83.87%; FB1:  85.25  30

2019-03-12 22:07:17,126 - log/train.log - INFO -               ORG: precision:  93.75%; recall:  83.33%; FB1:  88.24  16

2019-03-12 22:07:17,126 - log/train.log - INFO -               PER: precision:  84.21%; recall:  80.00%; FB1:  82.05  19
```
【BiLSTM-CRF】训练速度较慢，从测试结果来看，对于人名的识别准确率效果不错。
```Bash
2019-03-13 10:43:37,086 - log/train.log - INFO - iteration:100 step:5/305, NER loss: 0.064308
2019-03-13 10:43:49,521 - log/train.log - INFO - iteration:100 step:105/305, NER loss: 0.069525
2019-03-13 10:44:00,461 - log/train.log - INFO - iteration:100 step:205/305, NER loss: 0.059695
2019-03-13 10:44:14,539 - log/train.log - INFO - iteration:101 step:0/305, NER loss: 0.058964
2019-03-13 10:44:14,539 - log/train.log - INFO - evaluate:dev
2019-03-13 10:44:14,598 - log/train.log - INFO - processed 1553 tokens with 47 phrases; found: 54 phrases; correct: 44.

2019-03-13 10:44:14,598 - log/train.log - INFO - accuracy:  98.26%; precision:  81.48%; recall:  93.62%; FB1:  87.13

2019-03-13 10:44:14,598 - log/train.log - INFO -               LOC: precision:  80.00%; recall:  96.00%; FB1:  87.27  30

2019-03-13 10:44:14,598 - log/train.log - INFO -               ORG: precision:  88.89%; recall:  88.89%; FB1:  88.89  9

2019-03-13 10:44:14,598 - log/train.log - INFO -               PER: precision:  80.00%; recall:  92.31%; FB1:  85.71  15

2019-03-13 10:44:14,598 - log/train.log - INFO - evaluate:test
2019-03-13 10:44:14,655 - log/train.log - INFO - processed 1478 tokens with 69 phrases; found: 71 phrases; correct: 60.

2019-03-13 10:44:14,655 - log/train.log - INFO - accuracy:  96.48%; precision:  84.51%; recall:  86.96%; FB1:  85.71

2019-03-13 10:44:14,655 - log/train.log - INFO -               LOC: precision:  80.56%; recall:  93.55%; FB1:  86.57  36

2019-03-13 10:44:14,655 - log/train.log - INFO -               ORG: precision:  84.21%; recall:  88.89%; FB1:  86.49  19

2019-03-13 10:44:14,655 - log/train.log - INFO -               PER: precision:  93.75%; recall:  75.00%; FB1:  83.33  16
```
对比两种方法，准确率和召回率都相差不大。基于扩展卷积的从整个实验过程中来看在训练的速度上要明显快于双向长短时记忆网络，这是由于前者具有并行处理的优势。

-----
Updated by Hang-Hang Li on March 13.   
If you have any questions, please contact lihanghang@ucas.ac.cn

