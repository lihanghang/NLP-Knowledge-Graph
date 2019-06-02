# 使用分段卷积神经网络结合注意力机制实现文本关系提取
## 实验环境  
- python2.7  
- numpy1.13.3  
- pytorch1.0+  

## 实验数据  
- [NYT](https://pan.baidu.com/s/1DZYXaz7i9qRehnjMScKPdQ)提取码: j5g8  

## 实验原理  
> [Neural Relation Extraction with Selective Attention over Instances	](https://www.aclweb.org/anthology/P16-1200)  
> [A Review of Relation Extraction](https://www.cs.cmu.edu/~nbach/papers/A-survey-on-Relation-Extraction.pdf)  
## 代码结构  
├── checkpoints         # 保存预加载模型
├── config.py             # 参数
├── dataset                # 数据目录
│ ├── FilterNYT         # SMALL 数据
│ ├── NYT                 # LARGE 数据
│ ├── filternyt.py
│ ├── __init__.py
│ ├── nyt.py
├── main_att.py        # PCNN+ATT 主文件
├── models               # 模型目录
│ ├── BasicModule.py
│ ├── __init__.py
│ ├── PCNN_ATT.py
├── plot.ipynb
├── README.md
├── utils.py                # 工具函数

## 结果分析  

---
Updated on June 2,2019.
