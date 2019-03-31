'''
安装有python 2.7、java 8、tensorflow 1.5、jieba 0.39、pytorch 0.4.0、keras 2.1.6
gensim 3.4.0、pandas 0.22.0、sklearn 0.19.1、xgboost 0.71、lightgbm 2.1.1

Text-similarity model
author: hang hang li
date: 2019-3-30
'''

import tensorflow as tf
import pandas as pd

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        pre_train_df = pd.read_csv(input_file,
                                        names=["index", "s1", "s2", "label"],
                                        header=None, encoding='utf-8',
                                        sep='\t')
        texts_s1_train = pre_train_df['s1'].tolist()
        texts_s2_train = pre_train_df['s2'].tolist()
        texts_label_train = pre_train_df['label'].tolist()
        return texts_label_train

lines = DataProcessor._read_tsv("/home/hanghangli/桌面/图谱应用/Field-Knowledge-Graph/datasets/text-similarity/atec_nlp_sim_train_add.csv")
print(lines)