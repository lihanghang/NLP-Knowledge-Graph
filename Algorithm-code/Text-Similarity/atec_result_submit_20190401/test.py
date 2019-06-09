#! /usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import os 
import time
import datetime
from tensorflow.contrib import learn
from Input_preprocess import InputHelper
import sys


INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]


print(INPUT_FILE)
print(OUTPUT_FILE)

BATCH_SIZE = 64
VOCAB_FILE = "./vocab/vocab"
MODEL_PATH = "./model/model-11000"
ALLOW_SOFT_PLACEMENT = True
LOG_DEVICE_PLACEMENT = False

MAX_DOCUMENT_LENGTH = 40

inpH = InputHelper()
s1_test, s2_test = inpH.getTestDataSet(INPUT_FILE, VOCAB_FILE, MAX_DOCUMENT_LENGTH)

checkpoint_file = MODEL_PATH
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=ALLOW_SOFT_PLACEMENT,
        log_device_placement=LOG_DEVICE_PLACEMENT
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

    input_s1 = graph.get_operation_by_name("input_x1").outputs[0]
    input_s2 = graph.get_operation_by_name("input_x2").outputs[0]

    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    predictions = graph.get_operation_by_name("output/distance").outputs[0]

    sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]
    batches = inpH.batch_iter(list(zip(s1_test, s2_test)), 2*BATCH_SIZE, 1, shuffle=False)
    all_predictions = []
    all_d = []
    for db in batches:
        s1_dev_b, s2_dev_b = zip(*db)
        batch_predictions, batch_sim = sess.run([predictions, sim], {input_s1: s1_dev_b, input_s2: s2_dev_b,
                                                                    dropout_keep_prob: 1.0})
        all_predictions = np.concatenate([all_predictions, batch_predictions])
        all_d = np.concatenate([all_d, batch_sim])
    for ex in all_predictions:
        print(ex)

    f_output = open(OUTPUT_FILE, "a")
    index = 1
    predictVal = 0
    for item in all_d:
        if item > 0:
            predictVal = 1
        else:
            predictVal = 0
        f_output.write('{}\t{}\n'.format(index, predictVal))
        index += 1

    print("finished!")
