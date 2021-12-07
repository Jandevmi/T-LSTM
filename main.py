# Time-Aware LSTM
# main function for supervised task
# An example dataset is shared but the original synthetic dataset
# can be accessed from http://www.emrbots.org/.
# Inci M. Baytas, 2017
#
# How to run: Give the correct path to the data
# Data is a list where each element is a 3 dimensional matrix which contains same length sequences.
# Instead of zero padding, same length sequences are put in the same batch.
# Example: L is the list containing all the batches with a length of N.
#          L[0].shape gives [number of samples x sequence length x dimensionality]
# Please refer the example bash script
# You can use Split0 as the data.

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import sys
np.set_printoptions(threshold=sys.maxsize)
import math

from TLSTM import TLSTM


def convert_one_hot(label_list):
    for i in range(len(label_list)):
        sec_col = np.ones([label_list[i].shape[0], label_list[i].shape[1], 1])
        label_list[i] = np.reshape(label_list[i], [label_list[i].shape[0], label_list[i].shape[1], 1])
        sec_col -= label_list[i]
        label_list[i] = np.concatenate([label_list[i], sec_col], 2)
    return label_list


def training(path, learning_rate, training_epochs, train_dropout_prob, hidden_dim, fc_dim, key, model_path):
    path_string = path + '/data_train.pkl'
    data_train_batches = pd.read_pickle(path_string)

    path_string = path + '/elapsed_train.pkl'
    elapsed_train_batches = pd.read_pickle(path_string)

    path_string = path + '/label_train.pkl'
    labels_train_batches = pd.read_pickle(path_string)


    number_train_batches = len(data_train_batches)
    print("Train data is loaded!")

    input_dim = data_train_batches[0].shape[2]
    output_dim = labels_train_batches[0].shape[1]

    lstm = TLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    cross_entropy, y_pred, y, logits, labels = lstm.get_cost_acc()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        best_cost = 100
        best_sess = sess
        best_epoch = 0
        for epoch in range(training_epochs):  #
            # Loop over all batches
            total_cost = 0
            for i in range(number_train_batches):  #
                # batch_xs is [number of patients x sequence length x input dimensionality]
                batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], \
                                               elapsed_train_batches[i]
                batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
                _, loss = sess.run([optimizer, cross_entropy], feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys,
                                                   lstm.keep_prob: train_dropout_prob, lstm.time: batch_ts})
                total_cost += loss

            print('total_cost: ' + str(total_cost), end='')
            if total_cost <= best_cost:
                print(' better epoch: ' + str(epoch))
                best_cost = total_cost
                best_sess = sess
                best_epoch = epoch
            else:
                print(' Epoch worse')

            if best_epoch + 10 == epoch:
                print('Break!', end=' ')
                break

        print("Training is over!")
        saver.save(best_sess, model_path)
        saver.restore(sess, model_path)
        Y_pred = []
        Y_true = []
        Labels = []
        Logits = []
        for i in range(number_train_batches):  #
            batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], \
                                           elapsed_train_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(lstm.get_cost_acc(), feed_dict={
                lstm.input:
                    batch_xs, lstm.labels: batch_ys, \
                lstm.keep_prob: train_dropout_prob, lstm.time: batch_ts})

            if i > 0:
                Y_true = np.concatenate([Y_true, y_train], 0)
                Y_pred = np.concatenate([Y_pred, y_pred_train], 0)
                Labels = np.concatenate([Labels, labels_train], 0)
                Logits = np.concatenate([Logits, logits_train], 0)
            else:
                Y_true = y_train
                Y_pred = y_pred_train
                Labels = labels_train
                Logits = logits_train

        total_acc = accuracy_score(Y_true, Y_pred)
        total_auc = roc_auc_score(Labels, Logits, average='micro')
        total_auc_macro = roc_auc_score(Labels, Logits, average='macro')
        f1 = f1_score(Y_true, Y_pred, average='macro')
        print("Y_true")
        print(Y_true)
        print('Y_pred')
        print(Y_pred)
        print("Train F1 = {:.3f}".format(f1))
        print("Train Accuracy = {:.3f}".format(total_acc))
        print("Train AUC = {:.3f}".format(total_auc))
        print("Train AUC Macro = {:.3f}".format(total_auc_macro))


def testing(path, hidden_dim, fc_dim, key, model_path):
    path_string = path + '/data_test.pkl'
    data_test_batches = pd.read_pickle(path_string)

    path_string = path + '/elapsed_test.pkl'
    elapsed_test_batches = pd.read_pickle(path_string)

    path_string = path + '/label_test.pkl'
    labels_test_batches = pd.read_pickle(path_string)


    number_test_batches = len(data_test_batches)

    print("Test data is loaded!")

    input_dim = data_test_batches[0].shape[2]
    output_dim = labels_test_batches[0].shape[1]

    test_dropout_prob = 1.0
    lstm_load = TLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_path)

        Y_true = []
        Y_pred = []
        Logits = []
        Labels = []
        for i in range(number_test_batches):
            batch_xs, batch_ys, batch_ts = data_test_batches[i], labels_test_batches[i], \
                                           elapsed_test_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            c_test, y_pred_test, y_test, logits_test, labels_test = sess.run(lstm_load.get_cost_acc(),
                                                                             feed_dict={lstm_load.input: batch_xs,
                                                                                        lstm_load.labels: batch_ys, \
                                                                                        lstm_load.time: batch_ts, \
                                                                                        lstm_load.keep_prob: test_dropout_prob})
            if i > 0:
                Y_true = np.concatenate([Y_true, y_test], 0)
                Y_pred = np.concatenate([Y_pred, y_pred_test], 0)
                Labels = np.concatenate([Labels, labels_test], 0)
                Logits = np.concatenate([Logits, logits_test], 0)
            else:
                Y_true = y_test
                Y_pred = y_pred_test
                Labels = labels_test
                Logits = logits_test

        total_acc = accuracy_score(Y_true, Y_pred)
        total_auc = roc_auc_score(Labels, Logits, average='micro')
        total_auc_macro = roc_auc_score(Labels, Logits, average='macro')
        f1 = f1_score(Y_true, Y_pred, average='macro')
        print("Y_true")
        print(Y_true)
        print('Y_pred')
        print(Y_pred)
        print("Train F1 = {:.3f}".format(f1))
        print("Train Accuracy = {:.3f}".format(total_acc))
        print("Train AUC = {:.3f}".format(total_auc))
        print("Train AUC Macro = {:.3f}".format(total_auc_macro))


def main(argv):
    training_mode = int(sys.argv[1])
    path = str(sys.argv[2])

    if training_mode == 1:
        learning_rate = float(sys.argv[3])
        training_epochs = int(sys.argv[4])
        dropout_prob = float(sys.argv[5])
        hidden_dim = int(sys.argv[6])
        fc_dim = int(sys.argv[7])
        model_path = str(sys.argv[8])
        training(path, learning_rate, training_epochs, dropout_prob, hidden_dim, fc_dim, training_mode, model_path)
    else:
        hidden_dim = int(sys.argv[3])
        fc_dim = int(sys.argv[4])
        model_path = str(sys.argv[5])
        testing(path, hidden_dim, fc_dim, training_mode, model_path)


if __name__ == "__main__":
    main(sys.argv[1:])
