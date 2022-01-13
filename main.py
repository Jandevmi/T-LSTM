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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys
import asyncio
import math
from TLSTM import TLSTM

np.set_printoptions(threshold=sys.maxsize)


async def training(path, learning_rate, training_epochs, train_dropout_prob, hidden_dim, fc_dim, key, model_path):
    data_train_batches = pd.read_pickle(path + '/data_train.pkl')
    elapsed_train_batches = pd.read_pickle(path + '/time_train.pkl')
    labels_train_batches = pd.read_pickle(path + '/label_train.pkl')

    data_val_batches = pd.read_pickle(path + '/data_test.pkl')
    elapsed_val_batches = pd.read_pickle(path + '/time_test.pkl')
    labels_val_batches = pd.read_pickle(path + '/label_test.pkl')

    print("Train data is loaded!")
    number_train_batches = len(data_train_batches)
    number_val_batches = len(data_val_batches)
    input_dim = data_train_batches[0].shape[2]
    output_dim = labels_train_batches[0].shape[1]

    lstm = TLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    cross_entropy, y_pred, y, logits, labels = lstm.get_cost_acc()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        best_f1, best_epoch = 0, 0
        best_sess = sess
        for epoch in range(training_epochs):
            total_cost = 0
            for i in range(number_train_batches):
                # batch_xs is [number of patients x sequence length x input dimensionality]
                batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], \
                                               elapsed_train_batches[i]
                batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
                _, loss = sess.run([optimizer, cross_entropy], feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys,
                                                                          lstm.keep_prob: train_dropout_prob,
                                                                          lstm.time: batch_ts})
                total_cost += loss
            print("Training Loss: {:.3f}".format(total_cost))

            tasks, y_pred, y_true, labels, logits = [], [], [], [], []
            for i in range(number_val_batches):
                batch_xs, batch_ys, batch_ts = data_val_batches[i], labels_val_batches[i], \
                                               elapsed_val_batches[i]
                batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
                tasks.append(
                    asyncio.create_task(
                        asyncio.coroutine(sess.run)(
                            lstm.get_cost_acc(),
                            feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys,
                                       lstm.keep_prob: train_dropout_prob, lstm.time: batch_ts})))
            results = await asyncio.gather(*tasks)

            for r in results:
                cost, y_pred_val, y_val, logits_val, labels_val = r[0], r[1], r[2], r[3], r[4]
                if np.size(y_true) > 0:
                    y_true = np.concatenate([y_true, y_val], axis=0)
                    y_pred = np.concatenate([y_pred, y_pred_val], axis=0)
                    labels = np.concatenate([labels, labels_val], axis=0)
                    logits = np.concatenate([logits, logits_val], axis=0)
                else:
                    y_true = y_val
                    y_pred = y_pred_val
                    labels = labels_val
                    logits = logits_val

            f1 = f1_score(y_true, y_pred, average='macro')
            print("Validation F1:  {:.3f}".format(f1))
            print(confusion_matrix(y_true, y_pred))

            if f1 > best_f1:
                print('better epoch: ' + str(epoch))
                best_f1 = f1
                best_sess = sess
                best_epoch = epoch
                saver.save(sess, model_path)
            else:
                print('Epoch worse')

            if (best_epoch + 15 == epoch) & (total_cost < 10):
                print('Break!')
                break
            print()
        print("Training is over!")
        saver.restore(best_sess, model_path)

        y_pred, y_true, labels, logits = [], [], [], []
        for i in range(number_train_batches):
            batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], elapsed_train_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(lstm.get_cost_acc(),
                                                                                  feed_dict={lstm.input: batch_xs,
                                                                                             lstm.labels: batch_ys,
                                                                                             lstm.keep_prob: train_dropout_prob,
                                                                                             lstm.time: batch_ts})

            if i > 0:
                y_true = np.concatenate([y_true, y_train], 0)
                y_pred = np.concatenate([y_pred, y_pred_train], 0)
                labels = np.concatenate([labels, labels_train], 0)
                logits = np.concatenate([logits, logits_train], 0)
            else:
                y_true = y_train
                y_pred = y_pred_train
                labels = labels_train
                logits = logits_train

        target_names = ['Success', 'Graft Failure']
        total_auc_macro = roc_auc_score(labels, logits, average='macro')
        print(classification_report(y_true, y_pred, target_names=target_names))
        print("Train AUC Macro = {:.3f}".format(total_auc_macro))
        print(confusion_matrix(y_true, y_pred))


def testing(path, hidden_dim, fc_dim, key, model_path):
    data_test_batches = pd.read_pickle(path + '/data_test.pkl')
    elapsed_test_batches = pd.read_pickle(path + '/time_test.pkl')
    labels_test_batches = pd.read_pickle(path + '/label_test.pkl')

    number_test_batches = len(data_test_batches)

    print("Test data is loaded!")

    input_dim = data_test_batches[0].shape[2]
    output_dim = labels_test_batches[0].shape[1]
    open(path + "/features/IO.txt", 'w').close()
    test_dropout_prob = 1.0
    lstm_load = TLSTM(input_dim, output_dim, hidden_dim, fc_dim, 2)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_path)

        y_true, y_pred, logits, labels = [], [], [], []
        for i in range(number_test_batches):
            batch_xs, batch_ys, batch_ts = data_test_batches[i], labels_test_batches[i], \
                                           elapsed_test_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            c_test, y_pred_test, y_test, logits_test, labels_test = sess.run(lstm_load.get_cost_acc(),
                                                                             feed_dict={lstm_load.input: batch_xs,
                                                                                        lstm_load.labels: batch_ys,
                                                                                        lstm_load.time: batch_ts,
                                                                                        lstm_load.keep_prob: test_dropout_prob})
            if i > 0:
                y_true = np.concatenate([y_true, y_test], 0)
                y_pred = np.concatenate([y_pred, y_pred_test], 0)
                labels = np.concatenate([labels, labels_test], 0)
                logits = np.concatenate([logits, logits_test], 0)
            else:
                y_true = y_test
                y_pred = y_pred_test
                labels = labels_test
                logits = logits_test

        total_auc_macro = roc_auc_score(labels, logits, average='macro')
        target_names = ['Success', 'Graft Failure']
        print(classification_report(y_true, y_pred, target_names=target_names))
        print("Train AUC Macro = {:.3f}".format(total_auc_macro))
        print(confusion_matrix(y_true, y_pred))


def extract_embeddings(path, hidden_dim, fc_dim, key, model_path, embeddings_path):
    data_batches = pd.read_pickle(path + '/data_train.pkl')
    elapsed_batches = pd.read_pickle(path + '/time_train.pkl')
    labels_batches = pd.read_pickle(path + '/label_train.pkl')

    # Delete old features, hard coded atm :(
    open(embeddings_path + "/embeddings.txt", 'w').close()

    number_test_batches = len(data_test_batches)

    print("Data is loaded!")

    input_dim = data_batches[0].shape[2]
    output_dim = labels_batches[0].shape[1]

    test_dropout_prob = 1.0
    lstm_load = TLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_path)

        for i in range(number_test_batches):
            batch_xs, batch_ys, batch_ts = data_batches[i], labels_batches[i], \
                                           elapsed_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            sess.run(lstm_load.get_cost_acc(),
                     feed_dict={lstm_load.input: batch_xs, lstm_load.labels: batch_ys, lstm_load.time: batch_ts,
                                lstm_load.keep_prob: test_dropout_prob})


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
        asyncio.run(
            training(path, learning_rate, training_epochs, dropout_prob, hidden_dim, fc_dim, training_mode, model_path))
    elif training_mode == 0:
        hidden_dim = int(sys.argv[3])
        fc_dim = int(sys.argv[4])
        model_path = str(sys.argv[5])
        testing(path, hidden_dim, fc_dim, training_mode, model_path)
    elif training_mode == 2:
        hidden_dim = int(sys.argv[3])
        fc_dim = int(sys.argv[4])
        model_path = str(sys.argv[5])
        embeddings_path = str(sys.argv[6])
        extract_embeddings(path, hidden_dim, fc_dim, training_mode, model_path, embeddings_path)
    else:
        print("Wrong training mode. Use training_mode: 0=test, 1=training, 2=extract embeddings")


if __name__ == "__main__":
    main(sys.argv[1:])
