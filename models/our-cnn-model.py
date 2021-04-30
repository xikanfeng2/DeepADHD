from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_curve, auc

tf.logging.set_verbosity(tf.logging.INFO)


class TrainValTensorBoard(TensorBoard): ## using tensorborad to view the traning process
    def __init__(self, log_dir='./logs_dnn_cnn/l2_'+sys.argv[1], **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(
                self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k,
                    v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def cnn_classifier(train_data, train_labels, data_size, pool_size, learning_rate):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        data_size, input_dim=data_size, activation='relu'))
    model.add(keras.layers.Reshape((int(data_size/4), 4)))
    model.add(keras.layers.Conv1D(
        batch_input_shape=(None, int(data_size/4), 4),
        filters=32,
        kernel_size=4,
        padding="valid",
        activation="relu"
    ))
    model.add(keras.layers.MaxPool1D(
        pool_size=pool_size,
        strides=pool_size
    ))
    model.add(keras.layers.Conv1D(
        filters=64,
        kernel_size=4,
        padding="same",
        activation="relu"
    ))
    model.add(keras.layers.MaxPool1D(
        pool_size=pool_size,
        strides=pool_size
    ))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('sigmoid'))
    adam = keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data[100:],
              train_labels[100:],
              epochs=100,
              batch_size=512,
              validation_data=(train_data[:100], train_labels[:100]),
              verbose=1,
                callbacks=[TrainValTensorBoard(write_graph=False)]
              )
    return model


print(datetime.now())
# read dataset data
snps = pd.read_csv('{0}.input.csv'.format(sys.argv[1]))

# get training dataset and test dataset with pre-stored index file
train_index = list(pd.read_csv('train_index.csv', header=None)[0])
test_index = list(pd.read_csv('test_index.csv', header=None)[0])
train = snps.iloc[train_index]
test = snps.iloc[test_index]

# process training dataset
train_data = train.iloc[:, 6:].values
train_data = train_data.astype(np.float16)
train_labels = train.iloc[:, 5].values
train_labels = train_labels.astype(np.int32)
train_labels = train_labels - 1

# process testing dataset
test_data = test.iloc[:, 6:].values
test_data = test_data.astype(np.float16)
test_labels = test.iloc[:, 5].values
test_labels = test_labels.astype(np.int32)
test_labels = test_labels - 1

# training model
class_model = cnn_classifier(train_data, train_labels, int(train_data.shape[1]), 4, 0.0001)
loss, acc = class_model.evaluate(test_data, test_labels)

# calculate AUC and draw the ROC graph
test_scores = class_model.predict_proba(test_data)
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(test_labels.reshape(len(test_labels), 1), test_scores)
roc_auc = auc(fpr, tpr)
training_loss = class_model.history.history['loss']
test_loss = class_model.history.history['val_loss']
acc = class_model.history.history['acc']
val_acc = class_model.history.history['val_acc']
# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# get each curve data
graph_data = pd.DataFrame(columns=['acc', 'val_acc', 'loss', 'val_loss'])
graph_data['acc'] = acc
graph_data['val_acc'] = val_acc
graph_data['loss'] = training_loss
graph_data['val_loss'] = test_loss

# draw graph
plt.figure(1)
plt.subplot(211)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(212)
plt.plot(epoch_count, acc, 'r--')
plt.plot(epoch_count, val_acc, 'b-')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('roc.png')
print(datetime.now())

