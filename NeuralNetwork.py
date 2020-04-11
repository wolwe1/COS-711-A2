#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras

import pathlib

import datetime

import DataContainer as DC

# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling

import PrintUtils

Plotter = PrintUtils.PrintUtils()

import matplotlib.pyplot as plt

_dataContainer = DC.DataContainer()


class NeuralNetwork:

    def __init__(self):

        self.dataContainer = _dataContainer
        self.testing_data = []
        self.testing_labels = []
        self.epochs = 20

        self.model = self.buildModel()

  # For tensorboard

        self.log_dir = 'logs\\fit\\' \
            + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)

    def buildModel(self):
        inputDataShape = len(self.dataContainer.getColumnNames())-1 #-1 for quality

        self.model = keras.Sequential([keras.layers.Dense(64,
                activation='relu', input_shape=[inputDataShape]),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1)])

        sgdOptimizer = keras.optimizers.SGD(learning_rate=0.01,
                momentum=0.0, nesterov=False)

        self.model.compile(optimizer=sgdOptimizer,
                           loss= 'mse',
                           metrics=['mae', 'mse'])

        return self.model

    def train(self):

        training_data = self.dataContainer.getTrainingData()
        training_labels = self.dataContainer.getTrainingLabels()

        self.testing_data = self.dataContainer.getTestingData()
        self.testing_labels = self.dataContainer.getTestingLabels()

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)

        early_history = self.model.fit(
            training_data,
            training_labels,
            epochs=self.epochs,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stop, self.tensorboard_callback],
            )

    def evaluate(self):
      
        (test_loss, test_mae, test_mse) = self.model.evaluate(self.testing_data,self.testing_labels, verbose=2)

        print('Testing set accuracy: {:5.2f} points'.format(test_mae))

        print('\nTest loss:', test_loss)

    def predict(self):
        print('''Making predictions''')

  # probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

  # predictions = probability_model.predict(self.testing_data)

        test_predictions = model.predict(self.testing_data).flatten()

  # Plotter.plot_value_array(i=1, predictions_array=predictions[0], true_label =self.testing_labels)
  # _ = plt.xticks(range(10), range(10), rotation=45)
  # plt.show()

        a = plt.axes(aspect='equal')
        plt.scatter(self.testing_labels, test_predictions)
        plt.xlabel('True Values [Point]')
        plt.ylabel('Predictions [Points]')
        lims = [0, 10]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)

        error = test_predictions - self.testing_labels
        plt.hist(error, bins=25)
        plt.xlabel('Prediction Error [Points]')
        _ = plt.ylabel('Count')


  # Plotter.plotMultiPredictions(predictions=test_predictions,test_labels=self.testing_labels,test_data=self.testing_data)
