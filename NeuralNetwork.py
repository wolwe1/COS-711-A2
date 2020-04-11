from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras

import datetime

import PrintUtils

Plotter = PrintUtils.PrintUtils()

import matplotlib.pyplot as plt

class NeuralNetwork :
  def __init__(self,dataContainer) :
    self.setupLayers()
    self.compile()
    self.model
    self.dataContainer = dataContainer
    self.log_dir
    self.tensorboard_callback
    self.testing_data = []
    self.testing_labels = []
    self.epochs = 20

  
  def setupLayers(self) :
    self.model = keras.Sequential([
    keras.layers.Flatten(data_format=None),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    ])

  def compile(self) :
    self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= self.log_dir, histogram_freq=1)

  def train(self) :
    
    training_data = self.dataContainer.getTrainingData()
    training_labels = self.dataContainer.getTrainingLabels()

    self.testing_data = self.dataContainer.getTestingData()
    self.testing_labels = self.dataContainer.getTestingLabels()

    self.model.fit(training_data, training_labels, epochs=self.epochs, callbacks=[self.tensorboard_callback])

  def evaluate(self) :
    print("Evaluating model")
    test_loss, test_acc = self.model.evaluate(self.testing_data,  self.testing_labels, verbose=2)

    
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:',test_loss )


  def predict(self) :
    print("\nMaking predictions\n")
    #print("testDataShape:", self.testing_data[0].shape)
    #data = (np.expand_dims(self.testing_data[0],0))
    #print("After expansion:",data.shape)

    

    probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    predictions = probability_model.predict(self.testing_data)

    print("Guessing first 5 White")

    for i in range(5) :
      print("Model predicted: ",predictions[i] ," actual : " , self.dataContainer.getTestingLabels()[i])

    #Plotter.plot_value_array(i=1, predictions_array=predictions[0], true_label =self.testing_labels)
    #_ = plt.xticks(range(10), range(10), rotation=45)
    #plt.show()

    Plotter.plotMultiPredictions(predictions=predictions,test_labels=self.testing_labels,test_data=self.testing_data)