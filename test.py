import NeuralNetwork as NN
import DataContainer as DC
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as to
from tensorflow_model_optimization.sparsity import keras as sparsity
from kerassurgeon import Surgeon
import ModelPruner as MP

def evaluateNet(evaluations,predictions) :
    losses = []
    MAEs = []
    MSEs = []

    for i in range( len(evaluations) ):
        (loss,MAE,MSE) = evaluations[i]
        losses.append(loss)
        MAEs.append(MAE)
        MSEs.append(MSE)

    print("Averages")
    print("Loss:",np.average(losses))
    print("Mean absolute error(MAE):",np.average(MAEs))
    print("Mean squared error (MSE):",np.average(MSEs))

net = NN.NeuralNetwork
data = DC.DataContainer()
neuralNet = net('relu','sgd','Medium')

evaluationsMedium = []
predictionsMedium = []

count = 1

for i in range(count) :
    neuralNet.train()

    evaluationsMedium.append( neuralNet.evaluate() )

    predictionsMedium.append(neuralNet.predict() )

# print("\n\nPerformed ",count," test runs\n")

# print("\nAverage of evaluations\n")
# evaluateNet(evaluationsMedium,predictionsMedium)

#_, keras_file = tempfile.mkstemp('.h5')
#print('Saving model to: ', keras_file)
#tf.keras.models.save_model(neuralNet.model, keras_file, include_optimizer=False)
#C:\Users\jarro\AppData\Local\Temp\tmpdx9g_ub9.h5



#Prune
# modelPruner = MP.ModelPruner(neuralNet)
# modelPruner.prune()


#create pruned model
prunedNN = neuralNet.prune()


(original_test_loss, original_test_mae, original_test_mse) = evaluationsMedium[0]
(test_loss, test_mae, test_mse) = prunedNN.evaluate(neuralNet.dataContainer.getTestingData(), neuralNet.dataContainer.getTestingLabels(), verbose=0)

lossDiff = test_loss - original_test_loss
MAELoss = test_mae - original_test_mae
MSELoss = test_mse - original_test_mse
print("\nPruned model performance:")
print("Loss:",test_loss," - Diff ",lossDiff)
print("MAE:",test_mae," - Diff ",MAELoss)
print("MSE:",test_mse," - Diff ",MSELoss)

#strip model
final_model = sparsity.strip_pruning(prunedNN)
neuralNet.model.summary()
final_model.summary()