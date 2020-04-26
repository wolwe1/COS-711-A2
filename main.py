import NeuralNetwork as NN 
import DataContainer as DC
import numpy as np


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

def calculateAccuracy(predictions,truth) :

    numCorrect = 0
    print(predictions,truth)
    for i in range(len(predictions)) :
        if round(predictions[i]) == truth[i] :
            numCorrect = numCorrect + 1 
    
    return (numCorrect /len(predictions)) * 100

def getBest(evaluations) :
    minMAE = 100
    minIndex = -1

    for i in range ( len(evaluations) ):
        if evaluations[i][1] < minMAE : 
            minIndex = i
            minMAE = evaluations[i][1]

    return minIndex



net = NN.NeuralNetwork
#data = DC.DataContainer()

neuralNetMedium = net('relu','sgd','Medium')
neuralNetLarge = net('relu','sgd','Largeprop')
neuralNetSmall = net('relu','sgd','Small')

print("\n\n\n\n\nNN Created.\n")

evaluationsMedium = []
predictionsMedium = []

evaluationsLarge = []
predictionsLarge = []

evaluationsSmall = []
predictionsSmall = []
count = 5

for i in range(count) :
    neuralNetMedium.train()
    neuralNetLarge.train()
    neuralNetSmall.train()

    evaluationsMedium.append( neuralNetMedium.evaluate() )
    evaluationsLarge.append( neuralNetLarge.evaluate() )
    evaluationsSmall.append( neuralNetSmall.evaluate() )

    predictionsMedium.append(neuralNetMedium.predict() )
    predictionsLarge.append(neuralNetLarge.predict() )
    predictionsSmall.append(neuralNetSmall.predict() )

print("\n\nPerformed ",count," test runs\n")

print("\nAverage of evaluations\n")
#bestIndex = getBest(evaluations)
#bestRun = evaluations[bestIndex]

# #print("Best run")
# print("Loss:",bestRun[0])
# print("Mean absolute error(MAE):",bestRun[1])
# print("Mean squared error (MSE):",bestRun[2])
# print("Accuracy:",calculateAccuracy( predictions[bestIndex],data.getTestingLabels().values ) )
print("Medium")
evaluateNet(evaluationsMedium,predictionsMedium)
print("Large")
evaluateNet(evaluationsLarge,predictionsLarge)
print("Small")
evaluateNet(evaluationsSmall,predictionsSmall)




