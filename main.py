import NeuralNetwork as NN 
import DataContainer as DC
import numpy as np

def calculateAccuracy(predictions,truth) :

    numCorrect = 0
    print(predictions,truth)
    for i in range(len(predictions)) :
        if round(predictions[i]) == truth[i] :
            numCorrect = numCorrect + 1 
    
    return (numCorrect /len(predictions)) * 100

def averageResultValues( arr ) :

    test_loss = 0
    test_mae = 0 
    test_mse = 0
    count = len(arr) 

    for i in range(count) :
        test_loss = test_loss + arr[i][0]
        test_mae = test_mae + arr[i][1]
        test_mse = test_mse + arr[i][2]

    test_mae = test_mae / count
    test_mse = test_mse / count
    test_loss = test_loss / count

    return (test_loss, test_mae, test_mse)



net = NN.NeuralNetwork
data = DC.DataContainer()

neuralNet = net()
print("\n\n\n\n\n\n\nNN Created.\n")

evaluations = []
predictions = []
count = 1

for i in range(count) :
    neuralNet.train()
    print("\nNN Trained.\n")
    evaluations.append( neuralNet.evaluate() )
    print("\nNN Evaluated.\n")
    predictions.append( calculateAccuracy( neuralNet.predict(), data.getTestingLabels().values ) )
    print("\nNN Predicted.\n")

print("\n\nPerformed ",count," test runs\n")
#print(evaluations)

print("\n\nAverage of evaluations\n")
avgEvaluation = averageResultValues(evaluations)
print("Loss:",avgEvaluation[0])
print("Mean absolute error(MAE):",avgEvaluation[1])
print("Mean squared error (MSE):",avgEvaluation[2])
print("Accuracy:",predictions)

