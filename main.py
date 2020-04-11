import NeuralNetwork as NN 
import DataContainer as DC

net = NN.NeuralNetwork
data = DC.DataContainer

neuralNet = net( data())
print("\n\n\n\n\n\n\nNN Created.\n")
neuralNet.train()
print("\nNN Trained.\n")
neuralNet.evaluate()
print("\nNN Evaluated.\n")
neuralNet.predict()
print("\nNN Predicted.\n")
