import NeuralNetwork as NN 
import DataContainer as DC
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as to
from tensorflow_model_optimization.sparsity import keras as sparsity
from kerassurgeon import Surgeon

class PrunableLayer :
    def __init__(self):
        self.nodes = []
        self.layerInModel = -1
        self.layerName = ""
        self.currentLowestIndex = 0
    
    def sort(self):
        self.nodes = np.sort(self.nodes)
        return self.nodes

    def display(self):
        return self.layerName + ":\nLayer in model - " + str(self.layerInModel) + "\nL1 norms" + str(self.nodes) + "\n" 
    
    def getLowest(self):
        self.sort()
        return self.nodes[self.currentLowestIndex]

    def popLowestAndGetIndex(self):
        currentLowest = self.currentLowestIndex
        self.currentLowestIndex = self.currentLowestIndex + 1
        return currentLowest

def findLowestL1(prunableLayers):

    #Find out how many nodes there are
    totalNumNodes = 0
    for i in range( len( prunableLayers)) :
        totalNumNodes += len(prunableLayers[i].nodes)

    #Set how many to remove using rule 5% prune rate
    toBeRemoved = int(totalNumNodes * 0.05)
    lowestValue = float('inf')
    lowestIndex = -1
    removedList = []

    for i in range(toBeRemoved) :
        for x in range( len (prunableLayers)) :
            if prunableLayers[x].getLowest() < lowestValue :
                lowestValue = prunableLayers[x].getLowest() 
                lowestIndex = x
        
        #Remove lowest
        removedList.append( (prunableLayers[lowestIndex].layerInModel,prunableLayers[lowestIndex].popLowestAndGetIndex() ) )
        lowestIndex = -1
        lowestValue = float('inf')
    
    return removedList

def layerInArray(array,target):

    for i in range( len(array)):
        (layer,node) = array[i]

        if target == layer:
            return True
    
    return False

def groupList(removeList):
    groupedList = []
    for i in range( len(removeList)):
        currentNodes = []
        (currentLayer,currentNode) = removeList[i]
        currentNodes.append(currentNode)

        if not layerInArray(groupedList,currentLayer) :

            for x in range(i+1, len(removeList)):
                (comparableLayer,comparableNode) = removeList[x]

                if comparableLayer == currentLayer :
                    currentNodes.append(comparableNode)
        
            groupedList.append( (currentLayer,currentNodes) )
    
    return groupedList


class ModelPruner :
    def __init__(self,model):
        self.originalModel = model
        (self.Original_test_loss, self.Original_test_mae, self.Original_test_mse) = self.originalModel.evaluate()
        self.prunedModel = model

        self.prunableLayers = self.getPrunableLayers()
    
    def getPrunableLayers(self):
        prunableLayersWeights = []

        for i in range( len(self.originalModel.model.layers) ) :
            if i % 2 == 0 :
                prunableLayersWeights.append(self.originalModel.model.layers[i].get_weights())

        summary = []


        for i in range(len (prunableLayersWeights)) :
            prunableLayer = PrunableLayer()
            prunableLayer.layerName = 'layer_{}'.format(i)
            prunableLayer.layerInModel = i*2

            layer = prunableLayersWeights[i]

            nodes = []
            for j in range( len(layer[0]) ) :
                nodes.append(np.sum(abs(layer[0][j]) ))

            prunableLayer.nodes = nodes
            prunableLayer.sort()
            summary.append(prunableLayer)
        
        return summary
    
    def displayL1(self):
        print("\nDisplaying L1 values:\n")

        for i in range( len(self.prunableLayers)) :
            print( self.prunableLayers[i].display())


    def pruneModel(self):
        removeList = findLowestL1(self.prunableLayers)
        #Group multiple nodes from the same layer together
        removeList = groupList(removeList)

        for i in range( len(removeList)):
            surgeon = Surgeon(self.prunedModel.model)
    
            (layerName,nodes) = removeList[i]
            layer = self.prunedModel.model.layers[layerName]
            print("surgeon.add_job('delete_channels',",layer,",channels=",nodes,")")
            surgeon.add_job('delete_channels',layer,channels=nodes)

        self.prunedModel = surgeon.operate()

    def evaluatePrune(self):
        self.prunedModel.compile()

        (test_loss, test_mae, test_mse) = self.prunedModel.evaluate()

        print('Testing set accuracy: {:5.2f} points'.format(test_mae))
        print('\nTest loss:', test_loss)
        lossDiff = test_loss - self.Original_test_loss
        MAELoss = test_mae - self.Original_test_mae
        MSELoss = test_mse - self.Original_test_mse
        print("\nPruned model performance:")
        print("Loss:",test_loss," - Diff ",lossDiff)
        print("MAE:",test_mae," - Diff ",MAELoss)
        print("MSE:",test_mse," - Diff ",MSELoss)

        if MAELoss > 0 and MSELoss > 0 :
            return False #Performing worse 
        else :
            return True
    
    def prune(self):
        print("Original Model")
        runCount = 0
        self.originalModel.model.summary()

        while self.evaluatePrune() == True :
            print("Prune run:",runCount)
            self.pruneModel()
        
        print("\nPruned Model:\n")
        self.prunedModel.model.summary()

