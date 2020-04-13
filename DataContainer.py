import numpy as np
import pandas as pd 
import os
from os.path import dirname, abspath

import matplotlib.pyplot as plt

import seaborn as sns
import scipy
from scipy import stats

DATAFILE = "winequality_data"
FILENAMERED = "winequality-red.csv"
FILENAMEWHITE = "winequality-white.csv"

ROOTDIRECTORY = dirname(abspath(__file__)) 
FILEDIRECTORY = ROOTDIRECTORY + "\\" +DATAFILE + "\\"

class DataContainer :
  def __init__(self):

    redWineList = pd.read_csv(FILEDIRECTORY + FILENAMERED,sep = ";")
    whiteWineList =  pd.read_csv(FILEDIRECTORY + FILENAMEWHITE,sep = ";")

    self.columnNames = whiteWineList.columns

    if whiteWineList.columns.values.all() != redWineList.columns.values.all() :
      print("ERR: WINE LIST HEADERS DO NOT MATCH")


    (self.testData, self.trainData) = self.splitDataSet(redWineList,whiteWineList)

    self.trainData = self.removeOutliers(self.trainData)
  
    self.testData = self.normaliseData(self.testData)
    self.trainData = self.normaliseData(self.trainData)

    self.testLabels =  self.testData["quality"]
    self.trainLabels =  self.trainData["quality"]


    self.testData.pop("quality")
    self.trainData.pop("quality")
  
    #self.printDataSetStats(self.testData)
    #self.printDataSetStats(self.trainData)

   
    #sns.pairplot(self.trainData[["fixed acidity","volatile acidity","citric acid","residual sugar"]], diag_kind="kde") 
    #plt.show()
    


  def splitDataSet(self,dataSetOne,dataSetTwo) :
    redWineTrainingSet = dataSetOne.sample(frac=0.7,random_state=0)
    redWineTestingSet = dataSetOne.drop(redWineTrainingSet.index)

    whiteWineTrainingSet = dataSetTwo.sample(frac=0.7,random_state=0)
    whiteWineTestingSet = dataSetTwo.drop(whiteWineTrainingSet.index)

    trainingSet = pd.concat([redWineTrainingSet,whiteWineTrainingSet],sort=False)
    testingSet = pd.concat([redWineTestingSet,whiteWineTestingSet],sort=False)

    return (testingSet,trainingSet)


  def normaliseData(self,dataSet) :

    dataCopy = dataSet.describe().transpose()

    normedData = self.norm(dataSet,dataCopy)

    normedData['quality'] = dataSet['quality']

    return normedData

  def norm(self,x,dataset) :
    return (x - dataset['mean']) /  dataset['std']

  def getColumnNames(self) :
    return self.columnNames

  def getTrainingData(self) :
    return self.trainData

  def getTrainingLabels(self) :
    return self.trainLabels


  def getTestingData(self) :
    return self.testData

  def getTestingLabels(self) :
    return self.testLabels


  def printDataSetStats(self,dataset) :
    print(dataset.describe().transpose())

  def scaleDataSet(self,dataSet) :
    dataCopy = dataSet.describe().transpose()

    scaledData = self.scaleInput(dataSet,dataCopy)

    return scaledData

  def scaleInput(self,x,dataset) :
    A = dataset['min']
    B = dataset['max']
    a = -1 #tanh
    b = 1 #tanh
    x = ( 
      ( (x - A)
      /
      (B - A) ) * (b - a) + a
      )
    return x

  def removeOutliers(self,dataSet) :
    #remove ouliers but donot take quality into account
    noOutlierList = dataSet[(np.abs(stats.zscore(dataSet.loc[:, self.trainData.columns != 'quality'])) < 3).all(axis=1)]
    return noOutlierList




data = DataContainer()


##sns.pairplot(data.getTrainingData()[["chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]], diag_kind="kde") 
#plt.show()
#sns.pairplot(data.getNormalisedTrainingData()[["chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]], diag_kind="kde") 
#plt.show()

