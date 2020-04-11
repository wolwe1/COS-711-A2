import numpy as np
import pandas as pd 
import os
from os.path import dirname, abspath

import matplotlib.pyplot as plt

import seaborn as sns

DATAFILE = "winequality_data"
FILENAMERED = "winequality-red.csv"
FILENAMEWHITE = "winequality-white.csv"

ROOTDIRECTORY = dirname(abspath(__file__)) 
FILEDIRECTORY = ROOTDIRECTORY + "\\" +DATAFILE + "\\"

class DataContainer :
  def __init__(self):
    self.normedRedData = []
    self.normedWhiteData = []

    self.redWineList = pd.read_csv(FILEDIRECTORY + FILENAMERED,sep = ";")
    self.whiteWineList =  pd.read_csv(FILEDIRECTORY + FILENAMEWHITE,sep = ";")
    self.columnNames = self.whiteWineList.columns

    if self.whiteWineList.columns.values.all() != self.redWineList.columns.values.all() :
      print("ERR: WINE LIST HEADERS DO NOT MATCH")

    self.redWineListLabels =  self.redWineList["quality"]
    self.whiteWineListLabels = self.whiteWineList["quality"]
    
    self.redWineList.pop("quality")

    self.whiteWineList.pop("quality")

    self.printDataSetStats(self.redWineList)
    self.normaliseData()
    self.printDataSetStats(self.normedRedData)
    #train_dataset = dataset.sample(frac=0.8,random_state=0)
    #test_dataset = dataset.drop(train_dataset.index)
    #sns.pairplot(self.redWineList[["fixed acidity","volatile acidity","citric acid","residual sugar"]], diag_kind="kde") 
    #plt.show()
    

  def normaliseData(self) :
    self.printDataSetStats(self.redWineList)

    redCopy = self.redWineList.describe().transpose()
    whiteCopy = self.whiteWineList.describe().transpose()


    self.normedRedData = self.norm(self.redWineList,redCopy)
    self.normedWhiteData = self.norm(whiteCopy,whiteCopy)

  def getColumnNames(self) :
    return self.columnNames

  def getTrainingData(self) :
    return self.redWineList

  def getNormalisedTrainingData(self) :
    return self.normedRedData

  def getTrainingLabels(self) :
    return self.redWineListLabels

  def getTestingData(self) :
    return self.whiteWineList

  def getNormalisedTestingData(self) :
    return self.normedWhiteData

  def getTestingLabels(self) :
    return self.whiteWineListLabels

  def norm(self,x,dataset) :
    return (x - dataset['mean']) /  dataset['std']

  def printDataSetStats(self,dataset) :
    print(dataset.describe().transpose())



#data = DataContainer()

##sns.pairplot(data.getTrainingData()[["chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]], diag_kind="kde") 
#plt.show()
#sns.pairplot(data.getNormalisedTrainingData()[["chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]], diag_kind="kde") 
#plt.show()


