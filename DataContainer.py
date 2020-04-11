import numpy as np
import pandas as pd 
import os
from os.path import dirname, abspath

DATAFILE = "winequality_data"
FILENAMERED = "winequality-red.csv"
FILENAMEWHITE = "winequality-white.csv"

ROOTDIRECTORY = dirname(abspath(__file__)) 
FILEDIRECTORY = ROOTDIRECTORY + "\\" +DATAFILE + "\\"

class DataContainer :
  def __init__(self):
    self.redWineList = pd.read_csv(FILEDIRECTORY + FILENAMERED,sep = ";")
    self.whiteWineList =  pd.read_csv(FILEDIRECTORY + FILENAMEWHITE,sep = ";")
    self.columnNames = self.whiteWineList.columns

    if self.whiteWineList.columns.values.all() != self.redWineList.columns.values.all() :
      print("ERR: WINE LIST HEADERS DO NOT MATCH")

    self.redWineListLabels =  self.redWineList["quality"]
    self.whiteWineListLabels = self.whiteWineList["quality"]
    
    del self.redWineList["quality"]
    del self.whiteWineList["quality"]

  
  def getTrainingData(self) :
    #train_dataset = dataset.sample(frac=0.8,random_state=0)
    #test_dataset = dataset.drop(train_dataset.index)
    return self.redWineList.values

  def getTrainingLabels(self) :
    return self.redWineListLabels

  def getTestingData(self) :
    return self.whiteWineList.values

  def getTestingLabels(self) :
    return self.whiteWineListLabels



data = DataContainer()