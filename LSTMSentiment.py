import pandas as pnd
import numpy as np
import nltk


def readCSV(filePath):

    csvTwitter=pnd.read_csv(filePath,header=None).values

    tSentiments=csvTwitter[:,0]

    textData=csvTwitter[:,5]

    return (tSentiments, textData)




targets, data=readCSV('Resources/Sentiment140/TestingData.csv')
