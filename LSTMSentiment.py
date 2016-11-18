import pandas as pnd
import numpy as np
import nltk

def processString(str):
    return str.translate(None, '@')

preprocess=np.vectorize(str)

def readCSV(filePath):

    csvTwitter=pnd.read_csv(filePath,header=None).values

    tSentiments=csvTwitter[:,0]

    textData=preprocess(csvTwitter[:,5])

    return (tSentiments, textData)




targets, data=readCSV('Resources/Sentiment140/TestingData.csv')
