import pandas as pnd
import numpy as np
import nltk

def processString(stro):
    return stro.translate(None, '@#')

preprocess=np.vectorize(processString)

def readCSV(filePath):

    csvTwitter=pnd.read_csv(filePath,header=None).values

    tSentiments=csvTwitter[:,0]

    textData=preprocess(csvTwitter[:,5])

    #textData=csvTwitter[:,5].str.translate(None, '@a')

    print(textData)

    return (tSentiments, textData)




targets, data=readCSV('Resources/Sentiment140/TestingData.csv')
