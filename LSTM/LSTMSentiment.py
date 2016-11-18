import pandas as pnd
import numpy as np
import nltk
#nltk.download('punkt')


def processString(stro):
    return stro.translate(None, '@#')

preprocess=np.vectorize(processString)

def readCSV(filePath):

    csvTwitter=pnd.read_csv(filePath,header=None).values

    tSentiments=csvTwitter[:,0]

    textData=preprocess(csvTwitter[:,5])

    return (tSentiments, textData)


def tokenzieSentence(tweets):
    tokenizedPhrase = []
    for phrase in tweets:
        tokenizedPhrase.append(nltk.word_tokenize(phrase))

    return tokenizedPhrase



(targets, data)=readCSV('../Resources/Sentiment140/TestingData.csv')
tokens=tokenzieSentence(data)

#print tokens
