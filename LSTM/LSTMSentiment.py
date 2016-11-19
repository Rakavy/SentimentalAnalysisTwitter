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
        tokenizedPhrase.append([word[0] for word in nltk.pos_tag(nltk.word_tokenize(phrase)) if word[1] not in ['CC','IN']])
    return tokenizedPhrase


def getMiniBatches(n, size, shuffle=False):
    indexLst=np.arange(n, dtype='int32')

    if shuffle:
        np.random.shuffle(indexLst)

    miniBatches=[indexLst[i*size:(i+1)*size] for i in range(n/size+1)]

    return zip(range(len(miniBatches)), miniBatches)

print(getMiniBatches(40,10))
print(getMiniBatches(50,6,True))

(targets, data)=readCSV('../Resources/Sentiment140/TestingData.csv')
tokens=tokenzieSentence(data)

#print tokens
