import pandas as pnd
import numpy as np
import nltk
#nltk.download('punkt')

popularWords = 5000

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


def wordFrequency(phraseArray):
    wordfreq = dict()

    for phrase in phraseArray:
        for word in phrase:
            if word not in wordfreq:
                wordfreq[word] = 1
            else:
                wordfreq[word] += 1

    return wordfreq

def indexWords(dictWords):
    counts = dictWords.values()
    keys = dictWords.keys()

    sorted_idx = np.argsort(counts)[::-1]
    
    indexedWords = dict()

    for idx, ss in enumerate(sorted_idx):
        indexedWords[keys[ss]] = idx+2  
        #print keys[ss], (idx+2)

    #print np.sum(counts), ' total words ', len(keys), ' unique words'

    return indexedWords

def replaceWordWithIndex(tokenziedArray, indexWords):
    indexedTweets = []

    for phrase in tokenziedArray: 
        tweet = []
        for word in phrase:
            #print word, str(indexWords[word])
            t=word.replace(word,str(indexWords[word]))
            tweet.append(t)

        indexedTweets.append(tweet)

    return indexedTweets

(targets, data)=readCSV('../Resources/Sentiment140/TestingData.csv')
tokens=tokenzieSentence(data)
freq=wordFrequency(tokens)
indexWords = indexWords(freq)
indexTweets = replaceWordWithIndex(tokens, indexWords)

#print(freq)


print indexTweets
