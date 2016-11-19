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

    return tokenizedPhrase

#Every step, a subset of the training data is used. getMiniBatches return a list of sample indices to use
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

def init_params(num_words, dimension, ydim):

    params = OrderedDict()
    # embedding
    randn = np.random.rand(num_words, dimension)
    params['Wemb'] = (0.01 * randn).astype(np.float16)

    params = init_param_lstm(dimension,params)

    # classifier
    params['U'] = 0.01 * np.random.randn(dimension, ydim).astype(np.float16)
    params['b'] = np.zeros((ydim,)).astype(np.float16)

    return params

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(np.float16)

def init_param_lstm(dimension, params):
    """
    Initialize the LSTM parameter:

    """

    #Weights for current value input
    W = np.concatenate([ortho_weight(dimension) for i in range(4)], axis=1)

    #Weights for memory value
    U = np.concatenate([ortho_weight(dimension) for i in range(4)], axis=1)

    b = np.zeros((4*dimension,)).astype(np.float16)#array of 0s for biases

    params['l_W'] = W
    params['l_U'] = U
    params['l_b'] = b

    return params

"""def lstm_layer(state_below, options):
    nsteps = state_below.shape[0]

    num_samples=1 if state_below.ndim !=3 else state_below.shape[1]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m, x, h, c):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]"""

(targets, data)=readCSV('../Resources/Sentiment140/TestingData.csv')
tokens=tokenzieSentence(data)
freq=wordFrequency(tokens)
indexWords = indexWords(freq)
indexTweets = replaceWordWithIndex(tokens, indexWords)

init_param_lstm(5, {})



#print(freq)

print indexTweets
