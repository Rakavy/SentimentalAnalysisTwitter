import pandas as pnd
import theano
from theano import config
import theano.tensor as tensor
import numpy as np
import sympy as sym
import nltk
from collections import OrderedDict
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

def lstmLayer(params, input_state, dimension, mask):

    #Number of time steps (number of sequences interacting with each other)
    nsteps = input_state.shape[0]

    #Number of samples in batch
    num_samples = input_state.shape[1]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    #TODO: really understand the following
    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h, params['l_U'])
        preact += x

        activations=[tensor.nnet.sigmoid(preact[:,:,i*dimension:(i+1)*dimension]) for i in range(3)]

        i=activations[0]
        f=activations[1]
        o=activations[2]

        c=tensor.tanh(preact[:,:,3*dimension:4*dimension])

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    input_state = (tensor.dot(input_state, params['l_W')]) +
                   params['l_b')])

    rval, updates = theano.scan(_step,
                                sequences=[mask, input_state],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dimension),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dimension)],
                                name='l_layers',
                                n_steps=nsteps)
    return rval[0]

SEED=11;

def buildModel(params, dimension):
    trng = np.random.seed(SEED)


    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float16')
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    #Pass input sequences through embedding layer
    emb = params['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, dimension])

    #Pass input sequences through LSTM layer
    output = lstmLayer(params, emb, dimension,mask)

    #Get average prediction for output layer across input sequences
    output = (output * mask[:, :, None]).sum(axis=0)

    output = output / mask.sum(axis=0)[:, None]

    #Create symbolic matrix of softmax computation of output layer
    predict = tensor.nnet.softmax(tensor.dot(output, params['U']) + params['b'])

    predictFun = theano.function([x, mask], predict.argmax(axis=1), name='predictFun')

    off = 1e-6

    #TODO: change the following, since not using categories, regression instead
    cost = -tensor.log(predict[tensor.arange(n_samples), y] + off).mean()

    return x, mask, y, predictFun, cost




(targets, data)=readCSV('../Resources/Sentiment140/TestingData.csv')
tokens=tokenzieSentence(data)
freq=wordFrequency(tokens)
indexWords = indexWords(freq)
indexTweets = replaceWordWithIndex(tokens, indexWords)

init_param_lstm(5, {})

def trainNetwork(
        txtData,
        target,
        valid_portion=0.05, #proportion of data used for validation
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=5000,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=5000,  # Vocabulary size
        saveto='lstm_model.npz',  # The best model will be saved there
        validFreq=370,  # Compute the validation error after this number of update.
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        reload_model=None,  # Path to a saved model we want to start from.
):


    #Split the data between training set and validation set

    n_validSet=np.round(float(len(txtData))*valid_portion)

    train_setx=txtData[n_validSet:]
    train_sety=tarfet[n_validSet:]
    valid_setx=txtData[:n_validSet]
    valid_sety=target[:n_validSet]

    yDim=max(train_sety)+1 #How many categories there are, 5 in our case (0-4), but technically only 3 (0,2,4)

    params=init_params(n_words, dim_proj, yDim)

    sharedParams = OrderedDict()
    for k in params.keys():
        sharedParams[k] = theano.shared(params[k], name=k)

    x, mask, y, predictF, cost=buildModel(sharedParams, dim_proj)








#print(freq)

print indexTweets
