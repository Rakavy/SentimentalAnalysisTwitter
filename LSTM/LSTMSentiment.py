import pandas as pnd
import numpy as np
import sympy as sym
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

def predict():
    return None

def buildModel(params, dimension, ):
    trng = np.random.seed(SEED)


    x = np.matrix(dtype='int64')
    mask = np.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost




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

    x=np.matrix() #The data will sit here
    mask=np.matrix() #1 is word at that location, 0 is sequence is already terminated(I think)
    y=np.ndarray() #The target values or the output







#print(freq)

print indexTweets
