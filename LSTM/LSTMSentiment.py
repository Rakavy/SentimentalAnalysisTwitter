import pandas as pnd
import theano
from theano import config
import mmap
import theano.tensor as tensor
import numpy as np
import sympy as sym
import nltk
from collections import OrderedDict
#nltk.download('punkt')

popularWords = 5000

def processString(stro):
    return stro.translate(str.maketrans({key:None for key in '@#'}))

preprocess=np.vectorize(processString)

def readCSV(filePath,max_sample_size):

    csvTwitter=pnd.read_csv(filePath,header=None,encoding='latin-1').values[:max_sample_size,:]



    tSentiments=csvTwitter[:,0]
    textData=preprocess(csvTwitter[:,5])

    return (tSentiments,textData)


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

    miniBatches=[indexLst[i*size:(i+1)*size] for i in range(n//size+1)]

    return zip(range(len(miniBatches)), miniBatches)



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
    counts = list(dictWords.values())
    keys = list(dictWords.keys())

    sorted_idx = np.argsort(counts)[::-1]

    indexedWords = dict()

    for idx, ss in enumerate(sorted_idx):
        indexedWords[keys[ss]] = idx+2 if idx<4997 else 0
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
    params['Wemb'] = (0.01 * randn).astype(config.floatX)

    params = init_param_lstm(dimension,params)

    # classifier
    params['U'] = 0.01 * np.random.randn(dimension, ydim).astype(config.floatX)
    params['b'] = np.zeros((ydim,)).astype(config.floatX)

    return params

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def init_param_lstm(dimension, params):
    """
    Initialize the LSTM parameter:

    """

    for gate in ['i','c','f','o']:

        params['_'.join(['l','W',gate])]=ortho_weight(dimension)
        params['_'.join(['l','U',gate])]=ortho_weight(dimension)
        params['_'.join(['l','b',gate])]=np.zeros((dimension,)).astype(config.floatX)

    params['l_V']=ortho_weight(dimension)

    return params

def lstmLayer(params, input_state, dimension, mask):

    #Number of time steps (number of word or character vectors interacting with each other)
    nsteps = input_state.shape[0]

    #Number of samples in the batch
    num_samples=input_state.shape[1] if input_state.ndim==3 else 1

    def step(m_, x_, h_, c_):
        i=tensor.nnet.sigmoid(tensor.dot(x_,params['l_W_i'])+tensor.dot(h_,params['l_U_i'])+params['l_b_i'])

        cnd=tensor.tanh(tensor.dot(x_,params['l_W_c'])+tensor.dot(h_,params['l_U_c'])+params['l_b_c'])

        f=tensor.nnet.sigmoid(tensor.dot(x_,params['l_W_f'])+tensor.dot(h_,params['l_U_f'])+params['l_b_f'])

        c=i*cnd+f*c_

        o=tensor.nnet.sigmoid(tensor.dot(x_,params['l_W_o'])+tensor.dot(h_,params['l_U_o'])+tensor.dot(c,params['l_V'])+params['l_b_o'])

        h=o*tensor.tanh(c)

        c=m_[:,None]*c+(1.-m_)[:,None]*c_

        h=m_[:,None]*h+(1-m_)[:,None]*h_

        return h,c


    rval, updates = theano.scan(step,
                                sequences=[mask, input_state],
                                outputs_info=[tensor.alloc(np.asarray(0., dtype=config.floatX),
                                                           num_samples,
                                                           dimension),
                                              tensor.alloc(np.asarray(0., dtype=config.floatX),
                                                           num_samples,
                                                           dimension)],
                                name='l_layers',
                                n_steps=nsteps)
    return rval[0]

SEED=11;


def buildModel(params, dimension):
    trng = np.random.seed(SEED)

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
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

#Gives the prediction error given input x, target y and lists of mini-batches idxLists
def pred_error(pred_fct, prepare, data, y, idxLists):


    validPred=0
    for _,idxList in idxLists:
        x, mask =prepare([data[t] for t in idxList])


        predictions=pred_fct(x,mask)
        targets=np.array(y)[idxList]
        validPred+=sum([1 for x in zip(predictions, targets) if x[0]==x[1]])

    validPred=1.0-np.asarray(validPred, dtype=config.floatX)/float(len(data))

    return validPred


def prepareData(batch):

    maxLength=max(map(len,batch))

    sequences=[tweet+[0]*(maxLength-len(tweet)) for tweet in batch]

    masks=[[1]*len(tweet)+[0]*(maxLength-len(tweet)) for tweet in batch]

    seqs = np.zeros((maxLength,len(batch))).astype('int64')
    masks = np.zeros((maxLength, len(batch))).astype(theano.config.floatX)
    for idx, s in enumerate(batch):
        seqs[:len(batch[idx]), idx] = s
        masks[:len(batch[idx]), idx] = 1.


    return seqs,masks

def reloadModel(path, n_words, dim, ydim):
    params=init_params(n_words, dim_proj, yDim)

    load_params(path,params)

    return params

def loadPredict_f(path, n_words, dim, ydim):
    params=init_params(n_words, dim_proj, yDim)

    load_params(path,params)

    x, mask, y, predictF, loss=buildModel(sharedParams, dim_proj)

    return predictF

def testExample(predictF,tweet):
    return predictF(tweet)

def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def trainNetwork(
        txtData,
        target,
        valid_portion=0.05, #proportion of data used for validation
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        lp_const=0.,  # Value of Lp Regularization parameters
        l_norm=2, # Chosen norm for Regularization
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

    n_validSet=int(np.round(float(len(txtData))*valid_portion))

    train_setx=txtData[n_validSet:]
    train_sety=target[n_validSet:]
    valid_setx=txtData[:n_validSet]
    valid_sety=target[:n_validSet]

    kf = getMiniBatches(len(train_setx), batch_size)


    for _,idxList in kf:
        x, mask =prepareData([train_setx[t] for t in idxList])

    yDim=max(train_sety)+1 #How many categories there are, 5 in our case (0-4), but technically only 3 (0,2,4)



    params=init_params(n_words, dim_proj, yDim)

    if reload_model:
        load_params('lstm_model.npz', params)

    sharedParams = OrderedDict()
    for k in params.keys():
        sharedParams[k] = theano.shared(params[k], name=k)

    x, mask, y, predictF, loss=buildModel(sharedParams, dim_proj)

    #Computing L2 Regularization

    if lp_const>0:
        lp_const=theano.shared(np.asarray(lp_const, dtype=config.floatX), name="lp_const")
        reg_val=lp_const*(sharedParams["U"]**l_norm).sum()
        loss+=reg_val


    f_loss = theano.function([x, mask, y], loss, name='f_loss')

    grads = tensor.grad(loss, wrt=list(sharedParams.values()))
    gradient = theano.function([x, mask, y], grads, name='gradient')

    updates=[(value, value-lrate*gr) for value,gr in zip(list(sharedParams.values()),grads)]

    sgd=theano.function([x,mask,y],loss,updates=updates)

    print('Optimization')

    kf_valid = getMiniBatches(len(valid_setx), valid_batch_size)
    kf_test = getMiniBatches(len(txtData), valid_batch_size)

    print("%d train examples" % len(train_setx))
    print("%d valid examples" % len(valid_sety))
    print("%d test examples" % len(txtData))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train_setx) // batch_size
    if saveFreq == -1:
        saveFreq = len(train_setx) // batch_size

    uidx = 0  # the number of update done
    early_stp = False  # early stop
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = getMiniBatches(len(train_setx),batch_size,shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                y = [train_sety[t] for t in train_index]
                x = [train_setx[t] for t in train_index]

                # Get the data in np.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                y=np.array(y).astype('int64')
                x, mask = prepareData(x)
                n_samples += x.shape[1]

                cur_loss = sgd(x, mask, y)

                if np.isnan(cur_loss) or np.isinf(cur_loss):
                    print('bad loss detected: ', cur_loss)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'loss ', cur_loss)

                if saveto and np.mod(uidx, saveFreq) == 0:

                    params=best_p if best_p is not None else sharedParams

                    np.savez(saveto, history_errs=history_errs, **params)

                if np.mod(uidx, validFreq) == 0:
                    train_err = pred_error(predictF, prepareData, train_setx, train_sety, kf)
                    valid_err = pred_error(predictF, prepareData, valid_setx, valid_sety, kf_valid)

                    test_err = pred_error(predictF, prepareData, txtData, targets, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= np.array(history_errs)[:, 0].min()):

                        best_p = params
                        bad_counter = 0

                    print('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            early_stp = True
                            break

            print('Seen %d samples' % n_samples)

            if early_stp:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    if best_p is  None:
        best_p = sharedParams

    kf_train_sorted = getMiniBatches(len(train_setx), batch_size)
    train_err = pred_error(predictF, prepareData, train_setx, train_sety, kf_train_sorted)
    valid_err = pred_error(predictF, prepareData, valid_setx, valid_sety, kf_valid)
    test_err = pred_error(predictF, prepareData, txtData, targets, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        np.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (eidx + 1))

    return train_err, valid_err, test_err






#print(freq)


(targets, data)=readCSV('../Resources/Sentiment140/trainingdata2.csv',5000)
targets=list(map(lambda x: x//2,targets))
tokens=tokenzieSentence(data)
freq=wordFrequency(tokens)
indexWordList = indexWords(freq)
indexTweets = replaceWordWithIndex(tokens, indexWordList)

trainNetwork(indexTweets,targets)



pred_f=loadPredict_f('lstm_model.npz',5000,128,3)

example="I have a Iran addiction. Thank you for pointing that out."

example_format=tokenzieSentence([example])
test=replaceWordWithIndex(example_format,indexWordList)

print(testExample(pred_f, test[0]))
