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


class LSTMNetwork:

    def __init__(self,trainPath,reloadModel=False,encode,seed):
        self.trainPath=trainPath
        self.encode=encode
        np.random.seed(seed)
        self.reloadModel=reloadModel
        self.saveParams='lstm_model.npz',

    def setNetwork()

    def processString(self,stro):
        return stro.translate(str.maketrans({key:None for key in '@#'}))

    self.preprocess=np.vectorize(processString)

    def readCSV(self,max_sample_size):

        csvTwitter=pnd.read_csv(self.trainPath,header=None,encoding=self.encode)

        csvTwitter=csvTwitter.values[::len(csvTwitter.index)//max_sample_size,:]

        tSentiments=csvTwitter[:,0]
        textData=preprocess(csvTwitter[:,5])

        return (tSentiments,textData)

    def prepareAllData(self):
        (targets, data)=readCSV(self,100)

        targets=list(map(lambda x: x//2,targets))
        tokens=self.tokenizeSentence(data)

        freq=self.wordFrequency(tokens)
        indexWordList = self.indexWords(freq)
        indexTweets = self.replaceWordWithIndex(tokens, indexWordList)

        self.tweetData=indexTweets
        self.targets=targets


    def tokenizeSentence(self,tweets):
        tokenizedPhrase = []

        for phrase in tweets:
            tokenizedPhrase.append([word[0] for word in nltk.pos_tag(nltk.word_tokenize(phrase)) if word[1] not in ['CC','IN']])

        return tokenizedPhrase


    #Every step, a subset of the training data is used. getMiniBatches return a list of sample indices to use
    def getMiniBatches(self,n, size, shuffle=False):
        indexLst=np.arange(n, dtype='int32')

        if shuffle:
            np.random.shuffle(indexLst)

        miniBatches=[indexLst[i*size:(i+1)*size] for i in range(n//size+1)]

        return list(zip(range(len(miniBatches)), miniBatches))

    def wordFrequency(self,phraseArray):
        wordfreq = dict()

        for phrase in phraseArray:
            for word in phrase:
                if word not in wordfreq:
                    wordfreq[word] = 1
                else:
                    wordfreq[word] += 1

        return wordfreq

    def indexWords(self,dictWords):
        counts = list(dictWords.values())
        keys = list(dictWords.keys())

        sorted_idx = np.argsort(counts)[::-1]

        indexedWords = dict()

        for idx, ss in enumerate(sorted_idx):
            indexedWords[keys[ss]] = idx+2 if idx<4997 else 0
            #print keys[ss], (idx+2)

        #print np.sum(counts), ' total words ', len(keys), ' unique words'

        return indexedWords

    def replaceWordWithIndex(self,tokenizedArray, indexWords):
        indexedTweets = []

        for phrase in tokenizedArray:
            tweet = []
            for word in phrase:
                #print word, str(indexWords[word])
                try:
                    tweet.append(indexWords[word])
                except KeyError:
                    pass

            indexedTweets.append(tweet)

        return indexedTweets

    def initialize(self,num_words, dimension, ydim):

        self.params = OrderedDict()
        # embedding
        randn = np.random.rand(num_words, dimension)
        self.params['embedding'] = (0.01 * randn).astype(config.floatX)

        self.initLSTM(dimension)

        # classifier
        self.params['U'] = 0.03 * np.random.randn(dimension, ydim).astype(config.floatX)
        self.params['b'] = np.zeros((ydim,)).astype(config.floatX)

        return self.params

    def ortho_weight(self,ndim):
        A = np.random.randn(ndim, ndim)
        x, y, z = np.linalg.svd(A)
        return x.astype(config.floatX)

    def initLSTM(self,dimension):
        """
        Initialize the LSTM parameter:

        """

        for gate in ['i','c','f','o']:

            self.params['_'.join(['l','W',gate])]=self.ortho_weight(dimension)
            self.params['_'.join(['l','U',gate])]=self.ortho_weight(dimension)
            self.params['_'.join(['l','b',gate])]=np.zeros((dimension,)).astype(config.floatX)

        self.params['l_V']=self.ortho_weight(dimension)

        return params

    def lstmLayer(self,params, input_state, dimension, mask):

        #Number of time steps (number of word or character vectors interacting with each other)
        nsteps = input_state.shape[0]

        #Number of samples in the batch
        num_samples=input_state.shape[1] if input_state.ndim==3 else 1

        def lstmStep(self,m_, x_, h_, c_):
            i=tensor.nnet.sigmoid(tensor.dot(x_,params['l_W_i'])+tensor.dot(h_,params['l_U_i'])+params['l_b_i'])

            cnd=tensor.tanh(tensor.dot(x_,params['l_W_c'])+tensor.dot(h_,params['l_U_c'])+params['l_b_c'])

            f=tensor.nnet.sigmoid(tensor.dot(x_,params['l_W_f'])+tensor.dot(h_,params['l_U_f'])+params['l_b_f'])

            c=i*cnd+f*c_

            o=tensor.nnet.sigmoid(tensor.dot(x_,params['l_W_o'])+tensor.dot(h_,params['l_U_o'])+tensor.dot(c,params['l_V'])+params['l_b_o'])

            h=o*tensor.tanh(c)

            c=m_[:,None]*c+(1.-m_)[:,None]*c_

            h=m_[:,None]*h+(1-m_)[:,None]*h_

            return h,c


        rval, updates = theano.scan(lstmStep,
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




    def createNetwork(self,params, dimension):

        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype=config.floatX)
        y = tensor.vector('y', dtype='int64')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        #Pass input sequences through embedding layer
        emb = params['embedding'][x.flatten()].reshape([n_timesteps, n_samples, dimension])

        #Pass input sequences through LSTM layer
        output = lstmLayer(params, emb, dimension,mask)

        #Get average prediction for output layer across input sequences
        output = (output * mask[:, :, None]).sum(axis=0)

        output = output / mask.sum(axis=0)[:, None]

        #Create symbolic matrix of softmax computation of output layer
        predict = tensor.nnet.softmax(tensor.dot(output, params['U']) + params['b'])

        predictFun = theano.function([x, mask], predict.argmax(axis=1), name='predictFun')

        off = 1e-8

        cost = -tensor.log(predict[tensor.arange(n_samples), y] + off).mean()

        return x, mask, y, predictFun, cost

    #Gives the prediction error given input x, target y and lists of mini-batches idxLists
    def getErrorRate(self,pred_fct, data, y, idxLists, pred_prob=None):

        validPred=0
        for _,idxList in idxLists:
            x, mask =self.prepareData([data[t] for t in idxList])

            predictions=pred_fct(x,mask)
            if(pred_prob):
                print(pred_prob(x,mask))
            #print(predictions)
            targets=np.array(y)[idxList]
            validPred+=sum([1 for x in zip(predictions, targets) if x[0]==int(x[1])])

        validPred=1.0-np.asarray(validPred, dtype=config.floatX)/float(len(data))

        return validPred

    def reassign(self,params, tparams):
        for kk, vv in params.items():
            tparams[kk].set_value(vv)

        return params


    def self.unshare(self,zipped):

        new_params = OrderedDict()
        for kk, vv in zipped.items():
            new_params[kk] = vv.get_value()
        return new_params

    def prepareData(self,batch):

        maxLength=max(map(len,batch))

        sequences=[tweet+[0]*(maxLength-len(tweet)) for tweet in batch]

        masks=[[1]*len(tweet)+[0]*(maxLength-len(tweet)) for tweet in batch]

        seqs = np.zeros((maxLength,len(batch))).astype('int64')
        masks = np.zeros((maxLength, len(batch))).astype(theano.config.floatX)
        for idx, s in enumerate(batch):
            seqs[:len(s), idx] = s
            masks[:len(s), idx] = 1.

        return seqs,masks

    def reloadModel(self,path, n_words, dim, ydim):
        initialize(n_words, dim, ydim)

        load_params()

        return params

    def getSharedParams(self):
        sharedParams = OrderedDict()
        for k in params.keys():
            sharedParams[k] = theano.shared(params[k], name=k)


    def loadPredict_f(self, n_words, dim, ydim):
        params=initialize(n_words, dim, ydim)

        load_params()


        x, mask, y, predictF, loss, predictProb=self.createNetwork(sharedParams, dim_proj)


        return predictF

    def testExample(self,predictF,tweet):
        return predictF(tweet)

    def load_params(self):
        pp = np.load(self.saveParams)
        for kk, vv in self.params.items():
            if kk not in pp:
                raise Warning('%s is not in the archive' % kk)
            self.params[kk] = pp[kk]

        return self.params

    def trainNetwork(
        valid_portion=0.05, #proportion of data used for validation
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        lp_const=0.,  # Value of Lp Regularization parameters
        l_norm=2, # Chosen norm for Regularization
        max_epochs=500,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        lrate=0.05,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=5000,  # Vocabulary size
        validFreq=370,  # Compute the validation error after this number of update.
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        reload_model=None,  # Path to a saved model we want to start from.
    ):

        #Split the data between training set and validation set

        n_validSet=int(np.round(float(len(self.tweetData))*valid_portion))

        train_setx=self.tweetData[n_validSet:]
        train_sety=self.targets[n_validSet:]
        valid_setx=self.tweetData[:n_validSet]
        valid_sety=self.targets[:n_validSet]

        kf = self.getMiniBatches(len(train_setx), batch_size, shuffle=True)

        for _,idxList in kf:
            x, mask =self.prepareData([train_setx[t] for t in idxList])

        print(x)
        print(mask)



        yDim=max(train_sety)+1 #How many categories there are, 5 in our case (0-4), but technically only 3 (0,2,4)

        params=initialize(n_words, dim_proj, yDim)

        if reload_model:
            load_params('lstm_model.npz', params)

        sharedParams = OrderedDict()
        for k in params.keys():
            sharedParams[k] = theano.shared(params[k], name=k)

        x, mask, y, predictF, loss=self.createNetwork(sharedParams, dim_proj)

        #Computing L2 Regularization

        #if lp_const>0:
        #    lp_const=theano.shared(np.asarray(lp_const, dtype=config.floatX), name="lp_const")
        #    reg_val=lp_const*(sharedParams["U"]**l_norm).sum()
        #    loss+=reg_val


        f_loss = theano.function([x, mask, y], loss, name='f_loss')

        grads = tensor.grad(loss, wrt=list(sharedParams.values()))
        gradient = theano.function([x, mask, y], grads, name='gradient')

        lr=theano.shared(lrate, name='lr')
        updates=[(value, value-lr*gr) for value,gr in list(zip(list(sharedParams.values()),grads))]

        sgd=theano.function([x,mask,y],loss,updates=updates)

        print('Optimization')

        kf_valid = self.getMiniBatches(len(valid_setx), valid_batch_size, shuffle=True)
        kf_test = self.getMiniBatches(len(self.tweetData), valid_batch_size, shuffle=True)

        print("%d train examples" % len(train_setx))
        print("%d valid examples" % len(valid_sety))
        print("%d test examples" % len(self.tweetData))

        history_errs = []
        best_p = None
        bad_count = 0

        if validFreq == -1:
            validFreq = len(train_setx) // batch_size
        if saveFreq == -1:
            saveFreq = len(train_setx) // batch_size

        uidx = 0  # the number of update done
        stop = False  # early stop

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
                x, mask = self.prepareData(x)
                n_samples += x.shape[1]

                cur_loss = sgd(x, mask, y)

                if np.isnan(cur_loss) or np.isinf(cur_loss):
                    print('bad loss detected: ', cur_loss)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'loss ', cur_loss)

                if saveto and np.mod(uidx, saveFreq) == 0:

                    params=best_p if best_p is not None else self.unshare(sharedParams)

                    np.savez(saveto, history_errs=history_errs, **params)

                if np.mod(uidx, validFreq) == 0:
                    train_err = self.getErrorRate(predictF, train_setx, train_sety, kf)
                    valid_err = self.getErrorRate(predictF, valid_setx, valid_sety, kf_valid)

                    test_err = self.getErrorRate(predictF, self.tweetData, self.targetss, kf_test)

                    print(sharedParams['b'].get_value())

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= np.array(history_errs)[:, 0].min()):

                        best_p = self.unshare(sharedParams)
                        bad_counter = 0

                    print('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('stop...')
                            stop = True
                            break

            if stop:
                break


        if best_p is not None:
            best_p = self.reassign(best_p,sharedParams)
        else:
            best_p=self.unshare(sharedParams)

        kf_train_sorted = self.getMiniBatches(len(train_setx), batch_size)
        train_err = self.getErrorRate(predictF, train_setx, train_sety, kf_train_sorted)
        valid_err = self.getErrorRate(predictF, valid_setx, valid_sety, kf_valid)
        test_err = self.getErrorRate(predictF, self.tweetData, self.targetss, kf_test)

        print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )

        if saveParams:
            np.savez(self.saveParams, train_err=train_err, valid_err=valid_err, test_err=test_err, history_errs=history_errs, **best_p)

        return train_err, valid_err, test_err


    def analyze(self,text):
        (targets, data)=self.readCSV('../Resources/Sentiment140/trainingdata2.csv',500)
        targets=list(map(lambda x: x//2,targets))
        tokens=self.tokenizeSentence(data)
        freq=self.wordFrequency(tokens)
        indexWordList = self.indexWords(freq)
        indexTweets = self.replaceWordWithIndex(tokens, indexWordList)

        #trainNetwork(indexTweets,targets)
        pred_f=self.loadPredict_f(5000,128,3)
        t = self.processString(text)
        example_format=self.tokenizeSentence([t])
        test = self.replaceWordWithIndex(example_format,indexWordList)

        return self.testExample(pred_f, test[0])
