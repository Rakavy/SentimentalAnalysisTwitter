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

    def __init__(self,trainPath='../Resources/Sentiment140/TestingData.csv',reloadModel=False,encode='latin-1',seed=47):
        self.trainPath=trainPath
        self.encode=encode
        np.random.seed(seed)
        self.reloadModel=reloadModel
        self.fctPredict=None
        self.saveParams='lstm_model.npz'
        self.train_setx=[]
        self.train_sety=[]
        self.valid_setx=[]
        self.valid_sety=[]

    def setupNetwork(self):
        prepareAllData()
        trainNetwork()

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
        (targets, data)=readCSV(self,500)

        targets=list(map(lambda x: x//2,targets))
        tokens=self.tokenizeSentence(data)

        freq=self.wordFrequency(tokens)
        indexWordList = self.indexWords(freq)
        indexTweets = self.replaceWordWithIndex(tokens, indexWordList)

        self.tweetData=indexTweets
        self.targets=targets


    def getPredictFun():
        if self.fctPredict is None:
            return self.loadPredict_f(5000, 128, 3)
        else
            return self.fctPredict

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


        retVal, updates = theano.scan(lstmStep,
                                    sequences=[mask, input_state],
                                    outputs_info=[tensor.alloc(np.asarray(0., dtype=config.floatX),
                                                               num_samples,
                                                               dimension),
                                                  tensor.alloc(np.asarray(0., dtype=config.floatX),
                                                               num_samples,
                                                               dimension)],
                                    name='l_layers',
                                    n_steps=nsteps)
        return retVal[0]

    def createNetwork(self,params, dimension):

        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype=config.floatX)
        y = tensor.vector('y', dtype='int64')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        #Pass input sequences through embedding layer
        emb = params['embedding'][x.flatten()].reshape([n_timesteps, n_samples, dimension])

        #Pass input sequences through LSTM layer
        out = lstmLayer(params, emb, dimension,mask)

        #Get average prediction for output layer across input sequences
        out = (out * mask[:, :, None]).sum(axis=0)/ mask.sum(axis=0)[:, None]

       #Create symbolic matrix of softmax computation of output layer
        predict = tensor.nnet.softmax(tensor.dot(output, params['U']) + params['b'])

        predictFun = theano.function([x, mask], predict.argmax(axis=1), name='predictFun')

        cost = -tensor.log(predict[tensor.arange(n_samples), y] + 1e-8).mean()

        return x, mask, y, predictFun, cost

    #Gives the prediction error given input x, target y and lists of mini-batches idxLists
    def getErrorRate(self,pred_fct, data, y, idxLists, pred_prob=None):

        validPred=0.0
        for _,idxList in idxLists:
            x, mask =self.prepareData([data[t] for t in idxList])

            predictions=pred_fct(x,mask)
            if(pred_prob):
                print(pred_prob(x,mask))

            targets=np.array(y)[idxList]
            validPred+=sum([1 for x in zip(predictions, targets) if x[0]==int(x[1])])

        validPred=1.0-np.asarray(validPred, dtype=config.floatX)/float(len(data))

        return validPred

    def reassign(self,params, tparams):
        for k, v in params.items():
            tparams[k].set_value(v)

        return params


    def unshare(self,zipped):

        params = OrderedDict()
        for k, v in zipped.items():
            params[kk] = v.get_value()
        return params

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

    def getSharedParams(self):
        sharedParams = OrderedDict()
        for k in self.params.keys():
            sharedParams[k] = theano.shared(params[k], name=k)

        return sharedParams

    def loadPredict_f(self, vocabLen, dim, ydim):
        self.initialize(vocabLen, dim, ydim)

        self.loadParams()

        sharedParams=self.getSharedParams()

        x, mask, y, predictF, loss=self.createNetwork(sharedParams, embDim)

        return predictF

    def testExample(self,predictF,tweet):
        return predictF(tweet)

    def loadParams(self):
        loaded = np.load(self.saveParams)
        for k, v in self.params.items():
            try:
                self.params[k] = pp[k]
            except KeyError:
                pass

        return self.params

    def splitData(self, validPortion):
        n_validSet=int(np.round(float(len(self.tweetData))*validPortion))

        self.train_setx=self.tweetData[n_validSet:]
        self.train_sety=self.targets[n_validSet:]
        self.valid_setx=self.tweetData[:n_validSet]
        self.valid_sety=self.targets[:n_validSet]

        return self.train_setx,self.train_sety,self.valid_setx,self.valid_sety

    def getAllErrors(self,pred_f,trainBatch,validBatch,testBatch,disp=True):
        trainErr = self.getErrorRate(pred_f, self.train_setx, self.train_sety, trainBatch)
        validErr = self.getErrorRate(pred_f, self.valid_setx, self.valid_sety, validBatch)
        testErr = self.getErrorRate(pred_f, self.tweetData, self.targets, testBatch)

        if disp:
            print( 'Train ', trainErr, 'Valid ', validErr, 'Test ', testErr )

        return trainErr,validErr,testErr

    def getSGD(x,mask,y,loss,lrate,params):
        f_loss = theano.function([x, mask, y], loss, name='f_loss')

        grads = tensor.grad(loss, wrt=list(sharedParams.values()))
        gradient = theano.function([x, mask, y], grads, name='gradient')

        lr=theano.shared(lrate, name='lr')
        updates=[(value, value-lr*gr) for value,gr in list(zip(list(sharedParams.values()),grads))]

        sgd=theano.function([x,mask,y],loss,updates=updates)

        return sgd

    def trainNetwork(
        self,
        validSubset=0.05, #proportion of data used for validation
        embDim=128,  # length of embedding vector and number of hidden units
        stopRule=10,  # Stop if no progress in n epochs
        maxEpochs=500,  # The maximum number of epoch to run
        lrate=0.05,  # Learning rate
        savep=True,
        vocabLen=5000,  # Vocabulary size
        batchSz=16,  # The batch size during training.
    ):

        #Split the data between training set and validation set

        train_setx,train_sety,valid_setx,valid_sety=splitData(validSubset)

        trainBatch = self.getMiniBatches(len(train_setx), batchSz, shuffle=True)

        for _,idxList in trainBatch:
            x, mask =self.prepareData([train_setx[t] for t in idxList])

        yDim=max(train_sety)+1 #How many categories there are, 5 in our case (0-4), but technically only 3 (0,2,4)

        initialize(vocabLen, embDim, yDim)

        sharedParams = getSharedParams()

        x, mask, y, predictF, loss=self.createNetwork(sharedParams, embDim)

        sgd=self.getSGD(x, mask, y, loss, sharedParams)

        validBatches = self.getMiniBatches(len(valid_setx), batchSz*4, shuffle=True)
        testBatches = self.getMiniBatches(len(self.tweetData), batchSz*4, shuffle=True)

        print(str(len(train_setx))+" train examples")
        print(str(len(valid_sety))+" valid examples")
        print(str(len(self.tweetData))+"test examples")

        lastErrors = []
        optimalParams = None
        decreaseCnt = 0

        validFreq = len(train_setx) // batchSz
        saveFreq = len(train_setx) // batchSz
        displayFreq=10

        num_updates = 0
        stop = False

        for i in range(maxEpochs):

            trainBatch = getMiniBatches(len(train_setx),batchSz,shuffle=True)

            for _, train_index in trainBatch:
                num_updates += 1

                # Get the samples for this minibatch
                y = [train_sety[t] for t in train_index]
                x = [train_setx[t] for t in train_index]

                y=np.array(y).astype('int64')
                x, mask = self.prepareData(x)

                cur_loss = sgd(x, mask, y)

                if num_updates%displayFreq == 0:
                    print('Epoch ', i, 'Update ', num_updates, 'loss ', cur_loss)

                if savep and num_updates%saveFreq == 0:

                    self.params=optimalParams if optimalParams is not None else self.unshare(sharedParams)

                    np.savez(self.saveParams,**self.params)

                if num_updates%validFreq == 0:

                    train_err,valid_err,test_err=self.getAllErrors(predictF,trainBatch,validBatches,testBatches)

                    lastErrors.append([valid_err, test_err])

                    if (optimalParams is None or
                        valid_err <= np.array(lastErrors)[:, 0].min()):

                        optimalParams = self.unshare(sharedParams)
                        decreaseCnter = 0

                    if (len(lastErrors) > stopRule and
                        valid_err >= np.array(lastErrors)[:-stopRule,
                                                               0].min()):
                        decreaseCnter += 1
                        if decreaseCnter > stopRule:
                            stop = True
                            break

            if stop:
                break

        if optimalParams is not None:
            optimalParams = self.reassign(optimalParams,sharedParams)
        else:
            optimalParams=self.unshare(sharedParams)

        if savep:
            np.savez(self.saveParams,**optimalParams)


    def analyze(self,text):
        (targets, data)=self.readCSV(500)
        targets=list(map(lambda x: x//2,targets))
        tokens=self.tokenizeSentence(data)
        freq=self.wordFrequency(tokens)
        indexWordList = self.indexWords(freq)
        indexTweets = self.replaceWordWithIndex(tokens, indexWordList)

        pred_f=self.getPredictFun()
        t = self.processString(text)
        example_format=self.tokenizeSentence([t])
        test = self.replaceWordWithIndex(example_format,indexWordList)

        return self.testExample(pred_f, test[0])*2
