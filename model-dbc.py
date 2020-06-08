import tensorflow as tf
import os
import numpy as np
import time
import sys
from random import randint, sample
from collections import Counter, OrderedDict
from subprocess import call
import io
import scipy.io
import math
import matplotlib.pyplot as plt

from keras.layers import Input, LSTM, TimeDistributed, Dense, Bidirectional, GRU, Layer
from keras.models import Sequential, load_model
from keras.layers.core import Dropout
from keras import initializers, optimizers, regularizers, constraints
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, Callback
from keras.utils import to_categorical, multi_gpu_model
baseDrive = '/home/istvan/2019_dec_paper/capgMyo/'

NUMBER_OF_FEATURES = 128
recurrent_dropout = 0.5
dropout = 0.5
number_of_classes = 12
cellNeurons = 512
denseNeurons = 512

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.75
	epochs_drop = 20.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def toMultiGpuModel(model):
    try:
        gpu_model = multi_gpu_model(model)
        return gpu_model
    except:
        print("gpu_model error")
        return None

def download(db='b', num=20):
    for i in range(num):
        name = "http://zju-capg.org/myo/data/db"+db+"-preprocessed-" + '{:03d}'.format(i+1) + ".zip"
        call(["wget", name])

def unzip(db='b', num=20):
    for i in range(num):
        name = "db"+db+"-preprocessed-" + '{:03d}'.format(i+1) + ".zip"
        call(["unzip", name, "-d", "unzipped-db"+db])

def playground(db='b'):
    data = scipy.io.loadmat("unzipped-db"+db+"/020-005-008.mat")
    print(type(data))
    print (data.keys())
    # dict_keys(['__header__', '__version__', '__globals__', 'trial', 'data', 'gesture', 'subject'])
    #print (data['trial'])
    print (data['data'].shape)
    #print (data['gesture'])
    #print (data['subject'])

    #arr = np.asarray(data)
    #print(arr)

def convert(db='b'):
    directory = "/home/istvan/capgMyo/unzipped-db"+db
    directory_out = "/home/istvan/capgMyo/data/db"+db
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mat"):
                aFile = os.path.join(root, file)
                matFile = scipy.io.loadmat(aFile)
                np.save(os.path.join(directory_out, '{:03d}-{:03d}-{:03d}.npy'.format(matFile['subject'][0][0], matFile['gesture'][0][0], matFile['trial'][0][0])), matFile['data'])


# Creating files
def sequenceBatchCreator(batchSize=1024, seq_len=150, strideMode=3, directory_out="numpy_seqs_stride3"):
    directory = "numpy"

    for i in range (11, 21): # second session
        for j in range (1, 9): # 8 gestures
            X = []
            y = []
            for k in range (1, 11): # 10 repetitions
                fileName = '{:03d}-{:03d}-{:03d}.npy'.format(i, j, k)
                aFile = np.load(os.path.join(directory, fileName))
                if strideMode == 1:
                    stride = 1
                elif strideMode == 2:                
                    stride = seq_len//2
                elif strideMode == 3:
                    stride = 50
                
                #numOfSequences = (aFile.shape[0]-seq_len)//stride + 1

                for l in range(0, aFile.shape[0]-seq_len+1, stride):
                    X.append(aFile[l:l+seq_len, :])
                    y.append(j-1) # cause of to_categorical()
            npX = np.array(X)
            np.save(os.path.join(directory_out,
                'X_{:03d}-{:03d}_{:03d}ms_sequences_{:03d}stride.npy'.format(i, j, seq_len, stride)),
                npX)
            npy = np.array(y)
            np.save(os.path.join(directory_out,
                'y_{:03d}-{:03d}_{:03d}ms_sequences_{:03d}stride.npy'.format(i, j, seq_len, stride)),
                npy)

def sequenceFixTrainingGenerator(batchSize=1024, seq_len=150, indexForValidation=20, strideMode=1):
    directory = "numpy"

    for i in sample(range(11, 21), 10): # second session
        if i == indexForValidation: continue
        for j in sample(range(1, 9), 8): # 8 gestures
            X_train = []
            y_train = []
            for k in sample(range(1, 11), 10): # 10 repetitions
                fileName = '{:03d}-{:03d}-{:03d}.npy'.format(i, j, k)
                aFile = np.load(os.path.join(directory, fileName))
                if strideMode == 1:
                    stride = 1
                elif strideMode == 2:                
                    stride = seq_len//2
                elif strideMode == 3:
                    stride = 50
                for l in range(0, aFile.shape[0]-seq_len+1, stride):
                    X_train.append(aFile[l:l+seq_len, :])
                    y_train.append(j-1) # cause of to_categorical()
            npX_train = np.array(X_train)
            yield npX_train, to_categorical(y_train, num_classes=number_of_classes)

def unisonShuffle(X, y):
  s = np.arange(X.shape[0])
  np.random.shuffle(s)
  return X[s], y[s]

def sequenceBatchGeneratorAbsMean2(batchSize,
    seq_len,
    indexes,
    stride,
    allNumOfSamples,
    meanOf,
    shuffling,
    standardize,
    amountOfRepetitions,
    amountOfGestures,
    totalGestures,
    totalRepetitions,
    directory,
    dataset,
    repetitions,
    number_of_classes):
    
    X = []
    y = []
    
    # DBC:
    # mean:	0.00011763552702395671
    # std:	0.008829399359531554
    if dataset == 'dbc':
        mean = 0.00011763552702395671
        std = 0.008829399359531554
    
    if shuffling:
        range_i = sample(indexes, len(indexes))
        range_j = sample(range(1, totalGestures+1), amountOfGestures)
        range_k = sample(range(1, totalRepetitions+1), amountOfRepetitions)
    else:
        range_i = indexes
        range_j = range(1, amountOfGestures+1)
        range_k = range(1, amountOfRepetitions+1)
    
    if repetitions is not None: # intra-session
        if len(repetitions) == 5: # intra-session
            if shuffling:
                range_k = sample(repetitions, amountOfRepetitions)
            else:
                range_k = repetitions[:amountOfRepetitions]
        
    counter = 0
    while True:
        for i in range_i: # users
            for j in range_j: # 8 gestures
                for k in range_k: # 10 repetitions
                    fileName = '{:03d}-{:03d}-{:03d}.npy'.format(i, j, k)
                    aFile = np.load(os.path.join(directory, fileName))
                    if standardize:
                        aFile = (aFile-mean)/std
                    aFile = np.abs(aFile)
                    absMeanFile = np.apply_along_axis(lambda m: np.convolve(m, np.ones((meanOf,))/meanOf, mode='valid'), axis=0, arr=aFile)
                    del aFile
                    for l in range(0, absMeanFile.shape[0]-seq_len+1, stride):
                        X.append(absMeanFile[l:l+seq_len, :])
                        y.append(j-1)
                        counter += 1
                        if counter % allNumOfSamples == 0:
                          #print('\ntraining counter: '+str(counter))
                          if shuffling:
                            yield unisonShuffle(np.array(X), to_categorical(y, num_classes=number_of_classes))
                          else:
                            yield np.array(X), to_categorical(y, num_classes=number_of_classes)
                          del X, y
                          X = []
                          y = []
                          counter = 0
                        elif counter % batchSize == 0:
                          #print('\ntraining counter: '+str(counter))
                          if shuffling:
                            yield unisonShuffle(np.array(X), to_categorical(y, num_classes=number_of_classes))
                          else:
                            yield np.array(X), to_categorical(y, num_classes=number_of_classes)
                          del X, y
                          X = []
                          y = []
                    del absMeanFile

def buildModel(classes, features, cellNeurons, cellDropout, denseDropout, denseNeurons, sequenceLength, stacked=False, bidirectional=False, l2=0.0):
    model = Sequential()
    model.add(TimeDistributed(Dense(features,
      kernel_initializer='identity',
      bias_initializer='zeros',
      name='customNn',
      activation=None), input_shape=(sequenceLength, features), name='td', trainable=False))
    if bidirectional:
        if stacked:
            model.add(Bidirectional(LSTM(cellNeurons, recurrent_dropout=cellDropout, name='rnn', trainable=True, return_sequences=True, kernel_regularizer=regularizers.l2(l2)), merge_mode='concat'))
            model.add(Bidirectional(LSTM(cellNeurons, recurrent_dropout=cellDropout, name='rnn_2nd_layer', trainable=True, kernel_regularizer=regularizers.l2(l2)), merge_mode='concat'))
        else:
            model.add(Bidirectional(LSTM(cellNeurons, recurrent_dropout=cellDropout, name='rnn', trainable=True, kernel_regularizer=regularizers.l2(l2)), merge_mode='concat'))
    else:
        if stacked:
            model.add(LSTM(cellNeurons, recurrent_dropout=cellDropout, name='rnn', trainable=True, return_sequences=True, kernel_regularizer=regularizers.l2(l2)))
            model.add(LSTM(cellNeurons, recurrent_dropout=cellDropout, name='rnn_2nd_layer', trainable=True, kernel_regularizer=regularizers.l2(l2)))
            #model.add(Attention(name='attention', trainable=True))
        else:
            model.add(LSTM(cellNeurons, recurrent_dropout=cellDropout, name='rnn', trainable=True, kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(denseNeurons, name='nn', trainable=True, kernel_regularizer=regularizers.l2(l2)))
    model.add(Dropout(denseDropout, name='nn_dropout', trainable=True))
    model.add(Dense(classes, activation="softmax", name='output_softmax', trainable=True, kernel_regularizer=regularizers.l2(l2)))
    #model.summary()
    
    if onTpu:
        model.compile(loss="categorical_crossentropy",
                optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                metrics=["accuracy"])
    
        multi_model = toTpuModel(model)
    else:
        multi_model = toMultiGpuModel(model)
        multi_model.compile(loss="categorical_crossentropy",
                optimizer=optimizers.Adam(lr=0.001, decay=0.0001),
                metrics=["accuracy"])

    return model, multi_model

class AltModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, alternate_model, **kwargs):
        """
        Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
        :param filepath:
        :param alternate_model: Keras model to save instead of the default. This is used especially when training multi-
                                gpu models built with Keras multi_gpu_model(). In that case, you would pass the original
                                "template model" to be saved each checkpoint.
        :param kwargs:          Passed to ModelCheckpoint.
        """

        self.alternate_model = alternate_model
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super().on_epoch_end(epoch, logs)
        self.model = model_before

def getAdaptationModel(modelPath, adaptationVersion, features, seqLen):
    fineTuneModel = load_model(modelPath)
    
    # Test optimizer's state:
    #print(fineTuneModel.optimizer.get_config())
    #print(dir(fineTuneModel.optimizer))
    #print(fineTuneModel.optimizer.lr)
    
    fineTuneModel.get_layer('td').trainable = True
    if adaptationVersion == 21:
        fineTuneModel.get_layer('td').activation = 'relu'
    if adaptationVersion == 22:
        fineTuneModel.get_layer('td').activation = 'exponential'
    if adaptationVersion == 23:
        fineTuneModel.get_layer('td').activation = 'elu'
    if adaptationVersion == 24:
        fineTuneModel.get_layer('td').activation = 'sigmoid'
    fineTuneModel.get_layer('rnn').trainable = False
    if fineTuneModel.get_layer('rnn_2nd_layer') != None:
        fineTuneModel.get_layer('rnn_2nd_layer').trainable = False
    fineTuneModel.get_layer('nn').trainable = False
    fineTuneModel.get_layer('nn_dropout').trainable = False
    fineTuneModel.get_layer('output_softmax').trainable = False
    
    if adaptationVersion == 31:
        fineTuneModel.get_layer('td').activation = 'relu'
        fineTuneModel.name = "existingModel"
        newModel = Sequential()
        newModel.add(TimeDistributed(Dense(features,
            kernel_initializer='identity',
            bias_initializer='zeros',
            activation='relu'), input_shape=(seqLen, features), name='td0', trainable=True))
        newModel.add(fineTuneModel)
        fineTuneModel = newModel
    if adaptationVersion == 32:
        fineTuneModel.get_layer('td').activation = 'exponential'
        fineTuneModel.name = "existingModel"
        newModel = Sequential()
        newModel.add(TimeDistributed(Dense(features,
            kernel_initializer='identity',
            bias_initializer='zeros',
            activation='exponential'), input_shape=(seqLen, features), name='td0', trainable=True))
        newModel.add(fineTuneModel)
        fineTuneModel = newModel
    if adaptationVersion == 34:
        fineTuneModel.get_layer('td').activation = 'sigmoid'
        fineTuneModel.name = "existingModel"
        newModel = Sequential()
        newModel.add(TimeDistributed(Dense(features,
            kernel_initializer='identity',
            bias_initializer='zeros',
            activation='sigmoid'), input_shape=(seqLen, features), name='td0', trainable=True))
        newModel.add(fineTuneModel)
        fineTuneModel = newModel
    if adaptationVersion == 4: # initializer does not work with this initializer cause it is not square
        fineTuneModel.get_layer('td').activation = 'relu'
        fineTuneModel.name = "existingModel"
        newModel = Sequential()
        newModel.add(TimeDistributed(Dense(10*features,
            kernel_initializer='identity',
            bias_initializer='zeros',
            activation='relu'), input_shape=(seqLen, features), name='td0', trainable=True))
        newModel.add(fineTuneModel)
        fineTuneModel = newModel
    
    if onTpu:
        multiFineTuneModel.compile(loss="categorical_crossentropy",
                    optimizer=tf.train.AdamOptimizer(lr=0.001),
                    metrics=["accuracy"])
        multiFineTuneModel = toTpuModel(fineTuneModel)
    else:
        multiFineTuneModel = toMultiGpuModel(fineTuneModel)
        multiFineTuneModel.compile(loss="categorical_crossentropy",
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics=["accuracy"])

    # Test optimizer's state:
    #print(fineTuneModel.optimizer.get_config())
    #print(dir(fineTuneModel.optimizer))
    #print(fineTuneModel.optimizer.lr)
    
    return fineTuneModel, multiFineTuneModel

def validationAccuracyValues(x):
    return(x[-10:-5])

def getBestModel(testUser, workingDirectory):
    file_list = os.listdir(workingDirectory+str(testUser))
    best = max(file_list, key=validationAccuracyValues)
    print('\nBest pre-trained model to start with: ' + str(best))
    return workingDirectory+str(testUser) + '/' + best

class WeightsNorm(Callback):
    def on_batch_end(self, batch, logs={}):
        # Norm clipping:
        print(str(math.sqrt(sum(np.sum(K.get_value(w)) for w in self.model.optimizer.weights))) + '\n')
        return

def checkdataStatistics():
    directory = "/home/istvan/capgMyo/data/dbc"
    
    bigArray = None
    
    isFirst = True
    for i in range (1, 11):
        for j in range (1, 9): # 8 gestures
            for k in range (1, 11): # 10 repetitions
                fileName = '{:03d}-{:03d}-{:03d}.npy'.format(i, j, k)
                aFile = np.load(os.path.join(directory, fileName))
                if isFirst:
                    bigArray = aFile
                    isFirst = False
                else:
                    bigArray = np.concatenate((bigArray, aFile), axis=0)
    mean, std = np.mean(bigArray), np.std(bigArray)
    print(mean)
    print(std)
    print(np.max(bigArray))
    print(np.min(bigArray))
    newArray = (bigArray - mean)/std
    mean2, std2 = np.mean(newArray), np.std(newArray)
    print(mean2)
    print(std2)
    print(np.max(newArray))
    print(np.min(newArray))
    plt.hist(bigArray, bins=10)
    plt.title("Histogram")
    plt.savefig('histogram-all.png')
    plt.close()
    plt.hist(newArray, bins=10)
    plt.title("Histogram")
    plt.savefig('histogram-new-all.png')

def preTrainingModel(trainingUsers,
        testUsers,
        allNumOfTrainingSamples,
        trainingStepsPerEpoch,
        allNumOfValidationSamples,
        validationStepsPerEpoch,
        amountOfRepetitions,
        amountOfGestures,
        preTrainingNumOfEpochs,
        trial,
        batchSize,
        totalGestures,
        totalRepetitions,
        directory,
        testUser,
        workingDirectory,
		trainingDataset,
		testDataset,
		trainingRepetitions,
		testRepetitions,
        number_of_classes,
        saveCheckpoints):
    
    base_model, multi_model = buildModel(classes=number_of_classes,
            features=NUMBER_OF_FEATURES,
            cellNeurons=cellNeurons,
            cellDropout=recurrent_dropout,
            denseDropout=dropout,
            denseNeurons=denseNeurons,
            sequenceLength=seq_len,
            stacked=True,
            bidirectional=False,
            l2=0.0)
    
    histories = {}
    
    path = workingDirectory+str(testUser)
    if not os.path.exists(path):
        os.makedirs(path)

    my_training_batch_generator = sequenceBatchGeneratorAbsMean2(batchSize=batchSize,
                                                            seq_len=seq_len,
                                                            indexes=trainingUsers,
                                                            stride=stride,
                                                            allNumOfSamples=allNumOfTrainingSamples,
                                                            meanOf=mean,
                                                            shuffling=True,
                                                            standardize=True,
                                                            amountOfRepetitions=totalRepetitions,
                                                            amountOfGestures=totalGestures,
                                                            totalGestures=totalGestures,
                                                            totalRepetitions=totalRepetitions,
                                                            directory=directory,
                                                            dataset=trainingDataset,
															repetitions=trainingRepetitions,
                                                            number_of_classes=number_of_classes)
    my_validation_batch_generator = sequenceBatchGeneratorAbsMean2(batchSize=batchSize,
                                                            seq_len=seq_len,
                                                            indexes=testUsers,
                                                            stride=stride,
                                                            allNumOfSamples=allNumOfValidationSamples,
                                                            meanOf=mean,
                                                            shuffling=False,
                                                            standardize=True,
                                                            amountOfRepetitions=totalRepetitions,
                                                            amountOfGestures=totalGestures,
                                                            totalGestures=totalGestures,
                                                            totalRepetitions=totalRepetitions,
                                                            directory=directory,
                                                            dataset=testDataset,
															repetitions=testRepetitions,
                                                            number_of_classes=number_of_classes)
    
    filepath=path + "/e{epoch:03d}-a{val_acc:.3f}.hdf5"
    
    #lrate = LearningRateScheduler(step_decay)
    if saveCheckpoints == True:
        if onTpu:
            modelCheckpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        else:
            modelCheckpoint = AltModelCheckpoint(filepath, alternate_model=base_model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [modelCheckpoint]
    else:
        callbacks_list = []
    
    startTime = int(round(time.time()))
    print("\n##### Start Time with test user "+str(testUser)+": "+str(startTime))
    histories[testUser] = multi_model.fit_generator(generator=my_training_batch_generator,
                                        steps_per_epoch=trainingStepsPerEpoch,
                                        epochs=preTrainingNumOfEpochs,
                                        #max_queue_size=5,
                                        verbose=2,
                                        callbacks=callbacks_list,
                                        validation_data=my_validation_batch_generator,
                                        validation_steps=validationStepsPerEpoch,
                                        use_multiprocessing=False)

    endTime = int(round(time.time()))
    print("\n##### End Time with test user "+str(testUser)+": "+str(endTime))
    toLog = str(preTrainingNumOfEpochs) + ',' + str(seq_len) + ',' + str(stride) + ',' + str(batchSize) + ',' + str(mean)
    with open(workingDirectory+"history.csv", "a") as myfile:
        myfile.write(str(endTime)\
            + ',' + str(trial)\
            + ',' + str(testUser)\
            + ',' + str(max(histories[testUser].history['acc']))\
            + ',' + str(max(histories[testUser].history['val_acc']))\
            + ',' + str(endTime-startTime)\
            + ',' + toLog\
            + ',' + str(amountOfRepetitions)\
            + ',' + str(amountOfGestures) + '\n')
    
    del histories
    del base_model, multi_model
    del my_training_batch_generator, my_validation_batch_generator

def adaptModel(fineTuneUsers, testUsers, allNumOfFineTuningSamples, fineTuningStepsPerEpoch, amountOfRepetitions, amountOfGestures,
        allNumOfValidationSamples,
        validationStepsPerEpoch,
        numberOfFineTuningEpochs,
        trial,
        batchSize,
        totalGestures,
        totalRepetitions,
        directory,
        testUser,
        workingDirectory,
		trainingDataset,
		testDataset,
		trainingRepetitions,
		testRepetitions,
        number_of_classes,
        adaptationVersion):
    
    base_model, multi_model = getAdaptationModel(modelPath=getBestModel(testUser, workingDirectory), adaptationVersion=adaptationVersion, features=NUMBER_OF_FEATURES, seqLen=seq_len)
    
    histories = {}
    
    path = workingDirectory+str(testUser)+'-adapted'
    if not os.path.exists(path):
        os.makedirs(path)

    my_training_batch_generator = sequenceBatchGeneratorAbsMean2(batchSize=batchSize,
                                                            seq_len=seq_len,
                                                            indexes=fineTuneUsers,
                                                            stride=stride,
                                                            allNumOfSamples=allNumOfFineTuningSamples,
                                                            meanOf=mean,
                                                            shuffling=True,
                                                            standardize=True,
                                                            amountOfRepetitions=amountOfRepetitions,
                                                            amountOfGestures=amountOfGestures,
                                                            totalGestures=totalGestures,
                                                            totalRepetitions=totalRepetitions,
                                                            directory=directory,
															dataset=trainingDataset,
															repetitions=trainingRepetitions,
                                                            number_of_classes=number_of_classes)
    my_validation_batch_generator = sequenceBatchGeneratorAbsMean2(batchSize=batchSize,
                                                            seq_len=seq_len,
                                                            indexes=testUsers,
                                                            stride=stride,
                                                            allNumOfSamples=allNumOfValidationSamples,
                                                            meanOf=mean,
                                                            shuffling=False,
                                                            standardize=True,
                                                            amountOfRepetitions=totalRepetitions,
                                                            amountOfGestures=totalGestures,
                                                            totalGestures=totalGestures,
                                                            totalRepetitions=totalRepetitions,
                                                            directory=directory,
															dataset=testDataset,
															repetitions=testRepetitions,
                                                            number_of_classes=number_of_classes)

    filepath=path + "/e{epoch:03d}-a{val_acc:.3f}.hdf5"
    #modelCheckpoint = AltModelCheckpoint(filepath, alternate_model=base_model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    #csv_logger = CSVLogger(workingDirectory+'log_u' + str(testUser) + '-adapted.csv', append=True, separator=',')
    #callbacks_list = [modelCheckpoint]
    startTime = int(round(time.time()))
    print("\n##### Start Time with test user "+str(testUser)+": "+str(startTime))
    histories[testUser] = multi_model.fit_generator(generator=my_training_batch_generator,
                                        steps_per_epoch=fineTuningStepsPerEpoch,
                                        epochs=numberOfFineTuningEpochs,
                                        #max_queue_size=5,
                                        verbose=2,
                                        #callbacks=callbacks_list,
                                        validation_data=my_validation_batch_generator,
                                        validation_steps=validationStepsPerEpoch,
                                        use_multiprocessing=False)

    endTime = int(round(time.time()))
    print("\n##### End Time with test user "+str(testUser)+": "+str(endTime))
    toLog = str(numberOfFineTuningEpochs) + ',' + str(seq_len) + ',' + str(stride) + ',' + str(batchSize) + ',' + str(mean)
    with open(workingDirectory+"history-adapted.csv", "a") as myfile:
        myfile.write(str(endTime)\
            + ',' + str(trial)\
            + ',' + str(testUser)\
            + ',' + str(max(histories[testUser].history['accuracy']))\
            + ',' + str(max(histories[testUser].history['val_accuracy']))\
            + ',' + str(endTime-startTime)\
            + ',' + toLog\
            + ',' + str(amountOfRepetitions)\
            + ',' + str(amountOfGestures) + '\n')
    
    del histories
    del base_model, multi_model
    del my_training_batch_generator, my_validation_batch_generator

#fineTuningEpochList = [1, 2, 4, 8, 16, 32, 64]
#fineTuningRepetitionList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#fineTuningGestureList = []
fineTuningEpochList = [100]
fineTuningRepetitionList = [10]
fineTuningGestureList = [12]

preTrainingNumOfEpochs = 200
seq_len = 150
stride = 70
mean=11
#seq_len = 40
#stride = 19
#mean=11
totalGestures=12
totalRepetitions=10

def generalScenario(validation, training, adaptationVersion):
    for trial in range(60, 63):
        for numberOfFineTuningEpochs in fineTuningEpochList:
            for amountOfGestures in fineTuningGestureList:
                for amountOfRepetitions in fineTuningRepetitionList:
                    batchSize = 780
                    
                    if validation == 'inter-subject':
                        allNumOfFineTuningSamples = 1 * amountOfGestures * amountOfRepetitions * ((1000-mean+1-seq_len)//stride + 1)
                        fineTuningStepsPerEpoch = allNumOfFineTuningSamples // batchSize
                        allNumOfValidationSamples = 1 * 12 * 10 * ((1000-mean+1-seq_len)//stride + 1)
                        validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                        allNumOfTrainingSamples = 9 * 12 * 10 * ((1000-mean+1-seq_len)//stride + 1)
                        trainingStepsPerEpoch = allNumOfTrainingSamples // batchSize
                        directory = baseDrive+"data/dbc"
                        workingDirectory = baseDrive+'data/inter-subject/dbc/'
                        trainingUsers = list(range(1, 11))
                        for testUser in trainingUsers: # LOSOCV for inter-subject validation
                            fineTuneUsers = [testUser]
                            testUsers = [testUser]
                            currentTrainingUsers = trainingUsers.copy()
                            currentTrainingUsers.remove(testUser)
                            if training == 'pre-training':
                                preTrainingModel(trainingUsers=currentTrainingUsers,
                                    testUsers=testUsers,
                                    allNumOfTrainingSamples=allNumOfTrainingSamples,
                                    trainingStepsPerEpoch=trainingStepsPerEpoch,
                                    allNumOfValidationSamples=allNumOfValidationSamples,
                                    validationStepsPerEpoch=validationStepsPerEpoch,
                                    amountOfRepetitions=amountOfRepetitions,
                                    amountOfGestures=amountOfGestures,
                                    preTrainingNumOfEpochs=preTrainingNumOfEpochs,
                                    trial=trial,
                                    batchSize=batchSize,
                                    totalGestures=totalGestures,
                                    totalRepetitions=totalRepetitions,
                                    directory=directory,
                                    testUser=testUser,
                                    workingDirectory=workingDirectory,
									trainingDataset='dbc',
									testDataset='dbc',
									trainingRepetitions=None,
									testRepetitions=None,
                                    number_of_classes=number_of_classes,
                                    saveCheckpoints=True)
                            elif training == 'fine-tuning':
                                adaptModel(fineTuneUsers=fineTuneUsers,
                                    testUsers=testUsers,
                                    allNumOfFineTuningSamples=allNumOfFineTuningSamples,
                                    fineTuningStepsPerEpoch=fineTuningStepsPerEpoch,
                                    allNumOfValidationSamples=allNumOfValidationSamples,
                                    validationStepsPerEpoch=validationStepsPerEpoch,
                                    amountOfRepetitions=amountOfRepetitions,
                                    amountOfGestures=amountOfGestures,
                                    numberOfFineTuningEpochs=numberOfFineTuningEpochs,
                                    trial=trial,
                                    batchSize=batchSize,
                                    totalGestures=totalGestures,
                                    totalRepetitions=totalRepetitions,
                                    directory=directory,
                                    testUser=testUser,
                                    workingDirectory=workingDirectory,
									trainingDataset='dbc',
									testDataset='dbc',
									trainingRepetitions=None,
									testRepetitions=None,
                                    number_of_classes=number_of_classes,
                                    adaptationVersion=adaptationVersion)
                    if validation == 'inter-subject-split':
                        directory = baseDrive+"data/dbc"
                        workingDirectory = baseDrive+'data/inter-subject-split/dbc/'
                        trainingUsers = list(range(1, 11))
                        for testUser in trainingUsers: # LOSOCV for inter-subject validation
                            testUsers = [testUser]
                            if training == 'pre-training':
                                currentTrainingUsers = trainingUsers.copy()
                                currentTrainingUsers.remove(testUser)
                                allNumOfTrainingSamples = 9 * 12 * 10 * ((1000-mean+1-seq_len)//stride + 1)
                                trainingStepsPerEpoch = allNumOfTrainingSamples // batchSize
                                allNumOfValidationSamples = 1 * 12 * 10 * ((1000-mean+1-seq_len)//stride + 1)
                                validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                                preTrainingModel(trainingUsers=currentTrainingUsers,
                                    testUsers=testUsers,
                                    allNumOfTrainingSamples=allNumOfTrainingSamples,
                                    trainingStepsPerEpoch=trainingStepsPerEpoch,
                                    allNumOfValidationSamples=allNumOfValidationSamples,
                                    validationStepsPerEpoch=validationStepsPerEpoch,
                                    amountOfRepetitions=10,
                                    amountOfGestures=12,
                                    preTrainingNumOfEpochs=preTrainingNumOfEpochs,
                                    trial=trial,
                                    batchSize=batchSize,
                                    totalGestures=12,
                                    totalRepetitions=10,
                                    directory=directory,
                                    testUser=testUser,
                                    workingDirectory=workingDirectory,
									trainingDataset='dbc',
									testDataset='dbc',
									trainingRepetitions=None,
									testRepetitions=None,
                                    number_of_classes=number_of_classes,
                                    saveCheckpoints=True)
                            elif training == 'fine-tuning':
                                trainingRepetitions = [1, 3, 5, 7, 9]
                                testRepetitions = [2, 4, 6, 8, 10]
                                fineTuneUsers = [testUser]
                                allNumOfFineTuningSamples = 1 * amountOfGestures * 5 * ((1000-mean+1-seq_len)//stride + 1)
                                fineTuningStepsPerEpoch = allNumOfFineTuningSamples // batchSize
                                allNumOfValidationSamples = 1 * 12 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                                validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                                adaptModel(fineTuneUsers=fineTuneUsers,
                                    testUsers=testUsers,
                                    allNumOfFineTuningSamples=allNumOfFineTuningSamples,
                                    fineTuningStepsPerEpoch=fineTuningStepsPerEpoch,
                                    allNumOfValidationSamples=allNumOfValidationSamples,
                                    validationStepsPerEpoch=validationStepsPerEpoch,
                                    amountOfRepetitions=5,
                                    amountOfGestures=amountOfGestures,
                                    numberOfFineTuningEpochs=numberOfFineTuningEpochs,
                                    trial=trial,
                                    batchSize=batchSize,
                                    totalGestures=totalGestures,
                                    totalRepetitions=5,
                                    directory=directory,
                                    testUser=testUser,
                                    workingDirectory=workingDirectory,
									trainingDataset='dbc',
									testDataset='dbc',
									trainingRepetitions=trainingRepetitions,
									testRepetitions=testRepetitions,
                                    number_of_classes=number_of_classes,
                                    adaptationVersion=adaptationVersion)
                    elif validation == 'intra-session':
                        allNumOfValidationSamples = 10 * 12 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                        validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                        allNumOfTrainingSamples = 10 * 12 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                        trainingStepsPerEpoch = allNumOfTrainingSamples // batchSize
                        directory = baseDrive+"data/dbc"
                        workingDirectory = baseDrive+'data/intra-session/dbc/'
                        trainingUsers = list(range(1, 11))
                        testUsers = list(range(1, 11))
                        trainingRepetitions = [1, 3, 5, 7, 9]
                        testRepetitions = [2, 4, 6, 8, 10]
                        testUser='evenRepetitions'
                        if training == 'pre-training':
                            preTrainingModel(trainingUsers=trainingUsers,
                                testUsers=testUsers,
                                allNumOfTrainingSamples=allNumOfTrainingSamples,
                                trainingStepsPerEpoch=trainingStepsPerEpoch,
                                allNumOfValidationSamples=allNumOfValidationSamples,
                                validationStepsPerEpoch=validationStepsPerEpoch,
                                amountOfRepetitions=5,
                                amountOfGestures=amountOfGestures,
                                preTrainingNumOfEpochs=preTrainingNumOfEpochs,
                                trial=trial,
                                batchSize=batchSize,
                                totalGestures=totalGestures,
                                totalRepetitions=5,
                                directory=directory,
                                testUser=testUser,
                                workingDirectory=workingDirectory,
                                trainingDataset='dbc',
                                testDataset='dbc',
                                trainingRepetitions=trainingRepetitions,
                                testRepetitions=testRepetitions,
                                number_of_classes=number_of_classes,
                                saveCheckpoints=False)
                    elif validation == 'intra-session-separated':
                        allNumOfValidationSamples = 1 * 12 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                        validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                        allNumOfTrainingSamples = 1 * 12 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                        trainingStepsPerEpoch = allNumOfTrainingSamples // batchSize
                        directory = baseDrive+"data/dbc"
                        workingDirectory = baseDrive+'data/intra-session-separated/dbc/'
                        trainingUsers = list(range(1, 11))
                        for subject in trainingUsers:
                            print("subject: " + str(subject))
                            trainingUsers = [subject]
                            testUsers = [subject]
                            trainingRepetitions = [1, 3, 5, 7, 9]
                            testRepetitions = [2, 4, 6, 8, 10]
                            testUser=subject
                            if training == 'pre-training':
                                preTrainingModel(trainingUsers=trainingUsers,
                                    testUsers=testUsers,
                                    allNumOfTrainingSamples=allNumOfTrainingSamples,
                                    trainingStepsPerEpoch=trainingStepsPerEpoch,
                                    allNumOfValidationSamples=allNumOfValidationSamples,
                                    validationStepsPerEpoch=validationStepsPerEpoch,
                                    amountOfRepetitions=5,
                                    amountOfGestures=amountOfGestures,
                                    preTrainingNumOfEpochs=preTrainingNumOfEpochs,
                                    trial=trial,
                                    batchSize=batchSize,
                                    totalGestures=totalGestures,
                                    totalRepetitions=5,
                                    directory=directory,
                                    testUser=testUser,
                                    workingDirectory=workingDirectory,
                                    trainingDataset='dbc',
                                    testDataset='dbc',
                                    trainingRepetitions=trainingRepetitions,
                                    testRepetitions=testRepetitions,
                                    number_of_classes=number_of_classes,
                                    saveCheckpoints=False)

#Istvan Titan V

#Szerda ejjel:
generalScenario(validation='inter-subject-split', training='fine-tuning', adaptationVersion=31)
generalScenario(validation='inter-subject', training='fine-tuning', adaptationVersion=31)