import tensorflow as tf
import os
import numpy as np
import time
import sys
from random import randint, sample
from collections import Counter, OrderedDict
import subprocess
import io
import scipy.io
import math
import matplotlib.pyplot as plt
from keras.layers import Input, LSTM, TimeDistributed, Dense, Layer
from keras.models import Sequential, load_model
from keras.layers.core import Dropout
from keras import initializers, optimizers, regularizers, constraints
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, Callback
from keras.utils import to_categorical, multi_gpu_model

baseDrive = '/home/istvan/2019_dec_paper/capgMyo/'

NUMBER_OF_FEATURES = 128
recurrent_dropout = 0.5
dropout = 0.5
number_of_classes = 8
cellNeurons = 512
denseNeurons = 512

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

def convert(db='b'):
    directory = "/home/istvan/capgMyo/unzipped-db"+db
    directory_out = "/home/istvan/capgMyo/data/db"+db
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mat"):
                aFile = os.path.join(root, file)
                matFile = scipy.io.loadmat(aFile)
                np.save(os.path.join(directory_out, '{:03d}-{:03d}-{:03d}.npy'.format(matFile['subject'][0][0], matFile['gesture'][0][0], matFile['trial'][0][0])), matFile['data'])


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
    
    # DBB:
    # 1st half's mean:	-1.494554769724577e-06
    # 1st half's std:	0.014456551081024154
    # 2nd half's mean:	-3.342415416859754e-07
    # 2nd half's std:	0.012518382863642336
    # All's mean:		-9.143981557052827e-07
    # All's std:		0.013522236859191582
    if dataset == 'dbb-1st':
      mean = -1.494554769724577e-06
      std = 0.014456551081024154
    elif dataset == 'dbb-2nd':
      mean = -3.342415416859754e-07
      std = 0.012518382863642336
    elif dataset == 'dbb-all':
      mean = -3.342415416859754e-07
      std = 0.012518382863642336
    
    if shuffling:
        range_i = sample(indexes, len(indexes))
        range_j = sample(range(1, totalGestures+1), amountOfGestures)
    else:
        range_i = indexes
        range_j = range(1, amountOfGestures+1)
    
    if repetitions is not None:
        if shuffling:
            range_k = sample(repetitions, len(repetitions))
        else:
            range_k = repetitions
        
    counter = 0
    while True:
        for i in range_i: # users
            for j in range_j: # gestures
                for k in range_k: # repetitions
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
                          if shuffling:
                            yield unisonShuffle(np.array(X), to_categorical(y, num_classes=number_of_classes))
                          else:
                            yield np.array(X), to_categorical(y, num_classes=number_of_classes)
                          del X, y
                          X = []
                          y = []
                          counter = 0
                        elif counter % batchSize == 0:
                          if shuffling:
                            yield unisonShuffle(np.array(X), to_categorical(y, num_classes=number_of_classes))
                          else:
                            yield np.array(X), to_categorical(y, num_classes=number_of_classes)
                          del X, y
                          X = []
                          y = []
                    del absMeanFile

def sequence_data(batchSize,
    seq_len,
    indexes,
    stride,
    meanOf,
    standardize,
    directory,
    dataset,
    repetitions,
    number_of_classes):
    
    # DBB:
    # 1st half's mean:	-1.494554769724577e-06
    # 1st half's std:	0.014456551081024154
    # 2nd half's mean:	-3.342415416859754e-07
    # 2nd half's std:	0.012518382863642336
    # All's mean:		-9.143981557052827e-07
    # All's std:		0.013522236859191582
    if dataset == 'dbb-1st':
      mean = -1.494554769724577e-06
      std = 0.014456551081024154
    elif dataset == 'dbb-2nd':
      mean = -3.342415416859754e-07
      std = 0.012518382863642336
    elif dataset == 'dbb-all':
      mean = -3.342415416859754e-07
      std = 0.012518382863642336
    

    range_i = indexes
    range_j = range(1, 9)
    range_k = repetitions
    
    counter = 0
    
    batch_X = []
    batch_y = []
    epoch_X = None
    epoch_y = None
    
    for i in range_i: # users
        for j in range_j: # gestures
            for k in range_k: # repetitions
                fileName = '{:03d}-{:03d}-{:03d}.npy'.format(i, j, k)
                aFile = np.load(os.path.join(directory, fileName))
                if standardize:
                    aFile = (aFile-mean)/std
                aFile = np.abs(aFile)
                absMeanFile = np.apply_along_axis(lambda m: np.convolve(m, np.ones((meanOf,))/meanOf, mode='valid'), axis=0, arr=aFile)
                del aFile
                for l in range(0, absMeanFile.shape[0]-seq_len+1, stride):
                    batch_X.append(absMeanFile[l:l+seq_len, :])
                    batch_y.append(j-1)
                    
                    counter += 1
                    
                    if counter % batchSize == 0:
                        if epoch_X is None:
                            epoch_X = np.array(batch_X)
                            epoch_y = to_categorical(batch_y, num_classes=number_of_classes)
                            batch_X = []
                            batch_y = []
                        else:
                            epoch_X = np.concatenate((epoch_X, np.array(batch_X)))
                            epoch_y = np.concatenate((epoch_y, to_categorical(batch_y, num_classes=number_of_classes)))
                            batch_X = []
                            batch_y = []
                del absMeanFile
    
    return epoch_X, epoch_y

def getAdaptationModel(modelPath, adaptationVersion, features, seqLen):
    fineTuneModel = load_model(modelPath)
    
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
    if fineTuneModel.get_layer('rnn_2nd_layer') is not None:
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
    

    multiFineTuneModel = toMultiGpuModel(fineTuneModel)
    multiFineTuneModel.compile(loss="categorical_crossentropy",
                optimizer=optimizers.Adam(lr=0.001, decay=0.0001),
                metrics=["accuracy"])

    return fineTuneModel, multiFineTuneModel

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
        else:
            model.add(LSTM(cellNeurons, recurrent_dropout=cellDropout, name='rnn', trainable=True, kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(denseNeurons, name='nn', trainable=True, kernel_regularizer=regularizers.l2(l2)))
    model.add(Dropout(denseDropout, name='nn_dropout', trainable=True))
    model.add(Dense(classes, activation="softmax", name='output_softmax', trainable=True, kernel_regularizer=regularizers.l2(l2)))
    
    multi_model = toMultiGpuModel(model)
    multi_model.compile(loss="categorical_crossentropy",
            optimizer=optimizers.Adam(lr=0.001),
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

def validationAccuracyValues(x):
    return(x[-10:-5])

def getBestModel(testUser, workingDirectory):
    file_list = os.listdir(workingDirectory+str(testUser))
    best = max(file_list, key=validationAccuracyValues)
    print('\nBest pre-trained model to start with: ' + str(best))
    return workingDirectory+str(testUser) + '/' + best

def checkdataStatistics():
    directory = "/home/istvan/capgMyo/data/dbb"
    
    bigArray = None
    
    isFirst = True
    for i in range (1, 21): # subjects
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
    # 2nd half's mean: -3.342415416859754e-07
    # 2nd half's std:  0.012518382863642336
    # All's mean: -9.143981557052827e-07
    # All's std:  0.013522236859191582
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
    
    filepath=path + "/e{epoch:03d}-a{val_accuracy:.3f}.hdf5"
    
    if saveCheckpoints == True:
        modelCheckpoint = AltModelCheckpoint(filepath, alternate_model=base_model, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
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
            + ',' + str(max(histories[testUser].history['accuracy']))\
            + ',' + str(max(histories[testUser].history['val_accuracy']))\
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
    
    path = workingDirectory+str(testUser)+'-adapted-'+str(adaptationVersion)
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

    filepath=path + "/e{epoch:03d}-a{val_accuracy:.3f}.hdf5"
    modelCheckpoint = AltModelCheckpoint(filepath, alternate_model=base_model, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    #csv_logger = CSVLogger(workingDirectory+'log_u' + str(testUser) + '-adapted.csv', append=True, separator=',')
    callbacks_list = [modelCheckpoint]
    startTime = int(round(time.time()))
    print("\n##### Start Time with test user "+str(testUser)+": "+str(startTime))
    histories[testUser] = multi_model.fit_generator(generator=my_training_batch_generator,
                                        steps_per_epoch=fineTuningStepsPerEpoch,
                                        epochs=numberOfFineTuningEpochs,
                                        #max_queue_size=5,
                                        verbose=2,
                                        callbacks=callbacks_list,
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

preTrainingNumOfEpochs = 100
numberOfFineTuningEpochs = 100
seq_len = 150
stride = 70
mean = 11


def intra_session_residual_distribution():
    batchSize = 520
    
    allNumOfValidationSamples = 1 * 8 * 5 * ((1000-mean+1-seq_len)//stride + 1)
    validationStepsPerEpoch = allNumOfValidationSamples // batchSize
    validationSubjects = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    validationRepetitions = [2, 4, 6, 8, 10]
    
    workingDirectory = baseDrive+'data/intra-session-separated-metric/dbb/'
    
    cross_entropy = None
    for subject in validationSubjects:
        print("subject: " + str(subject))
        
        X, y = sequence_data(batchSize=batchSize,
                                        seq_len=seq_len,
                                        indexes=[subject],
                                        stride=stride,
                                        meanOf=mean,
                                        standardize=True,
                                        directory=baseDrive+"data/dbb",
                                        dataset='dbb-all',
                                        repetitions=validationRepetitions,
                                        number_of_classes=number_of_classes)
        
        
        model = load_model(getBestModel(subject, workingDirectory))
        result = model.predict(X, batch_size=batchSize)
        
        cross_entropy_i = -np.log(result) * y
        cross_entropy_i = cross_entropy_i[np.where(cross_entropy_i > 0)]
        if cross_entropy is None:
            cross_entropy = cross_entropy_i
        else:
            cross_entropy = np.concatenate((cross_entropy, cross_entropy_i))
        
    np.save(workingDirectory+'intra-session_cross-entropy.npy', cross_entropy)
    return cross_entropy

def inter_session_residual_distribution(adaptationVersion):
    cross_entropy = None
    batchSize = 1040
    allNumOfValidationSamples = 1 * 8 * 10 * ((1000-mean+1-seq_len)//stride + 1)
    validationStepsPerEpoch = allNumOfValidationSamples // batchSize
    validationSubjects = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    validationRepetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    if adaptationVersion == 1000:
        workingDirectory = baseDrive+'data/inter-session-metric/dbb/'
        output = workingDirectory+'inter-session_cross-entropy_1000.npy'
    elif adaptationVersion == 31:
        workingDirectory = baseDrive+'data/inter-session-metric/dbb/'
        output = workingDirectory+'inter-session_cross-entropy_31.npy'
    elif adaptationVersion == 999:
        workingDirectory = baseDrive+'data/inter-session-correct-split/dbb/'
        output = workingDirectory+'inter-session_cross-entropy_999.npy'
    elif adaptationVersion > 1000:
        workingDirectory = baseDrive+'data/intra-session-separated-metric/dbb/'
        output = workingDirectory+'inter-session_cross-entropy.npy'
        
    for subject in validationSubjects:
        print("subject: " + str(subject))
        
        target_subject = subject + 1
        
        X, y = sequence_data(batchSize=batchSize,
                                        seq_len=seq_len,
                                        indexes=[target_subject],
                                        stride=stride,
                                        meanOf=mean,
                                        standardize=True,
                                        directory=baseDrive+"data/dbb",
                                        dataset='dbb-all',
                                        repetitions=validationRepetitions,
                                        number_of_classes=number_of_classes)
        
        model = None
        if adaptationVersion == 1000:
            model = load_model(getBestModel('2ndSession-adapted-1000', workingDirectory))
        elif adaptationVersion == 31:
            model = load_model(getBestModel('2ndSession-adapted-31', workingDirectory))
        elif adaptationVersion == 999:
            model = load_model(getBestModel('2ndSession', workingDirectory))
        elif adaptationVersion > 1000:
            model = load_model(getBestModel(subject, workingDirectory))
        
        result = model.predict(X, batch_size=batchSize)
        
        cross_entropy_i = -np.log(result) * y
        cross_entropy_i = cross_entropy_i[np.where(cross_entropy_i > 0)]
        if cross_entropy is None:
            cross_entropy = cross_entropy_i
        else:
            cross_entropy = np.concatenate((cross_entropy, cross_entropy_i))
        
    np.save(output, cross_entropy)
    return cross_entropy

def inter_subject_residual_distribution(adaptationVersion):
    batchSize = 1040
    
    allNumOfValidationSamples = 1 * 8 * 10 * ((1000-mean+1-seq_len)//stride + 1)
    validationStepsPerEpoch = allNumOfValidationSamples // batchSize
    validationSubjects = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    validationRepetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    if adaptationVersion == 1000:
        workingDirectory = baseDrive+'data/inter-subject-metric/dbb/'
        output = workingDirectory+'inter-subject_cross-entropy_1000.npy'
    elif adaptationVersion == 31:
        workingDirectory = baseDrive+'data/inter-subject-metric/dbb/'
        output = workingDirectory+'inter-subject_cross-entropy_31.npy'
    elif adaptationVersion == 999:
        workingDirectory = baseDrive+'data/inter-subject-correct-split/dbb/'
        output = workingDirectory+'inter-subject_cross-entropy_999.npy'
    elif adaptationVersion > 1000:
        workingDirectory = baseDrive+'data/intra-session-separated-metric/dbb/'
        output = workingDirectory+'inter-subject_cross-entropy.npy'
    
    cross_entropy = None
    counter = 0
    for subject in validationSubjects:
        print("subject: " + str(subject))
        
        if (adaptationVersion == 1000) or (adaptationVersion == 31):
            counter += 1
            print(counter)
            
            X, y = sequence_data(batchSize=batchSize,
                                            seq_len=seq_len,
                                            indexes=[subject],
                                            stride=stride,
                                            meanOf=mean,
                                            standardize=True,
                                            directory=baseDrive+"data/dbb",
                                            dataset='dbb-all',
                                            repetitions=validationRepetitions,
                                            number_of_classes=number_of_classes)
            
            
            model = load_model(getBestModel(str(subject)+'-adapted-'+str(adaptationVersion), workingDirectory))
            result = model.predict(X, batch_size=batchSize)
            
            cross_entropy_i = -np.log(result) * y
            cross_entropy_i = cross_entropy_i[np.where(cross_entropy_i > 0)]
            if cross_entropy is None:
                cross_entropy = cross_entropy_i
            else:
                cross_entropy = np.concatenate((cross_entropy, cross_entropy_i))
        elif adaptationVersion == 999:
            counter += 1
            print(counter)
            
            X, y = sequence_data(batchSize=batchSize,
                                            seq_len=seq_len,
                                            indexes=[subject],
                                            stride=stride,
                                            meanOf=mean,
                                            standardize=True,
                                            directory=baseDrive+"data/dbb",
                                            dataset='dbb-all',
                                            repetitions=validationRepetitions,
                                            number_of_classes=number_of_classes)
            
            
            model = load_model(getBestModel(str(subject), workingDirectory))
            result = model.predict(X, batch_size=batchSize)
            
            cross_entropy_i = -np.log(result) * y
            cross_entropy_i = cross_entropy_i[np.where(cross_entropy_i > 0)]
            if cross_entropy is None:
                cross_entropy = cross_entropy_i
            else:
                cross_entropy = np.concatenate((cross_entropy, cross_entropy_i))
        elif adaptationVersion > 1000:
            for target_subject in np.array(validationSubjects)[np.where(np.array(validationSubjects) == subject)[0][0]+1:]:
                counter += 1
                print(counter)
                
                X, y = sequence_data(batchSize=batchSize,
                                                seq_len=seq_len,
                                                indexes=[target_subject],
                                                stride=stride,
                                                meanOf=mean,
                                                standardize=True,
                                                directory=baseDrive+"data/dbb",
                                                dataset='dbb-all',
                                                repetitions=validationRepetitions,
                                                number_of_classes=number_of_classes)
                
                
                model = load_model(getBestModel(subject, workingDirectory))
                result = model.predict(X, batch_size=batchSize)
                
                cross_entropy_i = -np.log(result) * y
                cross_entropy_i = cross_entropy_i[np.where(cross_entropy_i > 0)]
                if cross_entropy is None:
                    cross_entropy = cross_entropy_i
                else:
                    cross_entropy = np.concatenate((cross_entropy, cross_entropy_i))
    
    np.save(output, cross_entropy)
    return cross_entropy

def generalScenario(validation, training, adaptationVersion):
    for trial in range(0, 1):
        if validation == 'intra-session-separated-metric':
            batchSize = 520
            
            allNumOfValidationSamples = 1 * 8 * 5 * ((1000-mean+1-seq_len)//stride + 1)
            validationStepsPerEpoch = allNumOfValidationSamples // batchSize
            allNumOfTrainingSamples = 1 * 8 * 5 * ((1000-mean+1-seq_len)//stride + 1)
            trainingStepsPerEpoch = allNumOfTrainingSamples // batchSize
            directory = baseDrive+"data/dbb"
            workingDirectory = baseDrive+'data/intra-session-separated-metric/dbb/'
            trainingUsers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
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
                        totalGestures=8,
                        totalRepetitions=5,
                        directory=directory,
                        testUser=testUser,
                        workingDirectory=workingDirectory,
                        trainingDataset='dbb-all',
                        testDataset='dbb-all',
                        trainingRepetitions=trainingRepetitions,
                        testRepetitions=testRepetitions,
                        number_of_classes=number_of_classes,
                        saveCheckpoints=True)
        elif validation == 'inter-subject-split':
            directory = baseDrive+"data/dbb"
            workingDirectory = baseDrive+'data/inter-subject-correct-split/dbb/'
            trainingUsers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            trainingRepetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            testRepetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for testUser in trainingUsers:
                testUsers = [testUser]
                if training == 'pre-training':
                    batchSize = 1040
                    
                    currentTrainingUsers = trainingUsers.copy()
                    currentTrainingUsers.remove(testUser)
                    allNumOfTrainingSamples = 9 * 8 * 10 * ((1000-mean+1-seq_len)//stride + 1)
                    trainingStepsPerEpoch = allNumOfTrainingSamples // batchSize
                    allNumOfValidationSamples = 1 * 8 * 10 * ((1000-mean+1-seq_len)//stride + 1)
                    validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                    preTrainingModel(trainingUsers=currentTrainingUsers,
                        testUsers=testUsers,
                        allNumOfTrainingSamples=allNumOfTrainingSamples,
                        trainingStepsPerEpoch=trainingStepsPerEpoch,
                        allNumOfValidationSamples=allNumOfValidationSamples,
                        validationStepsPerEpoch=validationStepsPerEpoch,
                        amountOfRepetitions=10,
                        amountOfGestures=8,
                        preTrainingNumOfEpochs=preTrainingNumOfEpochs,
                        trial=trial,
                        batchSize=batchSize,
                        totalGestures=8,
                        totalRepetitions=10,
                        directory=directory,
                        testUser=testUser,
                        workingDirectory=workingDirectory,
                        trainingDataset='dbb-all',
                        testDataset='dbb-all',
                        trainingRepetitions=trainingRepetitions,
                        testRepetitions=testRepetitions,
                        number_of_classes=number_of_classes,
                        saveCheckpoints=True)
                elif training == 'fine-tuning':
                    batchSize = 520
                    
                    trainingRepetitions = [1, 3, 5, 7, 9]
                    testRepetitions = [2, 4, 6, 8, 10]
                    fineTuneUsers = [testUser]
                    allNumOfFineTuningSamples = 1 * 8 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                    fineTuningStepsPerEpoch = allNumOfFineTuningSamples // batchSize
                    allNumOfValidationSamples = 1 * 8 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                    validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                    adaptModel(fineTuneUsers=fineTuneUsers,
                        testUsers=testUsers,
                        allNumOfFineTuningSamples=allNumOfFineTuningSamples,
                        fineTuningStepsPerEpoch=fineTuningStepsPerEpoch,
                        allNumOfValidationSamples=allNumOfValidationSamples,
                        validationStepsPerEpoch=validationStepsPerEpoch,
                        amountOfRepetitions=5,
                        amountOfGestures=8,
                        numberOfFineTuningEpochs=numberOfFineTuningEpochs,
                        trial=trial,
                        batchSize=batchSize,
                        totalGestures=8,
                        totalRepetitions=5,
                        directory=directory,
                        testUser=testUser,
                        workingDirectory=workingDirectory,
                        trainingDataset='dbb-all',
                        testDataset='dbb-all',
                        trainingRepetitions=trainingRepetitions,
                        testRepetitions=testRepetitions,
                        number_of_classes=number_of_classes,
                        adaptationVersion=adaptationVersion)
        elif validation == 'inter-session-split':
            trainingUsers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            testUsers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            fineTuneUsers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            trainingRepetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            testRepetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            directory = baseDrive+"data/dbb"
            workingDirectory = baseDrive+'data/inter-session-correct-split/dbb/'
            testUser='2ndSession'
            if training == 'pre-training':
                batchSize = 1040
                
                allNumOfTrainingSamples = 10 * 8 * 10 * ((1000-mean+1-seq_len)//stride + 1)
                trainingStepsPerEpoch = allNumOfTrainingSamples // batchSize
                allNumOfValidationSamples = 10 * 8 * 10 * ((1000-mean+1-seq_len)//stride + 1)
                validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                preTrainingModel(trainingUsers=trainingUsers,
                    testUsers=testUsers,
                    allNumOfTrainingSamples=allNumOfTrainingSamples,
                    trainingStepsPerEpoch=trainingStepsPerEpoch,
                    allNumOfValidationSamples=allNumOfValidationSamples,
                    validationStepsPerEpoch=validationStepsPerEpoch,
                    amountOfRepetitions=10,
                    amountOfGestures=8,
                    preTrainingNumOfEpochs=preTrainingNumOfEpochs,
                    trial=trial,
                    batchSize=batchSize,
                    totalGestures=8,
                    totalRepetitions=10,
                    directory=directory,
                    testUser=testUser,
                    workingDirectory=workingDirectory,
                    trainingDataset='dbb-all',
                    testDataset='dbb-all',
                    trainingRepetitions=trainingRepetitions,
                    testRepetitions=testRepetitions,
                    number_of_classes=number_of_classes,
                    saveCheckpoints=True)
            elif training == 'fine-tuning':
                batchSize = 2600
                
                allNumOfFineTuningSamples = 10 * 8 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                allNumOfValidationSamples = 10 * 8 * 5 * ((1000-mean+1-seq_len)//stride + 1)
                fineTuningStepsPerEpoch = allNumOfValidationSamples // batchSize
                validationStepsPerEpoch = allNumOfValidationSamples // batchSize
                fineTuneRepetitions = [1, 3, 5, 7, 9]
                testRepetitions = [2, 4, 6, 8, 10]
                adaptModel(fineTuneUsers=fineTuneUsers,
                    testUsers=testUsers,
                    allNumOfFineTuningSamples=allNumOfFineTuningSamples,
                    fineTuningStepsPerEpoch=fineTuningStepsPerEpoch,
                    allNumOfValidationSamples=allNumOfValidationSamples,
                    validationStepsPerEpoch=validationStepsPerEpoch,
                    amountOfRepetitions=5,
                    amountOfGestures=8,
                    numberOfFineTuningEpochs=numberOfFineTuningEpochs,
                    trial=trial,
                    batchSize=batchSize,
                    totalGestures=8,
                    totalRepetitions=5,
                    directory=directory,
                    testUser=testUser,
                    workingDirectory=workingDirectory,
                    trainingDataset='dbb-all',
                    testDataset='dbb-all',
                    trainingRepetitions=fineTuneRepetitions,
                    testRepetitions=testRepetitions,
                    number_of_classes=number_of_classes,
                    adaptationVersion=adaptationVersion)

# Istvan Titan V

'''
#a source-data absent divergence abrahoz kell intra-session-separated model ujra (20191203 kedd delutan):
generalScenario(validation='intra-session-separated-metric', training='pre-training', adaptationVersion=5555)
# kesz.

#kedd:
intra_session_residual_distribution()

#szerda:
inter_session_residual_distribution()


#szerda:
inter_subject_residual_distribution()



#pentek delben:
generalScenario(validation='inter-subject-split', training='pre-training', adaptationVersion=9999)
#done.

#pentek este:
#linear:
generalScenario(validation='inter-subject-split', training='fine-tuning', adaptationVersion=1000)
#deep-relu:
generalScenario(validation='inter-subject-split', training='fine-tuning', adaptationVersion=31)
# done.


#szombat:
inter_subject_residual_distribution(adaptationVersion=1000)
inter_subject_residual_distribution(adaptationVersion=31)
# done.

#20191209 hetfo este:
inter_session_residual_distribution(adaptationVersion=999)
inter_subject_residual_distribution(adaptationVersion=999)
# done.
'''
