import os
import pandas as pd
import numpy as np

import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random

from collections import Counter

from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Embedding, Bidirectional, LSTM, GRU, GlobalMaxPooling1D, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import itertools

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from unknown_sequence_classification.pipelines.modules.model_design import DesignModel

class RunningModel:
    def __init__(self):
        self.design = DesignModel()
    
    def create_dict(self):
        codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
         
        char_dict = {}
        for index, val in enumerate(codes):
            char_dict[val] = index+1

        return char_dict
        
    def integer_encoding(self, data):
        """
        - Encodes code sequence to integer values.
        - 20 common amino acids are taken into consideration
          and rest 4 are categorized as 0.
        """
        
        char_dict = self.create_dict()
        encode_list = []
        for row in data['sequence'].values:
            row_encode = []
            for code in row:
              row_encode.append(char_dict.get(code, 0))
            encode_list.append(np.array(row_encode))
      
        return encode_list
        
    def prepare_data(self, df_clean, voc):
        #df=pd.read_csv('../filtered_mutation_list.tsv', sep='\t', header=0)
        if(voc!='all'):
            df_clean=df_clean[ df_clean['voc']==voc ]
        df=df_clean
        pieces=[]
        auxdf=df[ (df['class']==1) ]
        chosen=random.sample(list(auxdf.index), 3500)
        pieces.append(df.iloc[chosen, :])
        
        auxdf=df[ (df['class']==0) ]
        chosen=random.sample(list(auxdf.index), 3500)
        pieces.append(df.iloc[chosen, :])
            
        dfv=pd.concat( pieces )
        
        train, validate, test = np.split( dfv.sample(frac=1, random_state=42), [int(.6*len(dfv)), int(.8*len(dfv))])
        
        train_encode = self.integer_encoding(train) 
        val_encode = self.integer_encoding(validate) 
        test_encode = self.integer_encoding(test)
        
        max_length = 1436
        train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
        val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
        test_pad = pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')
        
        train_ohe = to_categorical(train_pad)
        val_ohe = to_categorical(val_pad)
        test_ohe = to_categorical(test_pad)
        
        le = LabelEncoder()
        y_train_le = le.fit_transform(train['class'])
        y_val_le = le.transform(validate['class'])
        y_test_le = le.transform(test['class'])
        
        y_train = to_categorical(y_train_le)
        y_val = to_categorical(y_val_le)
        y_test = to_categorical(y_test_le)
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        
        bs=32
        max_length = 1436
        
        out={
            'bs': bs,
            'max_length': max_length,
            'train_pad': train_pad,
            'val_pad': val_pad,
            'test_pad': test_pad,
            'train_ohe': train_ohe,
            'val_ohe': val_ohe,
            'test_ohe': test_ohe,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
            }
        
        return out
        
    def run_bigru(self, data_parameters):
        bs= data_parameters['bs'],
        max_length= data_parameters['max_length'],
        train_pad= data_parameters['train_pad'],
        val_pad= data_parameters['val_pad'],
        test_pad= data_parameters['test_pad'],
        train_ohe= data_parameters['train_ohe'],
        val_ohe= data_parameters['val_ohe'],
        test_ohe= data_parameters['test_ohe'],
        y_train= data_parameters['y_train'],
        y_val= data_parameters['y_val'],
        y_test= data_parameters['y_test'],
        
        bs= bs[0]
        max_length= max_length[0]
        train_pad= train_pad[0]
        val_pad= val_pad[0]
        test_pad= test_pad[0]
        train_ohe= train_ohe[0]
        val_ohe= val_ohe[0]
        test_ohe= test_ohe[0]
        y_train= y_train[0]
        y_val= y_val[0]
        y_test= y_test[0]
        
        print("----------------", bs, y_train)
        results={}
        model1=self.design.mount_model_gru(max_length)
        model1.fit(
            train_pad, y_train,
            epochs=50, batch_size=bs,
            validation_data=(val_pad, y_val)
        )
        train_pred = model1.predict(train_pad)
        test_pred = model1.predict(test_pad)
        
        results['method']='bigru'
        results['accuracy']=accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        results['precision']=precision_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        
        #plot_model(model1, rankdir='LR', to_file= option+'.png')
        
        return results
        
    def run_bilstm(self, data_parameters):
        bs= data_parameters['bs'],
        max_length= data_parameters['max_length'],
        train_pad= data_parameters['train_pad'],
        val_pad= data_parameters['val_pad'],
        test_pad= data_parameters['test_pad'],
        train_ohe= data_parameters['train_ohe'],
        val_ohe= data_parameters['val_ohe'],
        test_ohe= data_parameters['test_ohe'],
        y_train= data_parameters['y_train'],
        y_val= data_parameters['y_val'],
        y_test= data_parameters['y_test'],
        
        bs= bs[0]
        max_length= max_length[0]
        train_pad= train_pad[0]
        val_pad= val_pad[0]
        test_pad= test_pad[0]
        train_ohe= train_ohe[0]
        val_ohe= val_ohe[0]
        test_ohe= test_ohe[0]
        y_train= y_train[0]
        y_val= y_val[0]
        y_test= y_test[0]
        
        results={}
        model1=self.design.mount_model_lstm(max_length)
        model1.fit(
            train_pad, y_train,
            epochs=50, batch_size=bs,
            validation_data=(val_pad, y_val)
        )
        train_pred = model1.predict(train_pad)
        test_pred = model1.predict(test_pad)
        
        results['method']='bilstm'
        results['accuracy']=accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        results['precision']=precision_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        
        #plot_model(model1, rankdir='LR', to_file= option+'.png')
        
        return results
        
    def run_conv1d(self, data_parameters):
        bs= data_parameters['bs'],
        max_length= data_parameters['max_length'],
        train_pad= data_parameters['train_pad'],
        val_pad= data_parameters['val_pad'],
        test_pad= data_parameters['test_pad'],
        train_ohe= data_parameters['train_ohe'],
        val_ohe= data_parameters['val_ohe'],
        test_ohe= data_parameters['test_ohe'],
        y_train= data_parameters['y_train'],
        y_val= data_parameters['y_val'],
        y_test= data_parameters['y_test'],
        
        bs= bs[0]
        max_length= max_length[0]
        train_pad= train_pad[0]
        val_pad= val_pad[0]
        test_pad= test_pad[0]
        train_ohe= train_ohe[0]
        val_ohe= val_ohe[0]
        test_ohe= test_ohe[0]
        y_train= y_train[0]
        y_val= y_val[0]
        y_test= y_test[0]
        
        results={}
        model1=self.design.mount_model_conv(max_length)
        model1.fit(train_pad, y_train, validation_data=(test_pad, y_test), epochs=50, batch_size=bs)
        train_pred = model1.predict(train_pad)
        test_pred = model1.predict(test_pad)
        
        results['method']='conv1d'
        results['accuracy']=accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        results['precision']=precision_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        
        #plot_model(model1, rankdir='LR', to_file= option+'.png')
        
        return results
        
    def run_conv2d(self, data_parameters):
        bs= data_parameters['bs'],
        max_length= data_parameters['max_length'],
        train_pad= data_parameters['train_pad'],
        val_pad= data_parameters['val_pad'],
        test_pad= data_parameters['test_pad'],
        train_ohe= data_parameters['train_ohe'],
        val_ohe= data_parameters['val_ohe'],
        test_ohe= data_parameters['test_ohe'],
        y_train= data_parameters['y_train'],
        y_val= data_parameters['y_val'],
        y_test= data_parameters['y_test'],
        
        bs= bs[0]
        max_length= max_length[0]
        train_pad= train_pad[0]
        val_pad= val_pad[0]
        test_pad= test_pad[0]
        train_ohe= train_ohe[0]
        val_ohe= val_ohe[0]
        test_ohe= test_ohe[0]
        y_train= y_train[0]
        y_val= y_val[0]
        y_test= y_test[0]
        
        results={}
        col=21
        row=max_length
        X_train = train_ohe.reshape(train_ohe.shape[0], row, col, 1)
        X_test = test_ohe.reshape(test_ohe.shape[0], row, col, 1)
        input_shape = (row, col, 1)
        
        model1=self.design.mount_model_conv2d(max_length, input_shape)
        model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=bs)
        train_pred = model1.predict(X_train, batch_size=bs)
        test_pred = model1.predict(X_test, batch_size=bs)
        
        results['method']='conv2d'
        results['accuracy']=accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        results['precision']=precision_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        
        #plot_model(model1, rankdir='LR', to_file= option+'.png')
        
        return results
        
    def run_lstmgru(self, data_parameters):
        bs= data_parameters['bs'],
        max_length= data_parameters['max_length'],
        train_pad= data_parameters['train_pad'],
        val_pad= data_parameters['val_pad'],
        test_pad= data_parameters['test_pad'],
        train_ohe= data_parameters['train_ohe'],
        val_ohe= data_parameters['val_ohe'],
        test_ohe= data_parameters['test_ohe'],
        y_train= data_parameters['y_train'],
        y_val= data_parameters['y_val'],
        y_test= data_parameters['y_test'],
        
        bs= bs[0]
        max_length= max_length[0]
        train_pad= train_pad[0]
        val_pad= val_pad[0]
        test_pad= test_pad[0]
        train_ohe= train_ohe[0]
        val_ohe= val_ohe[0]
        test_ohe= test_ohe[0]
        y_train= y_train[0]
        y_val= y_val[0]
        y_test= y_test[0]
        
        results={}
        model1=self.design.mount_model_lstm_v2(max_length)
        model1.fit(
            train_pad, y_train,
            epochs=50, batch_size=bs,
            validation_data=(val_pad, y_val)
        )
        train_pred = model1.predict(train_pad)
        test_pred = model1.predict(test_pad)
        
        results['method']='lstmgru'
        results['accuracy']=accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        results['precision']=precision_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        
        #plot_model(model1, rankdir='LR', to_file= option+'.png')
        
        return results
        
    def run_conv3layer(self, data_parameters):
        bs= data_parameters['bs'],
        max_length= data_parameters['max_length'],
        train_pad= data_parameters['train_pad'],
        val_pad= data_parameters['val_pad'],
        test_pad= data_parameters['test_pad'],
        train_ohe= data_parameters['train_ohe'],
        val_ohe= data_parameters['val_ohe'],
        test_ohe= data_parameters['test_ohe'],
        y_train= data_parameters['y_train'],
        y_val= data_parameters['y_val'],
        y_test= data_parameters['y_test'],
        
        bs= bs[0]
        max_length= max_length[0]
        train_pad= train_pad[0]
        val_pad= val_pad[0]
        test_pad= test_pad[0]
        train_ohe= train_ohe[0]
        val_ohe= val_ohe[0]
        test_ohe= test_ohe[0]
        y_train= y_train[0]
        y_val= y_val[0]
        y_test= y_test[0]
        
        results={}
        model1=self.design.mount_model_3layers_conv(max_length)
        model1.fit(
            train_pad, y_train,
            epochs=50, batch_size=bs,
            validation_data=(val_pad, y_val)
        )
        train_pred = model1.predict(train_pad)
        test_pred = model1.predict(test_pad)
        
        results['method']='conv3layer'
        results['accuracy']=accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        results['precision']=precision_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        
        #plot_model(model1, rankdir='LR', to_file= option+'.png')
        
        return results
        
    def run_lstmconv(self, data_parameters):
        bs= data_parameters['bs'],
        max_length= data_parameters['max_length'],
        train_pad= data_parameters['train_pad'],
        val_pad= data_parameters['val_pad'],
        test_pad= data_parameters['test_pad'],
        train_ohe= data_parameters['train_ohe'],
        val_ohe= data_parameters['val_ohe'],
        test_ohe= data_parameters['test_ohe'],
        y_train= data_parameters['y_train'],
        y_val= data_parameters['y_val'],
        y_test= data_parameters['y_test'],
        
        bs= bs[0]
        max_length= max_length[0]
        train_pad= train_pad[0]
        val_pad= val_pad[0]
        test_pad= test_pad[0]
        train_ohe= train_ohe[0]
        val_ohe= val_ohe[0]
        test_ohe= test_ohe[0]
        y_train= y_train[0]
        y_val= y_val[0]
        y_test= y_test[0]
        
        results={}
        model1=self.design.mount_model_lstm_conv(max_length)
        model1.fit(
            train_pad, y_train,
            epochs=50, batch_size=bs,
            validation_data=(val_pad, y_val)
        )
        train_pred = model1.predict(train_pad)
        test_pred = model1.predict(test_pad)
        
        results['method']='lstmconv'
        results['accuracy']=accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        results['precision']=precision_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))
        
        #plot_model(model1, rankdir='LR', to_file= option+'.png')
        
        return results
    
