import os
import pandas as pd
import numpy as np

import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

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

class DesignModel:
    def mount_model_gru(self, max_length):
        x_input = Input(shape=(max_length,))
        emb = Embedding(21, 256, input_length=max_length)(x_input)
        bi_rnn = Bidirectional(GRU(256, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)
        x = Dropout(0.3)(bi_rnn)
        # softmax classifier
        x_output = Dense(2, activation='softmax')(x)

        model1 = Model(inputs=x_input, outputs=x_output)
        model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #model1.summary()()
        
        return model1
        
    def mount_model_conv(self, max_length):
        embedding_dim = 8

        # create the model
        model = Sequential()
        model.add(Embedding(21, embedding_dim, input_length=max_length))
        model.add(Conv1D(filters=64, strides=1, kernel_size=6, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, strides=2, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
        
    def mount_model_conv2d(self, max_length, input_shape):
        embedding_dim = 32

        # create the model
        model = Sequential()
        #model.add(Embedding(21, embedding_dim, batch_size=bs, input_length=max_length))
        model.add(Conv2D(filters=32, strides=(1,1), kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2) ))
        model.add(Conv2D(filters=64, strides=(1,1), kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2) ))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.summary()()
        
        return model    
    
    def mount_model_lstm(self, max_length):
        x_input = Input(shape=(max_length,))
        emb = Embedding(21, 256, input_length=max_length)(x_input)
        bi_rnn = Bidirectional(LSTM(256, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01) ) )(emb)
        x = Dropout(0.3)(bi_rnn)
        # softmax classifier
        x_output = Dense(2, activation='softmax')(x)

        model1 = Model(inputs=x_input, outputs=x_output)
        model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #model1.summary()()
        
        return model1
    
    def mount_model_3layers_conv(self, max_length):
        
        model = Sequential()
        model.add(Embedding(21, 256, input_length=max_length))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=128, strides=1, kernel_size=12, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, strides=2, kernel_size=6, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, strides=3, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        #model.add(LSTM(100))  
        model.add(Flatten())
        model.add(Dense(2, activation='softmax')) 
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #model.summary()()
        
        return model 
            
    def mount_model_lstm_v2(self, max_length):
        lr=0.001
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=42)  # weights randomly between -0.05 and 0.05
        dropoutfract=0.1
        emb_dim=100
        
        model = Sequential()
        x_input = Input(shape=(max_length,))
        emb = Embedding(21, emb_dim, input_length=max_length)(x_input)
        #model.add(InputLayer(input_tensor=embl))
        
        l1= LSTM(emb_dim, kernel_initializer=weight_init,  return_sequences=True, recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), dropout=dropoutfract*1) (emb)
        l21= GRU(emb_dim, kernel_initializer=weight_init,  return_sequences=True, recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), dropout=dropoutfract*2) (l1)
        #l3= LSTM(emb_dim, kernel_initializer=weight_init,  return_sequences=True, dropout=dropoutfract*3) (l2)
        d1=Dropout(0.25) (l21)
        x = Flatten()(d1)
        x_output=Dense(2, activation='softmax' ) (x)
        model = Model(inputs=x_input, outputs=x_output)
        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        #model.summary()()
        
        return model
    
    def mount_model_lstm_conv(self, max_length):
        
        model = Sequential()
        model.add(Embedding(21, 128, input_length=max_length))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=64, strides=1, kernel_size=6, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, strides=2, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(256, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01) )) 
        #model.add(GRU(256, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01) ))  
        model.add(Dense(2, activation='softmax')) 
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #model.summary()()
        
        return model
        
