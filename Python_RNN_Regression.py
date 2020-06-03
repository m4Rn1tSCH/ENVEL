# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:54:19 2020

@author: bill-
"""
#import required packages
import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np
from datetime import datetime as dt
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from flask import Flask

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column, data
from tensorflow.keras import Model, layers, regularizers

#IMPORTED CUSTOM FUNCTION
#generates a CSV for daily/weekly/monthly account throughput; expenses and income
#from Python_spending_report_csv_function import spending_report
#contains the connection script
#from Python_SQL_connection import execute_read_query, create_connection, close_connection
#contains all credentials
import PostgreSQL_credentials as acc
#csv export with optional append-mode
from Python_CSV_export_function import csv_export
from Python_eda_ai import df_encoder

'''
                    RNN Regression
single-step and multi-step model for a recurrent neural network
'''
print("tensorflow regression running...")
bank_df = df_encoder(rng=4)
dataset = bank_df.copy()
print(dataset.head())
print(dataset.tail())
print(dataset.isna().sum())
sns.pairplot(bank_df[['amount', 'amount_mean_lag7', 'amount_std_lag7']])
#%%
# setting label and features (the df itself here)
model_label = dataset.pop('amount_mean_lag7')
model_label.astype('int64')

# EAGER EXECUTION NEEDS TO BE ENABLED HERE
# features and model labels passed as tuple
tensor_ds = tf.data.Dataset.from_tensor_slices((dataset.values, model_label.values))
for feat, targ in tensor_ds.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

train_dataset = tensor_ds.shuffle(len(bank_df)).batch(2)

'''
                Recurring Neural Network
-LSTM cell in sequential network
'''
    # Test of a RNN
    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))
    model.summary()

# Simple RNN
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(10))

    model.summary()

    encoder_vocab = 1000
    decoder_vocab = 2000

    encoder_input = layers.Input(shape=(None, ))
    encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

    # Return states in addition to output
    output, state_h, state_c = layers.LSTM(
        64, return_state=True, name='encoder')(encoder_embedded)
    encoder_state = [state_h, state_c]

    decoder_input = layers.Input(shape=(None, ))
    decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_output = layers.LSTM(
        64, name='decoder')(decoder_embedded, initial_state=encoder_state)
    output = layers.Dense(10)(decoder_output)

    model = tf.keras.Model([encoder_input, decoder_input], output)
    model.summary()

    lstm_layer = layers.LSTM(64, stateful=True)
    for s in sub_sequences:
      output = lstm_layer(s)
    #%%
    # Model resets weights of layers
    paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
    paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
    paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

    lstm_layer = layers.LSTM(64, stateful=True)
    output = lstm_layer(paragraph1)
    output = lstm_layer(paragraph2)
    output = lstm_layer(paragraph3)

    # reset_states() will reset the cached state to the original initial_state.
    # If no initial_state was provided, zero-states will be used by default.
    lstm_layer.reset_states()
    #%%
    # Model reuses states/ weights
    paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
    paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
    paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

    lstm_layer = layers.LSTM(64, stateful=True)
    output = lstm_layer(paragraph1)
    output = lstm_layer(paragraph2)

    existing_state = lstm_layer.states

    new_lstm_layer = layers.LSTM(64)
    new_output = new_lstm_layer(paragraph3, initial_state=existing_state)
    #%%
    batch_size = 64
    # Each data batch is a tensor of shape (batch_size, num_feat, num_feat)
    #                                       (batch_size, 21, 21)
    # Each input sequence will be of size (21, 21) (height is treated like time).
    input_dim = 21

    units = 64
    output_size = 10  # labels are from 0 to 9

    # Build the RNN model
    def build_model(allow_cudnn_kernel=True):
        # CuDNN is only available at the layer level, and not at the cell level.
        # This means `LSTM(units)` will use the CuDNN kernel,
        # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
        if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
            lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
        else:
            # Wrapping an LSTMCell in an RNN layer will not use CuDNN.
            # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
            lstm_layer = tf.keras.layers.RNN(
                tf.keras.layers.LSTMCell(units),
                input_shape=(None, input_dim))

        model = tf.keras.models.Sequential([
            lstm_layer,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(output_size)])

        return model
    #%%
    model = build_model(allow_cudnn_kernel=False)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(dataset,
              batch_size=batch_size,
              epochs=5,
              verbose=2)
    print("Tensorflow regression finished...")
    return model