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
from Python_spending_report_csv_function import spending_report
#contains the connection script
from Python_SQL_connection import execute_read_query, create_connection, close_connection
#contains all credentials
import PostgreSQL_credentials as acc
#csv export with optional append-mode
from Python_CSV_export_function import csv_export

    # model training is finished
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_dir = os.getcwd()

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        verbose=1,
        save_weights_only=True,
        period=5)

    # Create a new model instance
    model = get_compiled_model()

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_dir.format(epoch=5))

    # Train the model with the new callback
    model.fit(train_images,
              train_labels,
              epochs=50,
              callbacks=[cp_callback],
              validation_data=(test_images,test_labels),
              verbose=0)

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