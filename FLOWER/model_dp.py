import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, recall_score, accuracy_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from keras.optimizers import Adam

def create_model(input_dim=187, use_dp=False, l2_norm_clip=1.0, noise_multiplier=1.1, num_microbatches=32, learning_rate=0.01):
    model = Sequential()
    
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(5, activation='softmax'))
    
    
    if use_dp:
          optimizer = DPKerasSGDOptimizer(
              l2_norm_clip=l2_norm_clip,
              noise_multiplier=noise_multiplier,
              num_microbatches=num_microbatches,
              learning_rate=learning_rate,
          )
          loss = "binary_crossentropy"
    else:
        optimizer = SGD(learning_rate=learning_rate)
        loss = "binary_crossentropy"
    model.compile(optimizer=optimizer, 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    return model
    
