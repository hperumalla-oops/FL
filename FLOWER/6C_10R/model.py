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


from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler  # Optional
from sklearn.metrics import accuracy_score, classification_report  # For evaluation


def create_model(input_dim=187):
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
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    return model
    

def testting():
    df = pd.read_csv(r'C:\Users\hperu\OneDrive\Desktop\fl\FLOWER\1\mitbih_test.csv')

    X_test = df.iloc[:,:-1]  # or specify feature columns
    y_test = df.iloc[:,-1]

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    model = load_model('server_model.h5')

    predictions = model.predict(X_test_scaled)

    if predictions.shape[1] > 1:
        y_pred = np.argmax(predictions, axis=1)
    else:
        y_pred = (predictions > 0.5).astype(int).flatten()

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# testting()