from model import create_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

x_train, x_test, y_train, y_test = load_data(r"C:\Users\hperu\OneDrive\Desktop\fl\FLOWER\client2_data.csv")
m=create_model(input_dim=x_train.shape[1])

history = m.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])
print(history)

y_pred = np.argmax(m.predict(x_test), axis=1)
from sklearn.metrics import classification_report, recall_score, accuracy_score

print(classification_report(y_pred, y_test))
m.save("cleint2.h5")
