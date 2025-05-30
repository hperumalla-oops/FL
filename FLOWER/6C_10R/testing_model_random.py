import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler  # Optional
from sklearn.metrics import accuracy_score, classification_report  # For evaluation

df = pd.read_csv(r'C:\Users\hperu\OneDrive\Desktop\fl\FLOWER\1\mitbih_test.csv')

X_test = df.iloc[:, :-1].values
y_test = df.iloc[:, -1].values

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
