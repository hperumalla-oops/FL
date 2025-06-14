import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("1/mitbih_train.csv")  # You can also try with ptbdb files

# Shuffle and split the dataset into two equal parts (IID)
client1_data, client2_data = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)

# Save to new CSV files
client1_data.to_csv("client1_data.csv", index=False)
client2_data.to_csv("client2_data.csv", index=False)

print("Partitioning done. Files saved as client1_data.csv and client2_data.csv")
