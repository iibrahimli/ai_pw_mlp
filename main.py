import numpy as np
import pandas as pd
import mlp


# --- read and preprocess data ---

df = pd.read_csv("iris_num.csv", names=['pw', 'pl', 'sw', 'sl', 'label'])

# convert features to numpy array
feats = df.iloc[:, :4].values.astype(np.float32)
print("data shape:", feats.shape)

# standardize the data by column
# feats, means, stds = standardize()


# --- model ---


# --- train ---


# --- evaluate ---
