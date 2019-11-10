import numpy as np
import pandas as pd

import data
import mlp.functions as func
import mlp.network as nn


np.random.seed(1337)

# --- read and preprocess data ---

df = pd.read_csv("iris_num.csv", names=['pw', 'pl', 'sw', 'sl', 'label'])

# convert features to numpy array
feats = df.iloc[:, :4].values.astype(np.float32)
n_feats = feats.shape[1]
print(f"features: {feats.shape} {feats.dtype}")

"""
Our data is numerical, and the distribution is not centered around 0.
It would be better to standardize the data to avoid activation function saturation.
We should remember the mean and standard deviation values to use them on unseen data too.
"""
# standardize the data and save parameters
feats, means, stds = data.feat_standardize(feats)


# --- model ---
net = nn.network([n_feats, 8, 1], [func.tanh(), func.sigmoid()])

# print("weights:")
# for i, w in net.w.items():
#     print(f"{i}: {w.shape}")

# print("biases:")
# for i, b in net.b.items():
#     print(f"{i}: {b.shape}")

print(net.predict(feats[:10]))


# --- train ---


# --- evaluate ---
