import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data
import mlp.metrics as metrics
import mlp.functions as func
import mlp.network as nn


np.random.seed(1337)


df = pd.read_csv("iris_num.csv", names=['pw', 'pl', 'sw', 'sl', 'label'])

# convert features to numpy array
feats = df.iloc[:, :-1].values.astype(np.float32)
n_feats = feats.shape[1]

labels = df.iloc[:, -1]

"""
Our data is numerical, and the distribution is not centered around 0.
It would be better to standardize the data to avoid activation function saturation.
We should remember the mean and standard deviation values to use them on unseen data too.
"""
X, means, stds = data.feat_standardize(feats)

# one-hot encode labels
Y = data.int_to_one_hot(labels)

# shuffle the data
X, Y = data.shuffle(X, Y)

# separate data into training and test sets
(x_train, y_train), (x_test, y_test) = data.split(X, Y, ratio=0.7)

print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:",  x_test.shape,  "y_test:",  y_test.shape)


# network architecture
net = nn.network([n_feats, 8, 3], [func.tanh(), func.softmax()], func.categorical_crossentropy())


# training
history = net.fit(x_train, y_train, lr=0.001, n_epochs=150, batch_size=32, val_ratio=0.2)
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.show()


# evaluation
y_test_pred = net.predict(x_test)
acc = metrics.accuracy()(y_test, y_test_pred)
print("----------")
print(f"accuracy on test set ({len(x_test)} samples): {acc:.2f}")