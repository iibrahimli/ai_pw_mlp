import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlp.data as data
import mlp.metrics as metrics
import mlp.functions as func
import mlp.network as nn


np.random.seed(123)


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
(x_train, y_train), (x_test, y_test) = data.split(X, Y, ratio=0.8)

# integer labels
y_test_int = np.argmax(y_test, axis=1)

print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:",  x_test.shape,  "y_test:",  y_test.shape)
print()


# network architecture
net = nn.network([n_feats, 8, 3], [func.tanh(), func.softmax()], func.categorical_crossentropy())


# training
acc = metrics.accuracy()
history = net.fit(
    x_train, y_train,
    lr=(0.1, 750, 0.5),              # learning rate annealing
    n_epochs=2500,
    batch_size=64,
    val_ratio=0.2,                   # 20% of training data used for validation
    metrics=[acc],                   # evaluate accuracy for each epoch
    print_stats=250
)


# evaluation
y_test_pred = net.predict(x_test)
y_test_pred_int = np.argmax(y_test_pred, axis=1)
test_acc = acc(y_test_int, y_test_pred_int)
print()
print("----------")
print(f"accuracy on test set ({len(x_test)} samples): {test_acc:.2f}")



# plot training
print()
print("Uncomment lines 76-90 in main.py to plot training graphs")
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ax1.plot(history['train_loss'], label='train_loss')
# ax1.plot(history['val_loss'], label='val_loss')
# ax1.legend()
# ax1.set_xlabel('epochs')
# ax1.set_ylabel('loss')

# ax2.plot(history['train_accuracy'], label='train_accuracy')
# ax2.plot(history['val_accuracy'], label='val_accuracy')
# ax2.legend()
# ax2.set_xlabel('epochs')
# ax2.set_ylabel('accuracy')

# plt.show()