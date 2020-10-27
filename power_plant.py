import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.style as style


ds = pd.read_csv('power_plant.csv')
ds = ds.reindex(np.random.permutation(ds.index))

train = ds

s = StandardScaler()
for x in train.columns:
    if x != 'EP':
        train[x] = s.fit_transform(train[x].values.reshape(-1, 1)).astype('float64')


train_features = train.drop('EP',axis=1)
train_label = train.pop('EP')


input_dim = train_features.shape[1]
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_dim = input_dim, activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.GlorotNormal()))
model.add(keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.GlorotNormal()))
model.add(keras.layers.Dense(16,  activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.GlorotNormal()))
model.add(keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.GlorotNormal()))


model.compile(optimizer='rmsprop', loss='mse',
              metrics =tf.keras.metrics.RootMeanSquaredError())
#
#
history = model.fit(train_features, train_label, epochs=50, validation_split=0.6)

results = model.evaluate(train_features, train_label)
print('\nLoss, RMSE: \n',(results))


style.use('dark_background')
pd.DataFrame(history.history).plot(figsize=(11, 7),linewidth=4)
plt.title('Root Mean Squared Error',fontsize=14, fontweight='bold')
plt.xlabel('Epochs',fontsize=13)
plt.ylabel('Metrics',fontsize=13)
plt.show()  