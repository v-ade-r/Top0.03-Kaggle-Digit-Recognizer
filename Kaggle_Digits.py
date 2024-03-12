import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

pd.set_option('display.max_columns', None)
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)

tf.random.set_seed(33)
random_seed = 33

train_df = pd.read_csv('train_Digits.csv')
test_df = pd.read_csv('test_Digits.csv')

y = train_df['label']
train_df.drop('label', axis=1, inplace=True)

# 1.1 Normalization ---------------------------------------------------------------------------------------------------
train = train_df / 255.0
test = test_df / 255.0

# 1.2 Label encoding ---------------------------------------------------------------------------------------------------
y = to_categorical(y, num_classes=10)

# 1.3 Reshaping --------------------------------------------------------------------------------------------------------
train = train.values.reshape(-1,28,28,1)
X_test = test.values.reshape(-1,28,28,1)
results = np.zeros((X_test.shape[0], 10))

# 1.4 Data augmentation - preparation ----------------------------------------------------------------------------------
datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

# 1.5 Creating training and validation sets ----------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.1, random_state=random_seed)

# 1.6 Building CNNs ----------------------------------------------------------------------------------------------------
nets = 25
model = [0] * nets

for j in range(nets):
    model[j] = Sequential()

    """A convolution with stride 2 replaces pooling layers. These become learnable pooling layers."""
    model[j].add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size=4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=['accuracy'])


# 1.7 Setting annealer -------------------------------------------------------------------------------------------------
# Decreasing learning rate at each epoch
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# 1.8 Training ---------------------------------------------------------------------------------------------------------
history = [0] * nets
epochs = 45

for i in range(nets):
    history[i] = model[i].fit(datagen.flow(X_train, y_train, batch_size=64), epochs=epochs,
                                     validation_data=(X_val, y_val), callbacks=[annealer])

    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(i + 1, epochs,
                                                                                                max(history[i].history['accuracy']),
                                                                                                max(history[i].history['val_accuracy'])))

# 1.9 Predictions and submission ---------------------------------------------------------------------------------------
results = np.zeros((X_test.shape[0], 10))
print(results[:10])
for i in range(nets):
    results = results + model[i].predict(X_test)
print(results[:10])
results = np.argmax(results, axis=1)
print(results[:10])
sub = pd.DataFrame({'ImageId': pd.Series(range(1,28001)),
                        'Label': results})
sub.to_csv("Kaggle_mnist_cnn1.csv",index=False)

"""This code finally is almost a copy of this amazing notebook:
https://www.kaggle.com/code/cdeotte/25-million-images-0-99757-mnist/notebook.
After understanding the logic behind the tests leading to this code, I tested by myself a few things like other data 
augmentation, other layer compositions and parameters but I didn't get better results. The only improvement I made is an
increase of number of nets in ensemble. In the future I will increase it even further with parallel increase of number 
of epochs and maybe I will try a few more callbacks. No point in posting a way worse solution, so I am posting this as 
a blueprint for most nuanced future problems"""
