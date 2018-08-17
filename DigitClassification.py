import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd;
import numpy as np;

batch_size = 28
num_classes = 10
epochs = 12

trainDf = pd.read_csv("train.csv")
testDf = pd.read_csv("test.csv")
X = trainDf.iloc[:,1:].values
Y = trainDf.iloc[:,0].values


# The data, shuffled and split between train and test sets:
 # Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)



# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train[0]
#reshape it for CNN
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Initialize the model
model = Sequential()

# Create the model with two 32 convolution filters -> pooling layer -> two 64 conv filters -> pooling layer -> flattening -> fully conncted layer 
model.add(Conv2D(32, (3, 3), padding='same',
                  input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(28, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(36, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# if not data_augmentation:
model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              shuffle=True)

test_img = X_train[15];
test_img = np.expand_dims(test_img, axis = 0)
result = model.predict(test_img)
print(np.argmax(result))

    