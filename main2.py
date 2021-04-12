from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
from keras import models
from keras.optimizers import Adam
from keras import layers
import random
from sklearn.decomposition import PCA


def data_preprocessing(X, Y, index):
    # scaling of data
    X_train, X_test, Y_train, Y_test = None, None, None, None
    if index is 1:
        Y[Y == 0] = -1
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
        X_train = X_train.reshape((len(X_train), 68 * 2))
        X_test = X_test.reshape((len(X_test), 68 * 2))

        Scaler = StandardScaler()
        Scaler.fit(X_train)
        X_train = Scaler.transform(X_train)
        X_test = Scaler.transform(X_test)
    if index is 2:
        # scaling of data
        indices = np.arange(len(Y))
        random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        y = np.zeros((len(Y), 5))
        for row in range(len(Y)):
            column = Y[row]
            y[row, column] = 1
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        Scaler = StandardScaler()
        Scaler.fit(X_train)
        X_train = Scaler.transform(X_train)
        X_test = Scaler.transform(X_test)

        pca = PCA(n_components=68)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    if index is 3:
        # scaling of data
        length = len(Y)
        indices = np.arange(length)
        random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        y = np.zeros((length, 5))
        for row in range(length):
            column = Y[row]
            y[row, column] = 1
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        Scaler = StandardScaler()
        Scaler.fit(X_train)
        X_train = Scaler.transform(X_train)
        X_test = Scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test


training_images = np.load('136_features_A1.npy')
gender_labels = np.load('Gender_labels_A1.npy')
tr_X, te_X, tr_Y, te_Y = data_preprocessing(training_images, gender_labels, 1)

# Task A1

model_A1 = SVC(C=0.35111917342151344, kernel='linear')  # Build model object.
model_A1.fit(tr_X,
             tr_Y)  # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_train = accuracy_score(model_A1.predict(tr_X), tr_Y)  # train accuracy
acc_A1_test = accuracy_score(model_A1.predict(te_X), te_Y)  # Test model based on the test set.

# ======================================================================================================================
# Task A2
training_images_2A = np.load('136_features_A1.npy')
smiling_labels_2A = np.load('Smiling_labels_2A.npy')
tr_X, te_X, tr_Y, te_Y = data_preprocessing(training_images_2A, smiling_labels_2A, 1)
model_A2 = SVC(C=0.05994842503189409, kernel='linear', degree=2)
model_A2.fit(tr_X, tr_Y)
acc_A2_train = accuracy_score(model_A2.predict(tr_X), tr_Y)
acc_A2_test = accuracy_score(model_A2.predict(te_X), te_Y)

# ======================================================================================================================
# Task B1
training_images = np.load('face_features_136_1B.npy')
face_labels = np.load('Face_types_1B.npy')
tr_X, te_X, tr_Y, te_Y = data_preprocessing(training_images, face_labels, 2)

sgd = Adam(learning_rate=0.0068, beta_1=0.9, beta_2=0.7, amsgrad=False)

model_1B = models.Sequential()
model_1B.add(layers.Dense(136, activation='relu'))
model_1B.add(layers.Dropout(0.5))
model_1B.add(layers.Dense(68, activation='relu'))
model_1B.add(layers.Dropout(0.5))
model_1B.add(layers.Dense(5, activation='softmax'))
model_1B.compile(optimizer=sgd,
                 loss='categorical_crossentropy',
                 metrics=['acc'])
model_1B.fit(tr_X,
             tr_Y,
             epochs=70,
             batch_size=1000)

results = model_1B.evaluate(tr_X, tr_Y)
acc_B1_train = results[1]
results = model_1B.evaluate(te_X, te_Y)
acc_B1_test = results[1]

# ======================================================================================================================
# Task B2
training_images = np.load('RGB_features_2B.npy')
eye_labels = np.load('eye_color_2B.npy')
tr_X, te_X, tr_Y, te_Y = data_preprocessing(training_images, eye_labels, 3)
model_2B = models.Sequential()
model_2B.add(layers.Dense(3, activation='relu'))
model_2B.add(layers.Dense(12, activation='relu'))
model_2B.add(layers.Dense(5, activation='softmax'))
model_2B.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
model_2B.fit(tr_X, tr_Y, epochs=10, batch_size=100)
results = model_2B.evaluate(tr_X, tr_Y)
acc_B2_train = results[1]
results = model_2B.evaluate(te_X, te_Y)
acc_B2_test = results[1]

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'
