from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import scipy
from theano import *
import pandas
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


#seed - porownanie

np.random.seed(7)

#load features
#8 inputow, 1 outputy

dataset = np.loadtxt("train.csv", delimiter=",")
dataset1 = np.loadtxt("test.csv", delimiter=",")

#dzielenie na output i input

X=dataset[:,0:7]
Y=dataset[:,7]
Z=dataset1[:,0:7]
Z1=dataset1[:,7]

#model
model = Sequential()
model.add(Dense(64, input_dim=7, activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dense(1, activation='softmax'))

sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#compile model
model.compile(loss='sparse__categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting
model.fit(X, Y, epochs=20, batch_size=128)

#evaluation
scores=model.evaluate(Z, Z1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#przewidywanie
predictions = model.predict(Z)

#round
rounded = [round(x[0]) for x in predictions]
print(rounded)

#dokladnosc predykcji

