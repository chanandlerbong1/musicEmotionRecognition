import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from theano import *
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


#seed do badania parametrów
seed = 7
np.random.seed(seed)

#dane
#treningowe
dataframe = pandas.read_csv("train.csv", header=None)
dataset = dataframe.values

#testowe
dataframe1 = pandas.read_csv("test.csv", header=None)
dataset1 = dataframe1.values

#ankiety
#dataframe1 = pandas.read_csv("ank.csv", header=None)
#dataset2 = dataframe1.values


#dzielenie na output i input
X=dataset[:, 0:6].astype(float)
Y=dataset[:, 6]

# Z=dataset1[:,0:6].astype(float)
# Z1=dataset1[:,6]
# ankieta1=dataset2[:,0:6].astype(float)
# ankieta2=dataset2[:,6]


#ecnoding etykiet z nazwami emocji  na floaty
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# encoder.fit(Z1)
# encoded_Z = encoder.transform(Z1)
# dummy_z = np_utils.to_categorical(encoded_Z)
# encoder.fit(Z1)
# encoded_Z1 = encoder.transform(Z1)
# dummy_z1 = np_utils.to_categorical(encoded_Z1)

#budowa modelu
def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=6, activation='relu'))
    model.add(Dense(4, activation='softmax'))


#kompilacja modelu
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#fitting
estimator = KerasClassifier(build_fn=baseline_model, epochs=10000, batch_size=5)




    #evaluation
    # kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    # results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    # print(" ")
    # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    # print(' ')


 #podzielenie zbioru treningowego na zbior testowy 0.23 i treningowy 0.75
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.25, random_state=seed)


estimator.fit(X_train, Y_train)

#predykcja
predictions = estimator.predict(X_test)

#ocena
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X_test, Y_test, cv=kfold)


print(predictions)
print(encoder.inverse_transform(predictions))
print("____________")
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
