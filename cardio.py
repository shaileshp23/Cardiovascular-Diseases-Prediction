import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
df= pd.read_csv("cardioheart.csv")
df.rename(columns = {'cardio':'Target'}, inplace = True)
Y = df.Target
X = df.drop('Target', axis=1)
train, test, target, target_test = train_test_split(X,Y,test_size = 0.2,stratify=Y,random_state=2)
train.drop
print(X.shape,train.shape, test.shape)
Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(train, target)
acc_log = round(logreg.score(train, target) * 100, 2)
acc_logacc_test_log = round(logreg.score(test, target_test) * 100, 2)
#print(acc_test_log)
coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
svc = SVC()
svc.fit(train, target)
acc_svc = round(svc.score(train, target) * 100, 2)
#acc_svc
acc_test_svc = round(svc.score(test, target_test) * 100, 2)
#acc_test_svc
knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [2, 3]}, cv=10).fit(train, target)
acc_knn = round(knn.score(train, target) * 100, 2)
print(acc_knn, knn.best_params_)
acc_test_knn = round(knn.score(test, target_test) * 100, 2)
#acc_test_knn
gaussian = GaussianNB()
gaussian.fit(train, target)
acc_gaussian = round(gaussian.score(train, target) * 100, 2)
#acc_gaussian
acc_test_gaussian = round(gaussian.score(test, target_test) * 100, 2)
#acc_test_gaussian
perceptron = Perceptron()
perceptron.fit(train, target)
acc_perceptron = round(perceptron.score(train, target) * 100, 2)
#acc_perceptron
acc_test_perceptron = round(perceptron.score(test, target_test) * 100, 2)
#acc_test_perceptron
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train, target)
acc_decision_tree = round(decision_tree.score(train, target) * 100, 2)
#acc_decision_tree
acc_test_decision_tree = round(decision_tree.score(test, target_test) * 100, 2)
#acc_test_decision_tree
random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 300]}, cv=5).fit(train, target)
random_forest.fit(train, target)
acc_random_forest = round(random_forest.score(train, target) * 100, 2)
print(acc_random_forest,random_forest.best_params_)
acc_test_random_forest = round(random_forest.score(test, target_test) * 100, 2)
acc_test_random_forest
def build_ann(optimizer='adam'):

    # Initializing the ANN
    ann = Sequential()

    # Adding the input layer and the first hidden layer of the ANN with dropout
    ann.add(Dense(units=32, kernel_initializer='glorot_uniform', activation='relu', input_shape=(len(train.columns),)))

    # Add other layers, it is not necessary to pass the shape because there is a layer before
    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))
    ann.add(Dropout(rate=0.5))
    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))
    ann.add(Dropout(rate=0.5))

    # Adding the output layer
    ann.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    # Compiling the ANN
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return ann
from tensorflow.keras.optimizers import Adam
opt = optimizers.Adam(lr=0.001)
ann = build_ann(opt)
# Training the ANN
history = ann.fit(train, target, batch_size=16, epochs=100, validation_data=(test, target_test))
ann_prediction = ann.predict(train)
ann_prediction = (ann_prediction > 0.5)*1
acc_ann1 = round(metrics.accuracy_score(target, ann_prediction) * 100, 2)
acc_ann1
ann_prediction_test = ann.predict(test)
ann_prediction_test = (ann_prediction_test > 0.5)*1
acc_test_ann1 = round(metrics.accuracy_score(target_test, ann_prediction_test) * 100, 2)
acc_test_ann1
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
(train, target), (test, target_test) = tf.keras.datasets.mnist.load_data()
train = train.astype('float32')
target = target.astype('float32')
gray_scale = 255
train /= gray_scale
target /= gray_scale
print("Feature matrix:", train.shape)
print("Target matrix:", target.shape)
print("Feature matrix:", test.shape)
print("Target matrix:", target_test.shape)
model = Sequential([

    # reshape 28 row * 28 column data to 28*28 rows
    Flatten(input_shape=(28, 28)),

      # dense layer 1
    Dense(256, activation='sigmoid'),

    # dense layer 2
    Dense(128, activation='sigmoid'),

      # output layer
    Dense(10, activation='sigmoid'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train, target, epochs=10,
          batch_size=2000,
          validation_split=0.2)

results = model.evaluate(test,  target_test, verbose = 0)
print('test loss, test acc:', results)
cvscores = []
print("%s: %.2f%%" % (model.metrics_names[1], results[1]*100))
cvscores.append(results[1] * 100)
print(cvscores)
mlp_acc = "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))
print(mlp_acc)
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'k-Nearest Neighbors', 'Naive Bayes', 'Perceptron', 'Decision Tree Classifier', 'Random Forest', 'Neural Network '],

    'Score_train': [acc_log, acc_svc, acc_knn, acc_gaussian,  acc_perceptron, acc_decision_tree, acc_random_forest,acc_ann1,],

    'Score_test': [acc_test_log, acc_test_svc,  acc_test_knn, acc_test_gaussian,  acc_test_perceptron, acc_test_decision_tree, acc_test_random_forest,  acc_test_ann1]
})
print(models)
input_data = []
age = int(input("Enter age of person: "))
gender = int(input("Enter gender of person {for male : 1 ; for female : 2}:  "))
height = int(input("Enter height of person: "))
weight = int(input("Enter weight of person: "))
bp_lo = int(input("Enter systolic blood pressure of person: "))
bp_hi = int(input("Enter diastolic blood pressure of person: "))
cholesterol = int(input("Enter cholesterol of person: "))
heartrate = int(input("Enter glucose level of person: "))
smoker = int(input("Does the person smoke? {Yes: 1; No: 0}: "))
alcoholic = int(input("Is the person alcoholic? {Yes: 1; No: 0}: "))
physically_fit = int(input("Is the person physically active? {Yes: 1; No: 0}: "))
input_data.append(age)
input_data.append(gender)
input_data.append(height)
input_data.append(weight)
input_data.append(bp_lo)
input_data.append(bp_hi)
input_data.append(cholesterol)
input_data.append(heartrate)
input_data.append(smoker)
input_data.append(alcoholic)
input_data.append(physically_fit)
age = input_data[0]
gender = input_data[1]
height = input_data[2]
weight = input_data[3]
bp_lo = input_data[4]
bp_hi = input_data[5]
cholesterol = input_data[6]
heartrate = input_data[7]
smoker = input_data[8]
alcoholic = input_data[9]
physically_fit = input_data[10]
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(" *** WELCOME TO HEART DISEASE PREDICTOR ***")
print("-----------------------------------------------------")
print("age:" , age)
print("-----------------------------------------------------")
if gender ==1:
  print("gender: male")
else:
  print("gender: female")
print("-----------------------------------------------------")
print("weight: ", weight)
print("-----------------------------------------------------")
print("systolic blood pressure: ", bp_lo)
print("-----------------------------------------------------")
print("diastolic blood pressure: ", bp_hi)
print("-----------------------------------------------------")
print("cholesterol: ", cholesterol)
print("-----------------------------------------------------")
print("blood glucose level: ", heartrate)
print("-----------------------------------------------------")
if smoker == 1:
  print("is he/she a smoker?: yes")
else:
  print("is he/she a smoker?: no")
print("-----------------------------------------------------")
if smoker == 1:
  print("is he/she a alcoholic?: yes")
else:
  print("is he/she a alcoholic?: no")

print("-----------------------------------------------------")
if physically_fit ==1:
  print("does he/she do any physical activity?: yes")
else:
  print("does he/she do any physical activity?: no")
print("-----------------------------------------------------")
import math
score = 0
if age >= 40:
  score += 1
elif age >= 50:
  score += 2

if gender == 1:
  score += 2
else:
  score += 1

h = height/100
bmi = weight / (h*h)

if bmi >= 40:
  score += 2
elif bmi >= 30:
  score += 1

if bp_hi in range(130,140) and bp_lo in range(80,90):
  score += 1
elif bp_hi >= 140 and bp_lo >= 90:
  score += 2

score += (cholesterol + heartrate + smoker + alcoholic + physically_fit)

percentage = (score/16)*100
prediction = random_forest.predict(input_data_reshaped)

if (prediction[0] == 0):
    print('person does not have heart disease.')
else:
    print('person may have or may develop a heart disease in the future if this lifestyle is continued.')
print("-----------------------------------------------------")
print("chances of cardiovascular disease: ", math.ceil(percentage*100)/100, "%")
print("-----------------------------------------------------")
