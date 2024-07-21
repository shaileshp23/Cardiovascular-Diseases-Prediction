import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier


# loading and reading the dataset

heart = pd.read_csv("c2.csv")

# creating a copy of dataset so that will not affect our original dataset.
heart_df = heart.copy()
heart_df = np.array(heart_df)
# Renaming some of the columns 
#heart_df = heart_df.rename(columns={'perfect':'target'})
#print(heart_df.head())

# model building 
#X = heart_df[1:, 1:-1]
#y# = heart_df[1:, -1]
#y = y.astype('int')
#X = X.astype('int')
#fixing our data in x and y. Here y contains target data and X contains rest all the features.


X= heart_df.drop(columns= 'perfect')
Y= heart_df.target

# splitting our dataset into training and testing for this we will use train_test_split library.
train, test, target, target_test = train_test_split(X,Y,test_size = 0.2,stratify=Y,random_state=2)

#feature scaling
scaler= StandardScaler()
x_train_scaler= scaler.fit_transform(train)
x_test_scaler= scaler.fit_transform(test)

# creating K-Nearest-Neighbor classifier
random_forest = RandomForestClassifier(n_estimators=20)
#model = LogisticRegression(n_estimators=20)
random_forest.fit(train,target)
y_pred= random_forest.predict(x_test_scaler)
p = random_forest.score(x_test_scaler,test)
print(p)

print('Classification Report\n', classification_report(test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(test, y_pred)*100),2)))

cm = confusion_matrix(test, y_pred)
print(cm)
#inputt=[int(x) for x in "id age gender height weight ap_hi ap_lo cholesterol gluc smoke alco active".split(' ')]
#final=[np.array(inputt)]

#b = model.predict_proba(final)

# Creating a pickle file for the classifier
filename = 'final2.pkl'
pickle.dump(random_forest, open(filename, 'wb'))
