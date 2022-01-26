import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

#import the dataset using the url and read it with the help of pandas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)

#assign the x and y variables respectively to the features and labels of the iris dataset
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

#create the lda model/classifier with one linear discriminant
classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


#We can fit and evaluate the lda model using repeated stratified k-fold cross-validation via the RepeatedStratifiedKFold class.
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

cm = confusion_matrix(y_test, y_pred)

#print the relevant parameters used to evaluate a models perfomance
print("Train Accuracy:",classifier.score(X_train, y_train))
print('Cross Validation Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print("Test Accuracy:",accuracy_score(y_test, y_pred))
print(cm)
print(classification_report(y_test, y_pred))


