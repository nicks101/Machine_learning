from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#decision tree

clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X, Y)
prediction_tree = clf_tree.predict(X)
score_tree = accuracy_score(Y, prediction_tree)


#KNeighbors

clf_kneighbor = KNeighborsClassifier(n_neighbors=3)
clf_kneighbor.fit(X, Y)
prediction_kneighbor = clf_kneighbor.predict(X)
score_kneighbor = accuracy_score(Y, prediction_kneighbor)


#SVC

clf_svm = svm.SVC()
clf_svm.fit(X, Y)
prediction_svm = clf_svm.predict(X)
score_svm = accuracy_score(Y, prediction_svm)

#comparison

scores = np.array([['DecisionTreeClassifier',score_tree],['SVC' , score_svm],['KNeighbors',score_kneighbor]])
df = pd.DataFrame(scores, columns=['Model','accuracy_score'])
print(df)
print('\nBest classifier : ',df.sort_values(by=['accuracy_score'], ascending=False).iloc[0,0])

