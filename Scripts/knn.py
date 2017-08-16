import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.extmath import cartesian
from sklearn import preprocessing
from sklearn import metrics

df = pd.read_csv('student-por2.csv')
df = pd.get_dummies(df)#, drop_first=True)

def response_conv(arr):
    new = []
    for i in arr:
        if (i > 0 and i < 10):            # condition where student failed
            new.append(0)                 
                                          
        elif (i >= 10):                   # condition where student passed
            new.append(1)                 
    
        else:                             # condition where student received an incomplete
            new.append(2)
    return(new)                           # 1-dimensional array returned


X = df.drop('G3',1)                       # This is the design matrix
y = response_conv(list(df.G3))            # This is the multinomial response vector

select = SelectPercentile()
newX = select.fit_transform(X,y)          # Select most influential predictors

X_scale = preprocessing.scale(newX)       # Scaled design matrix
X_norm = preprocessing.normalize(newX)    # Normalized design matrix

random.seed(42)
X1_train, X1_test, y1_train, y1_test = train_test_split(newX, y, test_size=0.33,random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_scale, y, test_size=0.33,random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_norm, y, test_size=0.33,random_state=42)
########################################################################################################################
myList = list(range(1,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))
combos = cartesian([['uniform','distance'],neighbors])

def opt(X,y):
    acc = []

    for w, k in combos:
        knn = KNeighborsClassifier(n_neighbors=int(k),weights=str(w))
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        acc.append(scores.mean())
    
    #MSE = [1 - x for x in cv_scores]
    opt_ = combos[acc.index(max(acc))]
    return(opt_)

w1, k1 = opt(X1_train,y1_train)
w2, k2 = opt(X2_train,y2_train)
w3, k3 = opt(X3_train,y3_train)
########################################################################################################################
knn1 = KNeighborsClassifier(n_neighbors=int(k1),weights=str(w1)).fit(X1_train,y1_train)
knn2 = KNeighborsClassifier(n_neighbors=int(k2),weights=str(w2)).fit(X2_train,y2_train)
knn3 = KNeighborsClassifier(n_neighbors=int(k3),weights=str(w3)).fit(X3_train,y3_train)

knn_pred1 = knn1.predict(X1_test)
knn_pred2 = knn2.predict(X2_test)
knn_pred3 = knn3.predict(X3_test)
########################################################################################################################
cm_knn1 = pd.DataFrame(metrics.confusion_matrix(y1_test, knn_pred1), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_knn2 = pd.DataFrame(metrics.confusion_matrix(y2_test, knn_pred2), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_knn3 = pd.DataFrame(metrics.confusion_matrix(y3_test, knn_pred3), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])

print ("The accuracy of the Non-standarized KNN model is: ", knn1.score(X1_test,y1_test))
print("\n")
print ("The accuracy of the Standardized KNN model is: ", knn2.score(X2_test,y2_test))
print("\n")
print ("The accuracy of the Normalized KNN model is: ", knn3.score(X3_test,y3_test))
print("\n")

print("Non-standarized KNN Confusion Matrix: \n", cm_knn1)
print("\n")
print("Standarized KNN Confusion Matrix: \n", cm_knn2)
print("\n")
print("Normalized KNN Confusion Matrix: \n", cm_knn3)
print("\n")

print("Classification report for Non-standardized design matrix:\n", metrics.classification_report(y1_test,knn_pred1))
print("\n")
print("Classification report for standardized design matrix:\n", metrics.classification_report(y2_test,knn_pred2))
print("\n")
print("Classification report for Normalized design matrix:\n", metrics.classification_report(y3_test,knn_pred3))