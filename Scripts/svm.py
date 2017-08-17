import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.utils.extmath import cartesian
from sklearn import preprocessing
from sklearn import metrics, svm

df = pd.read_csv('student-por2.csv')
df = pd.get_dummies(df)#, drop_first=True)

def response_conv(arr):
    new = []
    for i in arr:
        if (i > 0 and i < 10):           # condition where student failed
            new.append(0)                 
                                          
        elif (i >= 10):                  # condition where student passed
            new.append(1)                 
    
        else:                            # condition where student received an incomplete
            new.append(2)
    return(new)                          # 1-dimensional response varibale returned

X = df.drop('G3',1)
y = response_conv(list(df.G3))

select = SelectPercentile()
newX = select.fit_transform(X,y)         # design matrix with most influential predictors only

X_scale = preprocessing.scale(newX)
X_norm = preprocessing.normalize(newX)

random.seed(42)
X1_train, X1_test, y1_train, y1_test = train_test_split(newX, y, test_size=0.33, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_scale, y, test_size=0.33, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_norm, y, test_size=0.33, random_state=42)
########################################################################################################################
start_time = time.time()
combos = cartesian([['linear','rbf'],[0.1,1,10,100,1000],[0.1,0.01,0.001,0.0001]])
def opt(X,y):
    acc = []

    for k,c,g in combos:
        svc = svm.SVC(C=float(c),kernel=str(k),gamma=float(g),probability=True,decision_function_shape='ovo')
        scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
        acc.append(scores.mean())
    
    #MSE = [1 - x for x in cv_scores]
    opt_ = combos[acc.index(max(acc))]
    return(opt_)

k1,c1,g1 = opt(X1_train,y1_train)
k2,c2,g2 = opt(X2_train,y2_train)
k3,c3,g3 = opt(X3_train,y3_train)
########################################################################################################################
SVM1 = svm.SVC(C=float(c1),kernel=str(k1),gamma=float(g1),decision_function_shape='ovo').fit(X1_train,y1_train)
SVM2 = svm.SVC(C=float(c2),kernel=str(k2),gamma=float(g2),decision_function_shape='ovo').fit(X2_train,y2_train)
SVM3 = svm.SVC(C=float(c3),kernel=str(k3),gamma=float(g3),decision_function_shape='ovo').fit(X3_train,y3_train)


svm1_pred = SVM1.predict(X1_test)
svm2_pred = SVM2.predict(X2_test)
svm3_pred = SVM3.predict(X3_test)

pred = pd.DataFrame(list(zip(y1_test, svm1_pred,svm2_pred,svm3_pred)), columns=['y_act','y_svm','y_svm_stand','y_svm_norm'])
pred.index.name = 'Obs'
pred
########################################################################################################################
cm_svm1 = pd.DataFrame(metrics.confusion_matrix(y1_test, svm1_pred), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_svm2 = pd.DataFrame(metrics.confusion_matrix(y2_test, svm2_pred), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_svm3 = pd.DataFrame(metrics.confusion_matrix(y3_test, svm3_pred), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])

print ("The accuracy of the Non-standardized SVM model is: ", SVM1.score(X1_test,y1_test))
print ("\n")
print ("The accuracy of the standardized SVM model is: ", SVM2.score(X2_test,y2_test))
print ("\n")
print ("The accuracy of the normalized SVM model is: ", SVM3.score(X3_test,y3_test))
print ("\n")

print("Non-standarized SVM Confusion Matrix: \n", cm_svm1)
print ("\n")
print("Standarized SVM Confusion Matrix: \n", cm_svm2)
print ("\n")
print("Normalized SVM Confusion Matrix: \n", cm_svm3)
print ("\n")

print("Classification report for Non-standardized design matrix:\n", metrics.classification_report(y1_test,svm1_pred))
print("\n")
print("Classification report for standardized design matrix:\n", metrics.classification_report(y2_test,svm2_pred))
print("\n")
print("Classification report for Normalized design matrix:\n", metrics.classification_report(y3_test,svm3_pred))