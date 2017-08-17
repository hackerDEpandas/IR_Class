import time
import random
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.utils.extmath import cartesian
from sklearn import metrics
from sklearn import preprocessing

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
    return(new)                           # 1-dimensional response varibale returned

X = df.drop('G3',1)                       # this is the design matrix
y = list(df.G3)                           # this is the discrete response vector
y_new = response_conv(y)                  # this is the multinomial response vector

clf = SGDClassifier()
clf.fit(X,y)

model = SelectFromModel(clf,prefit=True)
newX = model.transform(X)                 # design matrix with most influential predictors only

X_scale = preprocessing.scale(newX)
X_norm = preprocessing.normalize(newX)

random.seed(42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y_new, test_size=0.33, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_scale, y_new, test_size=0.33, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_norm, y_new, test_size=0.33, random_state=42)
########################################################################################################################
start_time = time.time()
combos = ['hinge','log','modified_huber','squared_hinge','perceptron','squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive']
    
def opt(X,y):
    acc = []
    for l in combos:
        sgd = SGDClassifier(loss=str(l),random_state=42)
        scores = cross_val_score(sgd, X, y, cv=10, scoring='accuracy')
        acc.append(scores.mean())
    

    opt_ = combos[acc.index(max(acc))]
    return(opt_)

l1 = opt(X1_train,y1_train)
l2 = opt(X2_train,y2_train)
l3 = opt(X3_train,y3_train)
########################################################################################################################
sgd1 = SGDClassifier(loss=l1,random_state=42).fit(X1_train,y1_train)
sgd2 = SGDClassifier(loss=l2,random_state=42).fit(X2_train,y2_train)
sgd3 = SGDClassifier(loss=l3,random_state=42).fit(X3_train,y3_train)


sgd_pred1 = sgd1.predict(X1_test)
sgd_pred2 = sgd2.predict(X2_test)
sgd_pred3 = sgd3.predict(X3_test)

pred = pd.DataFrame(list(zip(y1_test, sgd_pred1,sgd_pred2,sgd_pred3)), columns=['y_act','y_sgd','y_sgd_stand','y_sgd_norm'])
pred.index.name = 'Obs'
pred
########################################################################################################################
cm_sgd1 = pd.DataFrame(metrics.confusion_matrix(y1_test, sgd_pred1), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_sgd2 = pd.DataFrame(metrics.confusion_matrix(y2_test, sgd_pred2), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_sgd3 = pd.DataFrame(metrics.confusion_matrix(y3_test, sgd_pred3), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])

print ("The accuracy of the Non-standarized SGD model is: ", sgd1.score(X1_test,y1_test))
print("\n")
print ("The accuracy of the Standardized SGD model is: ", sgd2.score(X2_test,y2_test))
print("\n")
print ("The accuracy of the Normalized SGD model is: ", sgd3.score(X3_test,y3_test))
print("\n")

print("Non-standarized SGD Confusion Matrix: \n", cm_sgd1)
print("\n")
print("Standarized SGD Confusion Matrix: \n", cm_sgd2)
print("\n")
print("Normalized SGD Confusion Matrix: \n", cm_sgd3)
print("\n")

print("Classification report for Non-standardized design matrix:\n", metrics.classification_report(y1_test,sgd_pred1))
print("\n")
print("Classification report for standardized design matrix:\n", metrics.classification_report(y2_test,sgd_pred2))
print("\n")
print("Classification report for Normalized design matrix:\n", metrics.classification_report(y3_test,sgd_pred3))