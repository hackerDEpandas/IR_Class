import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
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
    return(new)                           # 1-dimensional array returned

X = df.drop('G3',1)                       # this is the design matrix
y = list(df.G3)                           # this is the discrete response vector
y_new = response_conv(y)                  # this is the multinomial response vector

clf = DecisionTreeClassifier()
clf.fit(X,y)

model = SelectFromModel(clf,prefit=True)
newX = model.transform(X)                 # select most influential predictors

X_scale = preprocessing.scale(newX)       # scaled design matrix
X_norm = preprocessing.normalize(newX)    # normalized design matrix

random.seed(42)
X1_train, X1_test, y1_train, y1_test = train_test_split(newX, y_new, test_size=0.33, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_scale, y_new, test_size=0.33, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_norm, y_new, test_size=0.33, random_state=42)
########################################################################################################################
combos = cartesian([['gini','entropy'],['best','random'],['auto','log2'],np.arange(1,(X1_train.shape[0]-1))])

def opt(X,y):
    acc = []

    for c,s,mf,md in combos:
        dt = DecisionTreeClassifier(criterion=c,splitter=s,max_features=mf,max_depth=int(md),random_state=42)
        scores = cross_val_score(dt, X, y, cv=10, scoring='accuracy')
        acc.append(scores.mean())
    
    opt_ = combos[acc.index(max(acc))]
    return(opt_)

c1,s1,mf1,md1 = opt(X1_train,y1_train)
c2,s2,mf2,md2 = opt(X2_train,y2_train)
c3,s3,mf3,md3 = opt(X3_train,y3_train)
########################################################################################################################
dt1 = DecisionTreeClassifier(criterion=c1,splitter=s1,max_features=mf1,max_depth=int(md1),random_state=42).fit(X1_train,y1_train)
dt2 = DecisionTreeClassifier(criterion=c2,splitter=s2,max_features=mf2,max_depth=int(md2),random_state=42).fit(X2_train,y2_train)
dt3 = DecisionTreeClassifier(criterion=c3,splitter=s3,max_features=mf3,max_depth=int(md3),random_state=42).fit(X3_train,y3_train)

dt_pred1 = dt1.predict(X1_test)
dt_pred2 = dt2.predict(X2_test)
dt_pred3 = dt3.predict(X3_test)

pred = pd.DataFrame(list(zip(y1_test, dt_pred1, dt_pred2, dt_pred3)), columns=['y_act','y_dt','y_dt_stan','y_dt_norm'])
pred.index.name = 'Obs'
pred
########################################################################################################################
cm_dt1 = pd.DataFrame(metrics.confusion_matrix(y1_test, dt_pred1), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_dt2 = pd.DataFrame(metrics.confusion_matrix(y2_test, dt_pred2), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_dt3 = pd.DataFrame(metrics.confusion_matrix(y3_test, dt_pred3), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])


print ("The accuracy of the Non-standardized Decision Tree model is: ", dt1.score(X1_test,y1_test))
print ("\n")
print ("The accuracy of the Standardized Decision Tree model is: ", dt2.score(X2_test,y2_test))
print ("\n")
print ("The accuracy of the Normalized Decision Tree model is: ", dt3.score(X3_test,y3_test))
print ("\n")

print("Non-standardized Decision Tree Confusion Matrix: \n", cm_dt1)
print ("\n")
print("Standardized Decision Tree Confusion Matrix: \n", cm_dt2)
print ("\n")
print("Normalized Decision Tree Confusion Matrix: \n", cm_dt3)
print ("\n")

print("Classification report for Non-standardized design matrix:\n", metrics.classification_report(y1_test,dt_pred1))
print("\n")
print("Classification report for standardized design matrix:\n", metrics.classification_report(y2_test,dt_pred2))
print("\n")
print("Classification report for Normalized design matrix:\n", metrics.classification_report(y3_test,dt_pred3))