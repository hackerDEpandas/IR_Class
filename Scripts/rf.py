import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import cartesian
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv('student-por2.csv')
df = pd.get_dummies(df)#, drop_first=True)

def response_conv(arr):
    new = []
    for i in arr:
        if (i > 0 and i < 10):           # condition where student failed
            new.append(0)                 
                                          
        elif (i >= 10):                   # condition where student passed
            new.append(1)                 
    
        else:                             # condition where student received an incomplete
            new.append(2)
    return(new)                           # 1-dimensional response varibale returned

X = df.drop('G3',1)                       # This is the design matrix
y = list(df.G3)                           # This is the discrete response vector
y_new = response_conv(y)                  # This is the multinomial response vector

clf = RandomForestClassifier()
clf.fit(X,y)

model = SelectFromModel(clf,prefit=True)  
newX = model.transform(X)                # design matrix with most influential predictors only

X_scale = preprocessing.scale(newX)
X_norm = preprocessing.normalize(newX)

random.seed(42)
X1_train, X1_test, y1_train, y1_test = train_test_split(newX, y_new, test_size=0.33, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_scale, y_new, test_size=0.33, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_norm, y_new, test_size=0.33, random_state=42)
########################################################################################################################
start_time = time.time()
combos = cartesian([['auto','log2',None],np.arange(10,101,10)])

def opt(X,y):
    acc = []

    for m,t in combos:
        rf = RandomForestClassifier(n_estimators=t,max_features=m,random_state=42)
        scores = cross_val_score(rf, X, y, cv=10, scoring='accuracy')
        acc.append(scores.mean())
    
    opt_k = combos[acc.index(max(acc))]
    return(opt_k)

m1,t1 = opt(X1_train,y1_train)
m2,t2 = opt(X2_train,y2_train)
m3,t3 = opt(X3_train,y3_train)
########################################################################################################################
rf1 = RandomForestClassifier(n_estimators=t1,max_features=m1,random_state=42).fit(X1_train,y1_train)
rf2 = RandomForestClassifier(n_estimators=t2,max_features=m2,random_state=42).fit(X2_train,y2_train)
rf3 = RandomForestClassifier(n_estimators=t3,max_features=m3,random_state=42).fit(X3_train,y3_train)

rf_pred1 = rf1.predict(X1_test)
rf_pred2 = rf2.predict(X2_test)
rf_pred3 = rf3.predict(X3_test)

pred = pd.DataFrame(list(zip(y1_test, rf_pred1, rf_pred2, rf_pred3)), columns=['y_act','y_rf','y_rf_stan','y_rf_norm'])
pred.index.name = 'Obs'
pred
########################################################################################################################
cm_rf1 = pd.DataFrame(metrics.confusion_matrix(y1_test, rf_pred1), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_rf2 = pd.DataFrame(metrics.confusion_matrix(y2_test, rf_pred2), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_rf3 = pd.DataFrame(metrics.confusion_matrix(y3_test, rf_pred3), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])

print ("The accuracy of the Non-standardized Random Forest model is: ", rf1.score(X1_test,y1_test))
print ("\n")
print ("The accuracy of the Standardized Random Forest model is: ", rf2.score(X2_test,y2_test))
print ("\n")
print ("The accuracy of the Normalized Random Forest model is: ", rf3.score(X3_test,y3_test))
print ("\n")

print("Non-standardized Random Forest Confusion Matrix: \n", cm_rf1)
print ("\n")
print("Standardized Random Forest Confusion Matrix: \n", cm_rf2)
print ("\n")
print("Normalized Random Forest Confusion Matrix: \n", cm_rf3)
print ("\n")

print("Classification report for Non-standardized design matrix:\n", metrics.classification_report(y1_test,rf_pred1))
print("\n")
print("Classification report for standardized design matrix:\n", metrics.classification_report(y2_test,rf_pred2))
print("\n")
print("Classification report for Normalized design matrix:\n", metrics.classification_report(y3_test,rf_pred3))