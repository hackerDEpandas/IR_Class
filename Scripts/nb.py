import time
import random
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.naive_bayes import MultinomialNB
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

X = df.drop('G3',1)                       # this is the design matrix
y = list(df.G3)                           # this is the discrete response vector
y_new = response_conv(y)                  # this is the multinomial response vector
X_norm = preprocessing.normalize(X)       # this is the normalized design matrix

random.seed(42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y_new, test_size=0.33, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_norm, y_new, test_size=0.33, random_state=42)
######################################################################################################################
def opt(X,y):
    acc = []
    alphas = 10.0**-np.arange(1,5)
    for a in alphas:
        nb = MultinomialNB(alpha=a)
        scores = cross_val_score(nb, X, y, cv=10, scoring='accuracy')
        acc.append(scores.mean())
    

    opt_ = alphas[acc.index(max(acc))]
    return(opt_)

a1 = opt(X1_train,y1_train)
a3 = opt(X3_train,y3_train)
######################################################################################################################
nb1 = MultinomialNB(alpha=a1).fit(X1_train,y1_train)
nb3 = MultinomialNB(alpha=a3).fit(X3_train,y3_train)

nb_pred1 = nb1.predict(X1_test)
nb_pred3 = nb3.predict(X3_test)

pred = pd.DataFrame(list(zip(y1_test, nb_pred1, nb_pred3)), columns=['y_act','y_nb','y_nb_norm'])
pred.index.name = 'Obs'

pred
######################################################################################################################
cm_nb1 = pd.DataFrame(metrics.confusion_matrix(y1_test, nb_pred1), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_nb3 = pd.DataFrame(metrics.confusion_matrix(y3_test, nb_pred3), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])

print ("The accuracy of the Non-standardized Random Forest model is: ", nb1.score(X1_test,y1_test))
print ("\n")
print ("The accuracy of the Normalized Random Forest model is: ", nb3.score(X3_test,y3_test))
print ("\n")

print("Non-standardized Random Forest Confusion Matrix: \n", cm_nb1)
print ("\n")
print("Normalized Random Forest Confusion Matrix: \n", cm_nb3)
print ("\n")

print("Classification report for Non-standardized design matrix:\n", metrics.classification_report(y1_test,nb_pred1))
print("\n")
print("Classification report for Normalized design matrix:\n", metrics.classification_report(y3_test,nb_pred3))