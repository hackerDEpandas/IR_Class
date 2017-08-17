import time
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils.extmath import cartesian
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv('student-por2.csv')
df = pd.get_dummies(df, drop_first=True)
########################################################################################################################
def var_if_fac(data_frame, ind_var):
    index = data_frame.columns.get_loc(ind_var)
    mat = data_frame.as_matrix()
    return(vif(mat, index))

no_response = df.drop('G3',1)
arr1 = []
arr2 = list(no_response)

for i in list(no_response):
    arr1.append(var_if_fac(no_response,i))
vif_df = pd.DataFrame(list(zip(arr2,arr1)),columns = ['Ind_Var','VIF'])
########################################################################################################################
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

drop_col_names = []

vifs = list(vif_df.VIF)
predictors = list(vif_df.Ind_Var)

for i in range(len(predictors)):
    if vifs[i] >= 10:
        drop_col_names.append(predictors[i])
        
df = df.drop(drop_col_names,1)            # this is the data frame with high VIF variables removed
X = df.drop('G3',1)                       # this is the design matrix
y = list(df.G3)                           # this is the discrete response vector
y_new = response_conv(y)                  # this is the multinomial response vector
X_scale = preprocessing.scale(X)
X_norm = preprocessing.normalize(X)
########################################################################################################################
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y_new, test_size=0.33, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_scale, y_new, test_size=0.33, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_norm, y_new, test_size=0.33, random_state=42)

log_reg1 = LogisticRegressionCV(cv=10,scoring='neg_log_loss',random_state=42).fit(X1_train, y1_train)
log_reg2 = LogisticRegressionCV(cv=10,scoring='neg_log_loss',random_state=42).fit(X2_train, y2_train)
log_reg3 = LogisticRegressionCV(cv=10,scoring='neg_log_loss',random_state=42).fit(X3_train, y3_train)
########################################################################################################################
log_pred1 = log_reg1.predict(X1_test)
log_pred2 = log_reg2.predict(X2_test)
log_pred3 = log_reg3.predict(X3_test)

pred = pd.DataFrame(list(zip(y1_test, log_pred1, log_pred2, log_pred3)), columns=['y_act','y_log','y_stan_log','y_norm_log'])
pred.index.name = 'Obs'
pred
########################################################################################################################
cm_lr1 = pd.DataFrame(metrics.confusion_matrix(y1_test, log_pred1), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_lr2 = pd.DataFrame(metrics.confusion_matrix(y2_test, log_pred2), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])
cm_lr3 = pd.DataFrame(metrics.confusion_matrix(y3_test, log_pred3), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])


print ("The accuracy of the Non-standardized Logistic Regression model is: ", log_reg1.score(X1_test,y1_test))
print ("\n")
print ("The accuracy of the Standardized Logistic Regression model is: ", log_reg2.score(X2_test,y2_test))
print ("\n")
print ("The accuracy of the Normalized Logistic Regression model is: ", log_reg3.score(X3_test,y3_test))
print ("\n")

print("Non-standardized Logistic Regression Confusion Matrix: \n", cm_lr1)
print ("\n")
print("Standardized Logistic Regression Confusion Matrix: \n", cm_lr2)
print ("\n")
print("Normalized Logistic Regression Confusion Matrix: \n", cm_lr3)
print ("\n")

print("Classification report for Non-standardized design matrix:\n", metrics.classification_report(y1_test,log_pred1))
print("\n")
print("Classification report for standardized design matrix:\n", metrics.classification_report(y2_test,log_pred2))
print("\n")
print("Classification report for Normalized design matrix:\n", metrics.classification_report(y3_test,log_pred3))