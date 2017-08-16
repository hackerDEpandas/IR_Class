import pandas as pd
import time
import random
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.neural_network import MLPClassifier
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
y = response_conv(list(df.G3))            # this is the multinomial response vector

clf = MLPClassifier(learning_rate_init=1)
clf.fit(X,y)

select = SelectPercentile(percentile=50)
newX = select.fit_transform(X,y)          # select most influential predictors

X_scale = preprocessing.scale(newX)       # scaled design matrix

random.seed(42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X_scale, y, test_size=0.33, random_state=42)
########################################################################################################################
zero = 0
one = 0
two = 0

for i in y1_test:
    if i == 0:
        zero += 1
    elif i == 1:
        one += 1
    else:
        two += 1

num1 = round((zero/len(y1_test))*100,2)
num2 = round((one/len(y1_test))*100,2)
num3 = round((two/len(y1_test))*100,2)
########################################################################################################################
combos = cartesian([['constant'],['sgd'],['logistic', 'tanh', 'relu'],10.0**-np.arange(1,7)])
in_layer_size = len(list(X))
out_layer_size = 3
hidden_layer_size = int((in_layer_size+out_layer_size)/2)

def opt(X,y):
    acc = []
    for learn,solver,act,alpha in combos:
        nn = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,),activation=act,solver=solver,learning_rate=learn,alpha=float(alpha),learning_rate_init = 0.5,random_state=42)
        scores = cross_val_score(nn, X, y, cv=10, scoring='accuracy')
        acc.append(scores.mean())
    

    opt_ = combos[acc.index(max(acc))]
    return(opt_)

lea1,sol1,act1,alp1 = opt(X1_train,y1_train)
########################################################################################################################
mlp1 = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,),activation=act1,solver=sol1,learning_rate=lea1,learning_rate_init = 0.5,alpha=float(alp1),random_state=42)
nn1 = mlp1.fit(X1_train,y1_train)
########################################################################################################################
nn_pred1 = nn1.predict(X1_test)

pred = pd.DataFrame(list(zip(y1_test, nn_pred1)), columns=['y_act','y_nn_stan'])
pred.index.name = 'Obs'
pred
########################################################################################################################
cm_nn1 = pd.DataFrame(metrics.confusion_matrix(y1_test, nn_pred1), index = ['Fail(0)','Pass(1)','Inc(2)'],columns=['Fail(0)','Pass(1)','Inc(2)'])

print ("The accuracy of the Standardized Neural Network model is: ", nn1.score(X1_test,y1_test))
print ("\n")

print("Standardized Neural Network Confusion Matrix: \n", cm_nn1)
print ("\n")

print("Classification report for standardized design matrix:\n", metrics.classification_report(y1_test,nn_pred1))