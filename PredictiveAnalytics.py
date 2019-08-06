#!/usr/bin/env python
# coding: utf-8

# In[8]:



## 363-1098-00L Business Analytics
## Project Report
## Authors: Belén Cantos Bernal
##          Enxhi Gjini
##          Joan Puig Sallés

## PREDICTIVE ANALYTICS

# Package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
get_ipython().run_line_magic('matplotlib', 'inline')

# In[57]:
# ### 1. Load the data 

df = pd.read_csv('ba.csv')
df = df.drop(df.columns[0], axis = 1) 
print(df)


# In[56]:
# ### 2. Specify input and output vectors

X = df.iloc[:,1:20]
y = df.iloc[:,20:22]




# In[4]:
# ### 3. Apply label encoding

# We need to convert categorical variables to numerical variables

X = X.apply(LabelEncoder().fit_transform)

# In[5]:

# ### 4. Split the train and test sets

# Train set = 80% <br>
# Test set = 20%


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)



# In[6]:

# ### 5. Oversampling to overcome data unbalance
# We have an unbalanced dataset, as the number of deaths represents less than 10% of the whole dataset. To make sure we do not miss any deaths, we need to counteract this phenomena and oversample the number of deaths. 


#Correct for Imbalanced data

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(X_train, y_train.iloc[:,0])


# In[18]:

# ### 6. Death Prediction

# First, we define the classifiers and train them with the oversampled x and y train tests. Then we predict the values of the x test set using the classifiers. Finally, we compute the accuracy, recall and F1 scores.

# ### 6.0 Overview of the classifiers

# Define classifiers
log_reg = LogisticRegression(solver="liblinear")
la = LassoLarsIC()
en = ElasticNet()
br = BayesianRidge()
gbc = GradientBoostingClassifier()
bc = BaggingClassifier()
rfc = clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clfs = [log_reg, la, en, br, gbc, bc, rfc]
labels = ["Logistic Regression", "Lasso Lars IC", "Elastic Net", 
              "Bayesian Ridge", "Gradient Boosting", "Bagging", "Random Forest"]

# Fit classifiers and predict
for clf, label in zip(clfs, labels):
    clf = clf.fit(X=x_train_res, y=y_train_res)
    #y_pred = np.round(clf.predict(X_test),0)
    
    y_pred = clf.predict(X_test)
    if (label == "Elastic Net") or (label == "Bayesian Ridge") or (label == "Lasso Lars IC"):
        y_pred_prob = y_pred
    else:
        y_pred_prob = clf.predict_proba(X_test)
        y_pred_prob = y_pred_prob[:,1]
    
    accuracy = round(accuracy_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)
    recall = round(recall_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)
    f1 = round(f1_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)
    
    print(label + " - Accuracy: " + str(round(accuracy_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)) +
          " / Recall:" + str(round(recall_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)) +
         " / F1 Score:" + str(round(f1_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)))
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test.iloc[:,0], y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (label, roc_auc))
 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_curve.png', dpi = 200)
plt.show()   # Display


# In[16]:


# Just for plotting the PRC Curves

# Define classifiers
log_reg = LogisticRegression(solver="liblinear")
la = LassoLarsIC()
en = ElasticNet()
br = BayesianRidge()
gbc = GradientBoostingClassifier()
bc = BaggingClassifier()
rfc = clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clfs = [log_reg, la, en, br, gbc, bc, rfc]
labels = ["Logistic Regression", "Lasso Lars IC", "Elastic Net", 
              "Bayesian Ridge", "Gradient Boosting", "Bagging", "Random Forest"]

# Fit classifiers and predict
for clf, label in zip(clfs, labels):
    clf = clf.fit(X=x_train_res, y=y_train_res)
    #y_pred = np.round(clf.predict(X_test),0)
    
    y_pred = clf.predict(X_test)
    if (label == "Elastic Net") or (label == "Bayesian Ridge") or (label == "Lasso Lars IC"):
        y_pred_prob = y_pred
    else:
        y_pred_prob = clf.predict_proba(X_test)
        y_pred_prob = y_pred_prob[:,1]
    
    accuracy = round(accuracy_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)
    recall = round(recall_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)
    f1 = round(f1_score(y_test.iloc[:,0], np.round(y_pred,0)), 3)
  
    precision, recall, _ =  precision_recall_curve(y_test.iloc[:,0], y_pred_prob)
    plt.plot(recall, precision,label='%s Average Recall = %0.2f' % (label,recall_score(y_test.iloc[:,0], np.round(y_pred))))
 
    print(label + " - Average Precision: " + str(average_precision_score(y_test.iloc[:,0], y_pred_prob)))

no_skill_line = sum(y_train_res)/len(y_train_res)

plt.plot([0, 1], [no_skill_line, no_skill_line],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.savefig('PRC_curve.png', dpi = 200)
plt.show()   # Display


# In[71]:


clf.predict_proba(X_test)[:,1]



# In[22]:

# ### 6.1 Elastic Net

#Define classifier
enet = ElasticNet() 

#Find best parameters 
param_grid = {"alpha": [0.01, 0.05, 0.1],
              "l1_ratio": np.arange(0.1, 1.0, 0.1)}
CV_enet = GridSearchCV(estimator=enet, param_grid=param_grid, cv=5)

#Fit model and print best parameters
CV_enet.fit(x_train_res, y_train_res)
print(CV_enet.best_params_)


# In[20]:


#Predict values using the test set
enet_pred = abs(np.round(CV_enet.predict(X_test),0))


# In[21]:


#Print metrics
print("Accuracy: " + str(round(accuracy_score(y_test.iloc[:,0], enet_pred), 3)) + 
      "\nRecall: " + str(round(recall_score(y_test.iloc[:,0], enet_pred), 3)) +
      "\nF1 Score: " + str(round(f1_score(y_test.iloc[:,0], enet_pred), 3)))




# In[103]:
# ### 6.1 Random Forest

#Define model
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

#Find best parameters 
param_grid = { 
    'n_estimators': [50,100],
    'max_depth': [4,5,6],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='f1')

#Fit model and print best parameters
CV_rfc.fit(x_train_res, y_train_res)
print(CV_rfc.best_params_)


# In[104]:


#Predict values using the test set
rfc_pred = CV_rfc.predict(X_test)


# In[105]:


#Print metrics
print("Accuracy: " + str(round(accuracy_score(y_test.iloc[:,0], rfc_pred), 3)) + 
      "\nRecall: " + str(round(recall_score(y_test.iloc[:,0], rfc_pred), 3)) +
      "\nF1 Score: " + str(round(f1_score(y_test.iloc[:,0], rfc_pred), 3)))




# In[107]:
# ### 6.3 Gradient Boosting

#Define model
gbc = GradientBoostingClassifier()

#Find best parameters 
param_grid = { 
    "loss":["deviance", "exponential"],
    "learning_rate": [0.05, 0.075, 0.1],
}

CV_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5)

#Fit model and print best parameters
CV_gbc.fit(x_train_res, y_train_res)
print(CV_gbc.best_params_)


# In[108]:


#Predict values using the test set
gbc_pred = CV_gbc.predict(X_test)


# In[109]:


#Print metrics
print("Accuracy: " + str(round(accuracy_score(y_test.iloc[:,0], gbc_pred), 3)) + 
      "\nRecall: " + str(round(recall_score(y_test.iloc[:,0], gbc_pred), 3)) +
      "\nF1 Score: " + str(round(f1_score(y_test.iloc[:,0], gbc_pred), 3)))


# In[25]:

# ### 7 Example of an incoming patient admission

##NEW ADMISSION

# Please, fill in the following data:

# Patient Information:
Age = 65
Gender = 1 #Female
Admission_Type = 2 #Emergency
Admissin_Location = 3 #EMERGENCY ROOM ADMIT
Insurance = 4 #Private
Religion = 0 #Not known
Marital_Status = 0 #Not known
Ethnicity = 3 #Asian
Diagnosis = 984 #AORTIC RUPTURE
ICD9_code = 7 #ICD-9 codes 390–459: diseases of the circulatory system

# ICU machines / Lab related data

Oxygen = 100
PO2 = 155.421
Bicarbonate = 20.3333
Bilirubin = 0.3
Sodium = 136.867
Urea_Nitrogen = 19.8667
Potassium = 4.70333
WBC = 7.13914
Urine = 3739.39

PatientX = np.array([[ICD9_code, Oxygen, PO2, Bicarbonate, Bilirubin, Sodium, Urea_Nitrogen, 
                    Potassium, WBC, Urine, Age, Gender, Admission_Type, Admissin_Location,
                    Insurance, Religion, Marital_Status, Ethnicity, Diagnosis]])


# In[55]:


#Prediction using the Elastic NET Model

model_Elnet = ElasticNet(alpha = 0.01, l1_ratio = 0.1)
model_Elnet.fit(x_train_res, y_train_res)
y_patientX = model_Elnet.predict(PatientX)

if y_patientX > 0.5:
    y_pred_px = "POSITIVE"
    prob = 100*y_patientX
else:
    y_pred_px = "NEGATIVE"
    prob = 100 - 100*y_patientX
 
print("The newly addmited patient has been classified %s for the death prediction.With a probability of %f %%" % (y_pred_px,prob))


# In[73]:

from sklearn.preprocessing import normalize

coeff = model_Elnet.sparse_coef_.todense()
coeff = np.squeeze(np.array(coeff[0,:]))
coeff_labels = list(X)

fig, ax = plt.subplots(figsize=(5,5))

plt.scatter(y = coeff_labels[10:19], x = coeff[10:19], marker = 's', s = 70, color = 'black')

plt.axvline(x=0, linestyle='--', color='black', linewidth=2)
ax.yaxis.grid(True)
plt.title('Value of the Categorical Coefficients for the Regression')
plt.savefig('Coeff_plot_cat.png', dpi = 250, bbox_inches="tight")

plt.show()

fig, ax = plt.subplots(figsize=(5,5))

plt.scatter(y = coeff_labels[1:10], x = coeff[1:10], marker = 's', s = 70, color = 'black')

plt.axvline(x=0, linestyle='--', color='black', linewidth=2)
ax.yaxis.grid(True)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_xlim([-0.0002, 0.0002])
plt.title('Value of the Numeric Coefficients for the Regression')
plt.savefig('Coeff_plot_num.png', dpi = 250, bbox_inches="tight")

plt.show()
