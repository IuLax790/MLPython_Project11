import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

Loans = pd.read_csv("loan_sanction_train.csv")
Loans.drop(columns='Loan_ID',axis=1,inplace=True)


columns = ['Gender', 'Married', 'Self_Employed', 'Dependents', 'Credit_History', 'Property_Area']

fig, axes = plt.subplots(6,2, figsize=(25, 30))
axes = axes.flatten()
for i, col in enumerate(columns):
    sns.countplot(data=Loans, x=col, ax=axes[i*2])
    sns.countplot(data=Loans, x=col, hue='Loan_Status', ax=axes[i*2 + 1])

#filling nan values for numeric and categorical variables
Loans['Gender'].fillna(Loans['Gender'].value_counts().idxmax(), inplace=True)
Loans['Married'].fillna(Loans['Married'].value_counts().idxmax(), inplace=True)
Loans['Dependents'].fillna(Loans['Dependents'].value_counts().idxmax(), inplace=True)
Loans['Self_Employed'].fillna(Loans['Self_Employed'].value_counts().idxmax(), inplace=True)
Loans["LoanAmount"].fillna(Loans["LoanAmount"].mean(skipna=True), inplace=True)
Loans['Loan_Amount_Term'].fillna(Loans['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
Loans['Credit_History'].fillna(Loans['Credit_History'].value_counts().idxmax(), inplace=True)

gender_stat = {"Female": 0, "Male": 1}
yes_no_stat = {'No' : 0,'Yes' : 1}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}
status = {'N':0,'Y':1}
#encoding categorical variables.
Loans['Gender'] = Loans['Gender'].replace(gender_stat)
Loans['Married'] = Loans['Married'].replace(yes_no_stat)
Loans['Dependents'] = Loans['Dependents'].replace(dependents_stat)
Loans['Education'] = Loans['Education'].replace(education_stat)
Loans['Self_Employed'] = Loans['Self_Employed'].replace(yes_no_stat)
Loans['Property_Area'] = Loans['Property_Area'].replace(property_stat)
Loans['Loan_Status'] = Loans['Loan_Status'].replace(status)

print(Loans.isnull().sum())


X = Loans.iloc[:,:-1]
y = Loans.iloc[:,-1]



plt.figure(figsize=(10,10))
print(sns.heatmap(Loans.corr(), annot=True, fmt='.2f'))
print(plt.show())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def logistic_regression(X_train,X_test,y_train,y_test):
  from sklearn.metrics import accuracy_score,classification_report
  LR = LogisticRegression(random_state = 0)
  LR.fit(X_train,y_train)
  print("Logistic Regression\n",classification_report(y_test, LR.predict(X_test)),"\n")
  print(accuracy_score(y_test, LR.predict(X_test)),"\n")


def SVM(X_train,X_test,y_train,y_test):
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score,classification_report
  svc = SVC(kernel = 'linear', random_state = 0)
  svc.fit(X_train,y_train)
  print("SVM\n",classification_report(y_test, svc.predict(X_test)),"\n")
  print(accuracy_score(y_test, svc.predict(X_test)),"\n")

def kernSVM(X_train,X_test,y_train,y_test):
  from sklearn.svm import SVC
  from sklearn.ensemble import AdaBoostClassifier
  from sklearn.metrics import accuracy_score,classification_report
  kernelsvm = SVC(kernel = 'rbf', random_state = 42)
  kernelsvm.fit(X_train,y_train)
  print("KernSVM\n",classification_report(y_test, kernelsvm.predict(X_test)),"\n")
  print(accuracy_score(y_test, kernelsvm.predict(X_test)),"\n")

def Naive_Bayes(X_train,X_test,y_train,y_test):
  from sklearn.naive_bayes import GaussianNB
  from sklearn.metrics import accuracy_score,classification_report
  NB = GaussianNB()
  NB.fit(X_train,y_train)
  print("Naive_Bayes\n",classification_report(y_test, NB.predict(X_test)),"\n")
  print(accuracy_score(y_test, NB.predict(X_test)),"\n")

logistic_regression(X_train,X_test,y_train,y_test)
SVM(X_train,X_test,y_train,y_test)
kernSVM(X_train,X_test,y_train,y_test)
Naive_Bayes(X_train,X_test,y_train,y_test)

