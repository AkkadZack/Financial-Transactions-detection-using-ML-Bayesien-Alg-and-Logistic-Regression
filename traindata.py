from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv('transactions.csv',nrows=5000)
print(df.head())
print(df.info())
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', 'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
print('number of null values :',df.isnull().values.sum())
df= df.copy()
frauds = df.loc[df['isFraud'] == 1]
non_frauds = df.loc[df['isFraud'] == 0]
print("We have", len(frauds), "fraud data points and", len(non_frauds), "nonfraudulent data points.")

ax = frauds.plot.scatter(x='amount', y='isFraud', color='Orange', label='Fraud')
non_frauds.plot.scatter(x='amount', y='isFraud', color='Blue', label='Normal', ax=ax)
plt.show()

f, ax = plt.subplots(1, 1, figsize=(5, 3))
df.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8))
plt.show()


plt.figure(figsize=(12,8))
sns.boxplot(x = 'isFraud', y = 'amount', data = df[df.amount < 1e5])
plt.show()
df['type'] = df['type'].replace({'CASH_IN': '0', 'CASH_OUT': '1', 'DEBIT': '2', 'PAYMENT': '3', 'TRANSFER': '4'})
print(df.head())
X = df[['type','amount','oldBalanceOrig','newBalanceOrig','oldBalanceDest','newBalanceDest']]
y = df['isFraud']
print("X and y sizes, respectively:", len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
print("Train and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
print("Total number of frauds:", len(y.loc[df['isFraud'] == 1]), len(y.loc[df['isFraud'] == 1])/len(y))
print("Number of frauds on y_test:", len(y_test.loc[df['isFraud'] == 1]), len(y_test.loc[df['isFraud'] == 1]) / len(y_test))
print("Number of frauds on y_train:", len(y_train.loc[df['isFraud'] == 1]), len(y_train.loc[df['isFraud'] == 1])/len(y_train))

log_reg = LogisticRegression()
log_reg.fit(X_train , y_train)
y_predlog=log_reg.predict(X_test)
clf = GaussianNB()
clf.fit(X_train , y_train)
y_predNB=clf.predict(X_test)


Mat_confusionlog=confusion_matrix(y_test,y_predlog)
accuracydumodellog=accuracy_score(y_test, y_predlog)
rappeldumodellog=recall_score(y_test, y_predlog)
precisiondumodellog=precision_score(y_test, y_predlog)
scoreF1dumodellog=f1_score(y_test, y_predlog)

Mat_confusionNB=confusion_matrix(y_test,y_predNB)
accuracydumodelNB=accuracy_score(y_test, y_predNB)
rappeldumodelNB=recall_score(y_test, y_predNB)
precisiondumodelNB=precision_score(y_test, y_predNB)
scoreF1dumodelNB=f1_score(y_test, y_predNB)

print("la matrice de confusion du model de la regression logistique est : \n ", Mat_confusionlog)
print("la matrice de confusion du model bayesian naif est : \n ", Mat_confusionNB)
print("l'accuracy du model de la regression logistique  est : ",accuracydumodellog," \t l'accuracy du model bayesian naif  est : ",accuracydumodelNB)
print("le rappel du model de la regression logistique  est : ",rappeldumodellog,"\t le rappel du model bayesian naif  est : ",rappeldumodelNB)
print("la precision du model de la regression logistique  est : ",precisiondumodellog," \t la precision du model bayesian naif  est : ",precisiondumodelNB)
print("le score F1 du model de la regression logistique  est : ",scoreF1dumodellog," \t le Score F1 du model bayesian naif  est : ",scoreF1dumodelNB)
