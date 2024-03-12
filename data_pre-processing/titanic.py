import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages

df = pd.read_csv('/Users/hyunjoolee/Downloads/train.csv')
cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1) #axis 0이면 row, 1이면 column

# df = df.dropna() na있는 row 드랍한다
dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
# print(titanic_dummies)
df = pd.concat((df, titanic_dummies), axis=1)
df = df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
df['Age'] = df['Age'].interpolate() #age에 na 좀 있어서 interpolation으로 채움
# print(df.info())

### NumPy ###
X = df.values
y = df['Survived'].values
X = np.delete(X, 1, axis = 1) #칼럼 1을 지운다

### Divide Training Data and Test Data ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
