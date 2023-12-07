import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib


def prepare_data():
    my_data = pd.read_csv(r'../drug200.csv')
    X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F', 'M'])
    X[:, 1] = le_sex.transform(X[:, 1])

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
    X[:, 2] = le_BP.transform(X[:, 2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit(['NORMAL', 'HIGH'])
    X[:, 3] = le_Chol.transform(X[:, 3])

    y = my_data["Drug"]
    return X, y


def train_model(X, y, criterion="entropy", max_depth=4, splitter='best'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    drugTree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, splitter=splitter)
    drugTree.fit(X_train, y_train)
    filename = 'finalized_model.sav'
    joblib.dump(drugTree, filename)


if __name__ == '__main__':
    X, y = prepare_data()
    train_model(X, y)
