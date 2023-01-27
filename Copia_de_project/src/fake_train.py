# src/train.py
import joblib
import pandas as pd
from sklearn import metrics 
from sklearn import tree
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def run():
    df = pd.read_csv("/Users/agustintumminello/Desktop/Copia_de_project/input/train.csv")

    df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    categorical_cols = list(X.select_dtypes(include="object"))
    numerical_cols = list(X.select_dtypes(exclude="object"))

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ])

    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessing = ColumnTransformer([
        ("categorical_preprocessing", categorical_pipe, categorical_cols),
        ("numerical_preprocessing", numerical_pipe, numerical_cols)
    ])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    X_train = preprocessing.fit_transform(X_train)

    X_valid = preprocessing.transform(X_valid)


    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()
    # fit the model on training data
    clf.fit(X_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(X_valid)
    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds) 
    print(f"Accuracy={accuracy}")

    # save the model
    joblib.dump(clf, f"/Users/agustintumminello/Desktop/Copia_de_project/models.bin")

if __name__ == "__main__": 
    run()

