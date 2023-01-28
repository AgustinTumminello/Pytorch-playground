import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

def run(fold):
    df = pd.read_csv("/Users/agustintumminello/Desktop/pro/input/train_folds.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    X_train = df_train.drop("Churn Value", axis=1)
    y_train = df_train["Churn Value"].values

    X_valid = df_valid.drop("Churn Value", axis=1)
    y_valid = df_valid["Churn Value"].values

    categorical = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Multiple Lines", 
                "Internet Service", "Online Security", "Online Backup", "Device Protection", "Tech Support", 
                "Streaming Movies", "Contract", "Paperless Billing", "Payment Method"]

    numerical = ["Zip Code"]

    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ])

    preprocessing = ColumnTransformer([
        ("numerical", numerical_pipe, numerical),
        ("categorical", categorical_pipe, categorical)
    ])

    X_train = preprocessing.fit_transform(X_train)
    X_valid = preprocessing.transform(X_valid)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"fold={fold}, accuracy={accuracy}")

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
