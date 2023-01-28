import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_excel("/Users/agustintumminello/Desktop/pro/input/train.xlsx")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df["Churn Value"].values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    df.to_csv("/Users/agustintumminello/Desktop/pro/input/train_folds.csv", index=False)
