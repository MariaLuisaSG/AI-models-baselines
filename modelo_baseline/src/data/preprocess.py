import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(input_path, output_path):

    df = pd.read_csv(input_path)

    df = df.drop_duplicates()
    print(df.isna().sum())
    df = df.fillna(df.mean(numeric_only=True))
    X = df.drop("recurrence", axis=1)
    y = df["recurrence"]
    df = pd.read_csv(input_path, na_values=["?"])
    df = df.drop_duplicates()
    for col in df.select_dtypes(include="object"):
       df[col] = df[col].fillna(df[col].mode()[0])
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["recurrence"] = y.values

    df_processed.to_csv(output_path, index=False)
    print(df.isna().sum())
