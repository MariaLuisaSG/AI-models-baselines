from sklearn.metrics import accuracy_score, f1_score

def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    return {"accuracy": acc, "f1": f1}
