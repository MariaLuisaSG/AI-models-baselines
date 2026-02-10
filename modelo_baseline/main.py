from src.data.load_data import load_data
from src.models import decision_tree
from src.models import logistic_regression
from src.models import random_forest
from src.models.train import train, evaluate
from src.evaluation.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    df = load_data("data/processed/breast_cancer_clean.csv")
    X = df.drop("recurrence", axis=1)
    y = df["recurrence"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    #model = create_model()
    #model = create_tree_model()
    #model = create_logistic_model()
    #model = train(model, X_train, y_train)
    #metrics = evaluate(model, X_test, y_test)
    models ={
        "Decision Tree": decision_tree.create_model(),
        "Logistic Regression":logistic_regression.create_model(),
        "Random forest": random_forest.create_model(),
    }
    resultados = {}
    for name, model in models.items():
        model = train(model, X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        resultados[name] = metrics
        print(f"{name}: {metrics}")
        plot_confusion_matrix(model, X_test, y_test)


if __name__ == "__main__":
    main()
