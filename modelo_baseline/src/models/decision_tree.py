from sklearn.tree import DecisionTreeClassifier
def create_model():
    model= DecisionTreeClassifier(random_state=42)
    return model