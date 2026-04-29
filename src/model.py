from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_model(name):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)

    elif name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42)

    elif name == "svm":
        return SVC(
            kernel="rbf",        
            C=1.0,               
            gamma="scale",       
            probability=True, 
            random_state=42
        )

    else:
        raise ValueError("Invalid model name")