from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from . import config
def evaluate(pipeline, y_test, y_pred):
    # score 
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy score : ", score)

    # matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    print("=== Matrice de confusion ===")
    print(cm)

    # rapport de classification

    clr = classification_report(y_test, y_pred, target_names=pipeline.classes_)
    print("=== Rapport de classification ===")
    print(clr)

    return score

