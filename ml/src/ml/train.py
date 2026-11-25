from sklearn.model_selection import train_test_split

from . import config
from .data import load_data
from .evaluate import evaluate
from .export import export_model_and_metadata
from .pipeline import build_model


def main():
    # Données
    X, y = load_data()
    # Découpage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=config.SEED, stratify=y, test_size=config.TEST_SIZE
    )
    # Model
    model = build_model()
    # Entrainement
    model.fit(X_train, y_train)
    print("Model entrainé sur les données d'entrainement ! ")
    # Prediction et évaluation
    score = evaluate(model, y_test=y_test, y_pred=model.predict(X_test))
    # Export du model et des métadonnées
    export_model_and_metadata(model=model, metric=score, features_names=list(X.columns))


if __name__ == "__main__":
    main()
