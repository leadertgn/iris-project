# Composant ML

Ce dossier contient le pipeline d'apprentissage et d'inférence du projet Iris.
Le modèle classifie une fleur parmi `setosa`, `versicolor` et `virginica` à
partir de ses quatre mesures :

- longueur du sépale (`sepal length (cm)`) ;
- largeur du sépale (`sepal width (cm)`) ;
- longueur du pétale (`petal length (cm)`) ;
- largeur du pétale (`petal width (cm)`).

## Prérequis

- Python `>= 3.14` ;
- [uv](https://docs.astral.sh/uv/) ;
- un environnement compatible avec les dépendances déclarées dans
  [`pyproject.toml`](pyproject.toml).

## Installation

Depuis ce dossier :

```bash
uv sync
```

Les dépendances de développement (Jupyter, IPython kernel et Ruff) sont
également installées par défaut avec `uv sync`.

## Données

Le jeu de données utilisé pour l'entraînement est
[`src/data/raw/iris.csv`](src/data/raw/iris.csv). La colonne `species` est la
cible ; toutes les autres colonnes sont utilisées comme variables explicatives.

Le chargement est centralisé dans [`src/ml/data.py`](src/ml/data.py), tandis
que les paramètres du projet sont définis dans
[`src/ml/config.py`](src/ml/config.py).

## Entraînement

Pour entraîner le modèle et générer les artefacts :

```bash
uv run python -m ml.train
```

Le script :

1. charge les données Iris ;
2. effectue une séparation entraînement/test stratifiée (`80 %` / `20 %`) ;
3. construit un pipeline avec imputation médiane, standardisation et
   `SVC(kernel="linear", C=0.1)` calibré pour produire des probabilités ;
4. évalue le modèle sur l'échantillon de test ;
5. exporte le modèle et ses métadonnées.

Le modèle est construit par [`src/ml/pipeline.py`](src/ml/pipeline.py) et
l'entraînement est orchestré par [`src/ml/train.py`](src/ml/train.py).
La graine utilisée pour le découpage est `42`, afin de rendre l'évaluation
reproductible.

## Artefacts générés

Les fichiers sont écrits dans [`src/models/`](src/models/) :

- `iris_svc_model_v<version>.joblib` : pipeline scikit-learn sérialisé ;
- `iris_svc_metrics_v<version>.json` : accuracy, variables, classes, version
  du modèle, date d'entraînement et version de scikit-learn.

La version courante est définie par `MODEL_VERSION` dans
[`src/ml/config.py`](src/ml/config.py). Après une modification du modèle ou
des données, incrémenter cette version avant de relancer l'entraînement.

## Prédiction

La fonction [`predict`](src/ml/predict.py) charge automatiquement le modèle et
ses métadonnées, puis accepte un dictionnaire contenant les quatre mesures :

```python
from ml.predict import predict

result = predict({
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
})

print(result)
# Exemple de structure :
# {
#     "species": "<classe prédite>",
#     "probabilities": {"setosa": 0.0, "versicolor": 0.0, "virginica": 0.0}
# }
```

Une `ValueError` est levée lorsqu'une mesure attendue est absente. Les
probabilités sont renvoyées pour chacune des classes connues du modèle.

## Exploration

Le notebook [`src/notebooks/01_exploration.ipynb`](src/notebooks/01_exploration.ipynb)
sert à explorer les données. Pour lancer Jupyter :

```bash
uv run jupyter notebook
```

## Structure

```text
ml/
├── pyproject.toml
├── src/
│   ├── data/raw/iris.csv       # données brutes
│   ├── ml/
│   │   ├── config.py           # chemins et hyperparamètres
│   │   ├── data.py             # chargement des données
│   │   ├── pipeline.py         # définition du pipeline
│   │   ├── train.py            # entraînement et export
│   │   ├── evaluate.py         # métriques et rapport
│   │   ├── export.py           # sérialisation des artefacts
│   │   └── predict.py          # inférence
│   ├── models/                 # modèle et métadonnées exportés
│   └── notebooks/              # exploration
```