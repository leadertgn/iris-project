# Projet Iris

Projet de classification des espèces de fleurs Iris à partir de quatre mesures
en centimètres : longueur et largeur des sépales, longueur et largeur des
pétales. Le dépôt contient actuellement le composant d’apprentissage ML et une
API HTTP qui expose les prédictions du modèle.

## Parcours du projet

1. Installer les dépendances et entraîner le modèle dans `ml/`.
2. Configurer puis démarrer l’API dans `api/`.
3. Envoyer une requête à l’API pour obtenir l’espèce prédite et les
   probabilités par classe.

L’interface utilisateur est envisagée pour une étape ultérieure. Elle n’est pas
incluse dans le projet actuel.

## Prérequis

- Python `>= 3.14` ;
- [uv](https://docs.astral.sh/uv/).

## Démarrage rapide

Les étapes ci-dessous supposent que les commandes sont lancées depuis la racine
du dépôt.

### 1. Entraîner le modèle

Installez les dépendances ML et générez le modèle ainsi que ses métadonnées :

```bash
cd ml
uv sync
uv run python -m ml.train
```

Les artefacts sont créés dans `ml/src/models/`. Les détails sur les données,
l’exploration, l’entraînement et les versions du modèle sont dans le
[README ML](ml/README.md).

### 2. Configurer et démarrer l’API

Dans un nouveau terminal, configurez l’API. Créez `api/.env` avec les chemins
des artefacts générés à l’étape précédente :

```dotenv
MODEL_PATH=../ml/src/models/iris_svc_model_v1.joblib
METADATA_PATH=../ml/src/models/iris_svc_metrics_v1.json
CORS_ORIGINS=["http://localhost:5173"]
```

Puis installez les dépendances et démarrez le serveur :

```bash
cd api
uv sync
uv run uvicorn api.main:app --app-dir src --reload
```

L’API est disponible à `http://127.0.0.1:8000`. Sa documentation interactive
se trouve sur [`/docs`](http://127.0.0.1:8000/docs). Les détails de
configuration et les routes sont dans le [README API](api/README.md).

### 3. Tester une prédiction

Depuis un autre terminal, envoyez les quatre mesures à la route de prédiction :

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

La réponse contient une espèce (`setosa`, `versicolor` ou `virginica`) et les
probabilités estimées pour chaque classe. Les mesures doivent être comprises
entre `0` et `10` inclus.

## Structure du dépôt

```text
.
├── ml/       # données, entraînement, évaluation et artefacts du modèle
└── api/      # API FastAPI pour exposer les prédictions
```

## Documentation des composants

- [Composant ML](ml/README.md)
- [API](api/README.md)
