# API du projet Iris

Cette API FastAPI charge le modèle entraîné par le composant
[`ml`](../ml/README.md) et expose ses prédictions en HTTP. Elle classe une fleur
parmi `setosa`, `versicolor` et `virginica` à partir de ses quatre mesures.

## Parcours du projet

1. Le composant ML entraîne et exporte le modèle ainsi que ses métadonnées.
2. L’API charge ces artefacts et met la prédiction à disposition via HTTP.
3. Une interface utilisateur pourra être ajoutée ultérieurement ; elle ne fait
   pas partie du projet actuel.

## Prérequis

- Python `>= 3.14` ;
- [uv](https://docs.astral.sh/uv/) ;
- les artefacts du modèle du composant [`ml`](../ml/README.md), à générer
  ci-dessous s’ils ne sont pas déjà présents.

## Installation

Depuis le dossier `api/`, installez les dépendances :

```bash
uv sync
```

## Préparer le modèle et configurer l’API

Si les artefacts n’existent pas encore, générez-les en suivant les étapes de
la section « Préparer le modèle » du [README ML](../ml/README.md). Par défaut,
ils sont écrits dans `ml/src/models/`.

Créez ensuite un fichier `.env` dans `api/` :

```dotenv
MODEL_PATH=../ml/src/models/iris_svc_model_v1.joblib
METADATA_PATH=../ml/src/models/iris_svc_metrics_v1.json
CORS_ORIGINS=["http://localhost:5173"]
```

Les chemins sont relatifs au dossier depuis lequel l’API est lancée.
`CORS_ORIGINS` est facultatif ; par défaut, l’origine autorisée est
`http://localhost:5173`. Elle pourra être adaptée lorsque l’interface sera
développée.

## Démarrer et utiliser l’API

Depuis `api/`, démarrez le serveur :

```bash
uv run uvicorn api.main:app --app-dir src --reload
```

L’API est disponible à l’adresse `http://127.0.0.1:8000`. La documentation
interactive est accessible sur `/docs` (ReDoc sur `/redoc`). Le modèle et ses
métadonnées sont chargés au démarrage ; si les chemins configurés sont absents
ou invalides, le démarrage échoue.

| Méthode | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Vérifie que le serveur répond. |
| `GET` | `/health` | Indique l’état du serveur et si le modèle est chargé. |
| `POST` | `/predict/` | Prédit l’espèce et renvoie les probabilités par classe. |

Les quatre mesures sont requises et doivent être comprises entre `0` et `10`
(inclus), en centimètres :

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

La réponse contient l’espèce prédite et la probabilité associée à chaque
classe :

```json
{
  "species": "setosa",
  "probabilities": {
    "setosa": 0.99,
    "versicolor": 0.01,
    "virginica": 0.0
  }
}
```

Les valeurs de cet exemple sont indicatives. Une requête invalide (mesure
manquante, valeur hors limites ou JSON incorrect) renvoie une erreur de
validation `422`.

## Interface utilisateur

L’API est prête à être consommée par un client, mais aucune interface
utilisateur n’est incluse à ce stade. Son développement est une étape future.

## Structure

```text
api/
├── pyproject.toml
└── src/api/
    ├── config.py      # configuration et variables d’environnement
    ├── deps.py        # dépendances FastAPI
    ├── main.py        # application et cycle de vie
    ├── predictor.py   # chargement du modèle et prédiction
    ├── routes.py      # routes HTTP
    └── schemas.py     # validation des entrées et sorties
```
