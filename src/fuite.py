
import numpy as np
import pandas as pd

from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv("data/transactions.csv")

# Variable créée après la transaction : fuite de données.
rng = np.random.default_rng(0)

df["num_dossier_litige"] = np.where(
    df["fraude"] == 1,
    rng.integers(10000, 99999, len(df)),
    0
)

X = df.drop(columns="fraude")
y = df["fraude"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

automl = AutoML()

automl.fit(
    Xtr,
    ytr,
    task="classification",
    metric="f1",
    time_budget=30,
    seed=42,
    verbose=0
)

print("Famille retenue :", automl.best_estimator)

modele = automl.model.estimator

if hasattr(modele, "feature_importances_"):
    importances = sorted(
        zip(X.columns, modele.feature_importances_),
        key=lambda paire: -paire[1]
    )

    print(
        "Importance des variables :",
        [(nom, round(float(valeur), 3))
         for nom, valeur in importances]
    )

predictions_test = automl.predict(Xte)

print(
    "F1 test avec la colonne piégée :",
    round(f1_score(yte, predictions_test), 4)
)

Xprod = Xte.copy()
Xprod["num_dossier_litige"] = 0

predictions_prod = automl.predict(Xprod)

print(
    'F1 "en production" (colonne vide) :',
    round(
        f1_score(yte, predictions_prod, zero_division=0),
        4
    )
)
