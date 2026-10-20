
import time
import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv("data/transactions.csv")

X = df.drop(columns="fraude")
y = df["fraude"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

modeles = {
    "reference-logistique": LogisticRegression(max_iter=1000),
    "manuel-rf-500-16": RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        random_state=42,
        n_jobs=-1
    ),
}

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp4-automl")

for nom, modele in modeles.items():
    t0 = time.time()
    modele.fit(Xtr, ytr)
    duree = round(time.time() - t0, 2)

    predictions = modele.predict(Xte)
    f1 = round(f1_score(yte, predictions), 4)

    with mlflow.start_run(run_name=nom):
        mlflow.log_params({
            "outil": "scikit-learn",
            "approche": nom
        })
        mlflow.log_metrics({
            "f1": f1,
            "duree_s": duree
        })

    print(f"{nom:22s} F1 = {f1} (entraînement : {duree} s)")
