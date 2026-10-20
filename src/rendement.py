import pandas as pd
import mlflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")

runs = mlflow.search_runs(
    experiment_names=["tp4-automl"]
)

if runs.empty:
    raise RuntimeError("Aucun run trouvé dans tp4-automl.")

auto = runs[
    runs["params.outil"].isin(["flaml", "sklearn-maison"])
].copy()

if auto.empty:
    raise RuntimeError("Aucun run AutoML trouvé.")

auto["budget"] = pd.to_numeric(
    auto["params.budget_s"], errors="coerce"
)

auto = auto.dropna(subset=["budget", "metrics.f1"])
auto = auto.sort_values("start_time")
auto = auto.drop_duplicates(
    ["params.outil", "budget"], keep="last"
)

refs = runs[
    runs["params.outil"] == "scikit-learn"
].sort_values("start_time")

plt.figure(figsize=(8, 5))

for outil, groupe in auto.groupby("params.outil"):
    groupe = groupe.sort_values("budget")
    plt.plot(
        groupe["budget"],
        groupe["metrics.f1"],
        "o-",
        label=f"AutoML ({outil})"
    )

    for _, ligne in groupe.iterrows():
        print(
            f"{outil} budget {ligne['budget']:g} s "
            f"-> F1 {ligne['metrics.f1']:.4f}"
        )

for _, ligne in refs.iterrows():
    nom = ligne.get("tags.mlflow.runName", "Référence")
    f1 = ligne["metrics.f1"]

    plt.axhline(f1, linestyle="--", linewidth=1)
    plt.text(
        10, f1 + 0.003, f"{nom} ({f1:.4f})", fontsize=8
    )
    print(f"{nom}: F1 {f1:.4f}")

plt.xscale("log")
budgets = sorted(auto["budget"].unique())
plt.xticks(budgets, [str(int(b)) for b in budgets])
plt.minorticks_off()
plt.xlabel("Budget de calcul (secondes, échelle logarithmique)")
plt.ylabel("F1 sur le jeu de test")
plt.title("Rendement du budget de calcul")
plt.legend()
plt.tight_layout()
plt.savefig("reports/rendement.png", dpi=150)
print("Graphique créé : reports/rendement.png")
