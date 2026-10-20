
import sys
import json
from collections import Counter

fichier = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "reports/flaml_60s.log"
)

essais = []

with open(fichier, encoding="utf-8") as f:
    for ligne in f:
        ligne = ligne.strip()

        if not ligne.startswith("{"):
            continue

        try:
            e = json.loads(ligne)
        except json.JSONDecodeError:
            continue

        if (
            "learner" in e
            and e.get("validation_loss") is not None
        ):
            essais.append(e)

print(len(essais), "essais journalisés dans", fichier)

if not essais:
    print(
        "Aucun essai reconnu. Vérifiez le format du journal FLAML."
    )
    raise SystemExit(1)

print(
    "Essais par famille :",
    dict(Counter(e["learner"] for e in essais))
)

meilleurs = {}

for e in essais:
    famille = e["learner"]

    if (
        famille not in meilleurs
        or e["validation_loss"]
        < meilleurs[famille]["validation_loss"]
    ):
        meilleurs[famille] = e

print("\nMeilleur F1 de validation par famille :")

for famille, e in sorted(
    meilleurs.items(),
    key=lambda paire: paire[1]["validation_loss"]
):
    print(
        f"  {famille:20s} "
        f"F1 val = {1 - e['validation_loss']:.4f} "
        f"(temps = {e.get('wall_clock_time', 0):.1f} s)"
    )

gagnant = min(
    essais,
    key=lambda e: e["validation_loss"]
)

print("\nEssai gagnant :")
print("Famille :", gagnant["learner"])
print("Configuration :", gagnant.get("config"))
print("Perte de validation :", gagnant["validation_loss"])
