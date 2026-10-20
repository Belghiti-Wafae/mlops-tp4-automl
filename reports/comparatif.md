
# Comparatif des approches - projet fraude (TP04)

Date de consultation des documentations cloud : 20/10/2026

| Approche | F1 test | Temps humain | Temps machine | Coût estimé | Explicabilité | Export / réversibilité |
|---|---:|---|---|---|---|---|
| Logistique (référence) | À mesurer | Faible | À mesurer | 0 € de facturation cloud, hors électricité | Bonne, coefficients interprétables | Bonne, modèle scikit-learn |
| Manuel RF (TP2) | 0,8669 au TP2 ; à vérifier ici | Moyen/élevé | À mesurer | 0 € de facturation cloud, hors électricité | Moyenne/faible | Bonne, modèle scikit-learn |
| FLAML 60 s | À mesurer | Moyen | À mesurer | 0 € de facturation cloud, hors électricité | Dépend du modèle retenu | Bonne, selon le modèle et ses dépendances |
| Vertex AI AutoML | Non mesuré | Faible à moyen | Non mesuré | Selon les ressources et la durée | Rapports et outils disponibles selon le service | À vérifier selon le format du modèle |
| SageMaker Autopilot | Non mesuré | Faible à moyen | Non mesuré | Calcul, stockage et déploiement facturés selon usage | Rapports de performance et d'explicabilité | À vérifier selon le modèle et le format |
| Azure Automated ML | Non mesuré | Faible à moyen | Non mesuré | Calcul et déploiement selon usage | Métriques et outils d'explication disponibles | Export possible selon le modèle et le format |

## Commentaire

1. La régression logistique est adaptée à une première référence rapide et interprétable.
2. Le réglage manuel est utile lorsque les choix techniques sont compris et que le contrôle est prioritaire.
3. FLAML est pertinent pour explorer automatiquement plusieurs candidats avec un budget limité.
4. Une plateforme cloud managée est intéressante pour industrialiser l'entraînement, le suivi et le déploiement.
5. Une solution locale est préférable pour limiter les dépendances cloud, maîtriser les données et réduire les coûts récurrents.
