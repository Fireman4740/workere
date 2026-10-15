# Architecture multipipelines

Le scoring historique (`data`, `labels`, `metrics`, `report`, `voting`) reste figé.
L’extraction de ces modules est vérifiée contre le commit `dev` `a253e5e` ; la parité
avec les runs historiques reste en attente de leurs fichiers d’origine.

Le format `workere-prediction/v2` et les protocoles du framework sont ajoutés autour de
ce cœur. Une réévaluation relit la suite et les prédictions enregistrées sans charger
le registre ni aucun modèle. Les interfaces sont dans `interfaces.py`, les objets
publics dans `schema.py` et le registre versionné dans `registry.py`.

```text
YAML → résolution/validation → suite figée + métadonnées de reproductibilité
                              │
                              ├─ entrée modèle sans annotations de relations
                              │    ├─ EntityExtractor → CandidateGenerator
                              │    │                    → RelationPredictor → PostProcessor(s)
                              │    └─ Pipeline monolithique → Prediction
                              │
                              └─ JSONL versionné → évaluateurs → métriques + bootstrap
                                                                 → comparateur
```

Les documents visibles par les modèles contiennent les tokens d’origine, des frontières
de phrases et une représentation textuelle canonique, sans le marquage gold `<ID ...>`.
Le protocole `relations_gold` est explicitement oracle : il fournit les mentions de
référence et les candidats historiques, jamais les labels de relations. L’extraction
et le bout en bout n’ont accès ni aux mentions ni aux candidats de référence.

Le moteur conserve la suite, la configuration résolue, les entrées publiques, les
prédictions, les erreurs et les mesures de coût/latence. Les campagnes et sweeps
déclinent des configurations ; ils ne modifient pas les évaluateurs. Les accès réseau
et les composants pouvant facturer sont refusés par défaut avant leur construction.

L’ajout d’un modèle consiste à implémenter une interface, enregistrer le composant avec
sa version et écrire un YAML. Aucun changement du moteur d’évaluation n’est nécessaire.
Les plugins Python sont du code de confiance : l’API évite les fuites accidentelles de
gold ; elle ne constitue pas un sandbox contre un plugin lisant arbitrairement le disque.
