# Rapport de migration du benchmark

Date : 15 octobre 2026. Base : `dev`, commit `a253e5e`. Branche dédiée :
`benchmark/standalone-evaluation`, dans le worktree de cette session.

**Conclusion : extraction implémentée et testée ; migration non déclarée terminée.
La parité des scores historiques reste à démontrer faute de rapports/prédictions source.**

## Audit et séparation des responsabilités

| Fichiers audités | Rôle et décision |
|---|---|
| `run.py`, `eci/runner.py`, `eci/config.py`, `config.yaml` | orchestration LLM, configuration stricte liée aux prompts, resampling, construction du rapport ; conservés sans modification |
| `eci/data.py`, `eci/labels.py` | split, sampling, candidats, spans et conventions de labels ; déplacés dans `benchmark/` |
| `eci/voting.py` | agrégation des passes, votes implicites négatifs, égalités et conversion binaire ; déplacé sans changement logique |
| `eci/metrics.py`, `eci/report.py` | arithmétique, univers scoré, intra/inter, agrégats par document et langue ; déplacés sans changement logique |
| `eci/parse.py`, `eci/scope.py` | parsing/normalisation de réponses, blocs et whitelists ; conservés côté producteur, non copiés |
| `eci/fewshot.py`, `eci/prompts.py`, `eci/llm.py`, `eci/tools.py`, `eci/tracking.py` | dépendances d’inférence et de suivi ; non incluses dans le benchmark |
| `eci/posthoc/runs.py`, `eci/posthoc/rescore.py`, `analyze.py` | résolution de rapports, recalcul, transformations des votes ; conservés, utilisent toujours les imports compatibles |
| `dataprep/*.py`, notes d’audit existantes | builders et variantes corrigées ; lus pour comprendre la provenance, aucune annotation reconstruite ou remplacée |
| `baselines/decision_eci.py`, `baselines/decision_eci_recal.py` | protocole décisionnel distinct : dépôts corrigés, candidats intra non ordonnés, calibration et métriques probabilistes ; non modifiés et non amalgamés aux scores historiques |
| `tests/test_parity.py`, notes de résultats existantes | références aux anciens runs ; aucune prédiction historique rejouable disponible ici |

Les notes historiques contiennent des résultats récapitulatifs, pas les décisions par
paire requises pour redémontrer les scores. Aucun résultat de ces notes n’a été repris
comme preuve nouvelle de parité.

## Fichiers réutilisés, modifiés et ajoutés

Les modules canoniques `benchmark/{data,labels,metrics,report,voting}.py` reprennent
exactement le contenu original après substitution des imports `eci` vers `benchmark`.
Un test compare le texte aux fichiers du commit `a253e5e`, puis un oracle séparé charge
ce code original pour tester indépendamment les sorties. Les anciens cinq fichiers
`eci/` sont remplacés par des réexports ; leurs fonctions sont les mêmes objets Python.

`README.md` ajoute l’accès au benchmark. Aucun fichier de configuration, de dépendances,
de prompts, d’annotations, de scoring des baselines, ni de test préexistant n’est modifié.

Nouveaux éléments :

- `benchmark/suite.py` : suite figée avec empreinte, export d’entrées sans gold et folds conservés.
- `benchmark/evaluate.py` : interface JSONL/API pour décisions finales ou passes ; détection d’événements séparée ; comparaison de suites compatibles.
- `benchmark/history.py` : rejeu en lecture seule de rapports/checkpoints et import conditionné à la parité.
- `benchmark/io.py`, `benchmark/__main__.py` : I/O sans écrasement et CLI indépendante.
- `benchmark/audit.py` : audit reproductible des versions des quatre datasets configurés.
- `benchmark/examples/` : classifieur externe, démonstration synthétique et producteur OpenRouter pour les deux modèles demandés.
- `benchmark/README.md`, ce rapport et `tests/test_benchmark.py` : contrats, workflow, limites et validation.

## Conventions conservées

Les splits et l’ordre initial sont ceux des dépôts configurés. Le shuffle global seedé
précède la coupe ; les IDs et `idx` ne sont pas recalculés après shuffle. Les candidats
restent la colonne `pair_list` ou tous les couples ordonnés distincts. Les folds servent
à l’exclusion de démonstrations, pas à changer le split évalué.

L’univers des métriques reste l’union gold/prédictions/votes. Les paires hors candidats
ne sont pas supprimées par un nouveau filtre. La normalisation des labels n’est pas
ajoutée au scorer ; elle reste la responsabilité du producteur. Les variantes négatives,
labels hors convention, dénominateurs nuls et exclusion des phrases inconnues des deux
sous-ensembles sont conservés. Aucune métrique historique n’a changé.

Le nouveau protocole d’extraction non typée utilise la fonction `set_metrics` existante
avec égalité exacte des indices de tokens. Il est explicitement distinct des relations.
Il ne couvre ni la NER générique, ni les types d’événements, ni les relations de bout en
bout entre mentions prédites et annotations gold.

## Vérification des datasets

L’audit a lu les quatre dépôts actifs, à leurs révisions enregistrées dans le livrable
`dataset-audit.json`, et vérifié tous les splits :

| Dataset configuré | Documents train / dev / test | Mentions train / dev / test | `pair_list` |
|---|---|---|---|
| MAVEN-ERE | 2 621 / 292 / 710 | 66 325 / 7 614 / 17 780 | non |
| MECI | 2 157 / 718 / 716 | 28 324 / 9 138 / 9 172 | oui |
| CausalTimeBank | 183 / — / — | 6 807 / — / — | oui |
| EventStoryLine | 233 / 25 / — | 4 668 / 666 / — | non |

Aucun span hors limites ni document sans annotations parallèles `mentions`/`spans`
n’a été trouvé. Aucune colonne `mention_type` n’existe dans ces versions. Les données
autorisent donc une tâche séparée de détection de mentions d’événements non typées.
La version historique de CTB conserve ses 6 807 mentions, sans substitution par les
6 811 du builder corrigé. Une suite des 50 documents ESC sélectionnés par le fichier
de configuration existant est fournie comme point de départ, avec ses entrées sans gold.

## Tests exécutés et écarts

| Vérification | Résultat |
|---|---|
| `uv sync --locked` | environnement existant reproduit ; manifestes et lock inchangés |
| `python -m unittest tests.test_benchmark tests.test_units -v` | **37 tests réussis**, dont 16 nouveaux |
| `python -m unittest tests.test_pipeline -v` | **12 tests réussis**, modèles simulés, corpus ESC chargé |
| Parité avec le code original | 48 configurations (3 modes dirigé/binaire, 2 règles d’égalité, 8 seeds), rapports identiques |
| CLI autonome | copie du seul dossier `benchmark/`, Python `-S`, scoring et refus d’écrasement validés |
| Contrôles négatifs | gold altéré, F1 corrompu, rapport incomplet, IDs dupliqués/absents, suite modifiée et comparaisons incompatibles refusés |
| `python -m unittest tests.test_parity tests.test_esc_dataprep tests.test_ctb_dataprep tests.test_maven_dataprep -v` | **21 tests ignorés**, dont 5 de parité historique ; fichiers historiques/sources brutes absents |
| Démonstration complète | entrées, deux producteurs jouets, scores et comparaison générés sur données synthétiques |
| Connexion des deux modèles demandés | réponses réelles de Mercury Decide et Span-01 Lite sur un exemple synthétique ; pas une mesure ECI |
| `git diff --check` | réussi |

Total : **49 tests réussis ; 21 ignorés ; aucune preuve de parité des anciens runs**.
La tolérance de rejeu est absolue $10^{-12}$. Les tests du pipeline affichent des tokens
et coûts simulés : ils ne correspondent pas à des appels LLM facturés. Les essais
OpenRouter réussis sauvegardés indiquent chacun un coût de zéro dans leur réponse.

Deux problèmes d’environnement rencontrés ont été résolus : accès en écriture aux caches
uv/Hugging Face nécessaire dans le sandbox. Graphviz `dot` est absent ; les tests
existants avertissent qu’ils ne rendent pas leurs graphes, puis passent. Le benchmark
n’utilise pas Graphviz. Les premiers essais d’API ont permis de corriger le contrat de
Span : état textuel et questions `noul` au lieu d’un `choice`. Aucun de ces essais ne
portait sur des prédictions historiques.

## Condition restante pour clôturer la migration

Fournir les rapports JSON historiques, idéalement leurs `docs.jsonl`, configurations
et traces. Exécuter `verify-history` sur chaque rapport, puis `import-history` sur une
suite figée issue de la configuration correspondante, et examiner tous les écarts.
Les deux niveaux de preuve (`checkpoint-to-report` ou `report-rows-only`) sont explicites.
Les sources sont lues sans modification et les sorties ne les écrasent pas.

Il n’y a ni run historique inventé, ni résultat synthétique rebaptisé historique, ni
appel LLM destiné à régénérer une baseline manquante. La clôture reste en attente de
cette étape. Voir `benchmark/README.md` pour les commandes exactes et les formats.
