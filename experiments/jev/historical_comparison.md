# Mercury et baseline LLM ESC

La baseline `e77431d6` est décrite dans `notes/eci_sota_to_beat.md` et
`notes/posthoc_sweep_e77431d6_dfe087fb.md` : DeepSeek-v4-flash, un exemple annoté
par passe, sept passes et vote majoritaire, 233 documents ESC corrigé, split
`train`. Le F1 binaire **rapporté** est 46,6 %. Les tableaux divergent légèrement
sur certains arrondis. Le score reste à reproduire depuis les prédictions originales.

Le smoke test Mercury précédent (`654d1602-1b6e-47f3-a88e-aa992adbe7b4`)
utilise seulement `41_9ecbplus.xml` sur `dev`. Son F1 de 58,33 % ne peut pas
être comparé à la baseline sur 233 documents.

## Nouvelle expérience

`esc_mercury_noul_train233.yaml` fixe ESC corrigé, split `train`, seed 42,
233 documents, mentions de référence, tous les candidats historiques ordonnés,
question Noul inchangée et seuil 0,5. Aucun réglage du seuil sur ce split.
La révision des données est `e6418e8747ac1c7fecf66d56ae41fbde7fd5967a`, publiée
le 26 septembre 2026 ; aucune nouvelle révision de données n'était publiée au
contrôle du 15 octobre 2026.

Le dry-run compte 122 130 paires : 17 998 intra-phrase et 104 132 inter-phrases.
À 20 appels/minute : environ 101,8 heures au minimum, hors quotas et reprises.
Les deux ordres restent évalués indépendamment ; aucun négatif n'est supprimé.

Le fichier `esc_e77431d6_evaluation_reconstructed.yaml` contient uniquement les
paramètres d'évaluation reconstruits par fusion de `config.yaml` et de l'entrée
`r7_esc` de `queues/baseline_coref.yaml`. Ce n'est **pas** le snapshot original.
Le paramètre historique des sept passes reste dans l'identité du banc ; Mercury
produit directement des prédictions finales et n'utilise pas ce vote. Les différences
de modèle, de prompting et d'inférence sont celles des pipelines comparés.
Seules les métriques binaires sont pertinentes pour Mercury Noul, qui ne prédit
pas la direction causale. « Inter » signifie inter-phrases dans un document.

```sh
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli dry-run \
  --config experiments/jev/esc_mercury_noul_train233.yaml --output logs/esc-train233-plan.json
```

L'expérience suivie utilise le contrat existant `bash experiments/jev/run.sh`,
avec cette configuration copiée dans `active.yaml`. Le provider reçoit le texte,
les mentions et la question ; les relations annotées restent dans l'évaluateur.
La route est exclusivement `inception/mercury-decide:free`, le budget est nul,
la version servie est vérifiée, et chaque tentative est journalisée.
Ne pas remplacer cette route par sa variante payante en cas de limite.

Le cache SQLite conserve les réponses réussies ; un nouveau lancement avec les
mêmes entrées les réutilise. Une erreur d'API interrompt l'évaluation (`raise`) :
aucun score complet n'est publié pour un corpus partiellement traité.

## Vérifier la baseline et comparer

Il faut retrouver le rapport JSON original de `e77431d6`, contenant `config`,
`results.per_pair_predictions` et `results.per_doc_metrics`. Conserver aussi
`docs.jsonl` et le snapshot YAML s'ils sont disponibles. Les tableaux Markdown
et `pass_variance.json` ne permettent pas de reconstruire ces prédictions.

```sh
python -m benchmark verify-history /chemin/original/run_e77431d6.json \
  --output logs/e77431d6-parity.json
python -m benchmark import-history --suite /chemin/run-mercury/suite.json \
  --report /chemin/original/run_e77431d6.json --model deepseek-v4-flash-e77431d6 \
  --output logs/e77431d6-imported.json
python -m benchmark compare-runs /chemin/run-mercury logs/e77431d6-imported.json \
  --metric binary --output logs/mercury-vs-llm.json --markdown logs/mercury-vs-llm.md
```

Ces commandes ne rappellent aucun modèle. L'import vérifie les documents, les
annotations, les scopes, les paramètres d'évaluation et les métriques ; il
échoue si les données originales diffèrent de la reconstruction. Dans ce cas,
repartir du vrai snapshot, puis réutiliser les probabilités Mercury en cache
pour les entrées identiques. Les intervalles et différences appariées au niveau
document ne peuvent être calculés qu'après récupération des prédictions LLM.
