# Banc d’évaluation indépendant

`benchmark/` contient le chargement des données, le vote et le scoring, sans client LLM,
prompts, few-shot, outils, spaCy ou MLflow. Les cinq modules historiques ont été déplacés
sans changement de logique : `data`, `labels`, `metrics`, `voting`, `report`. Les imports
`eci.*` correspondants restent disponibles comme réexports des mêmes fonctions.

**Statut de migration : validation historique en attente.** Les rapports et prédictions
des anciens runs ne sont pas fournis dans ce checkout. Les tests synthétiques et la
comparaison avec le code de `dev` ne remplacent pas leur rejeu. Les commandes ci-dessous
permettent de vérifier et d’importer ces résultats dès qu’ils sont accessibles.

## Démarrage hors ligne

Depuis la racine du dépôt, avec Python 3.12 ou ultérieur :

```bash
python3 -m benchmark.examples.demo --output-dir /tmp/workere-demo
python3 -m benchmark compare /tmp/workere-demo/toy-norel.result.json /tmp/workere-demo/toy-lexical.result.json --output /tmp/workere-demo/compare-again.json
```

Cet exemple utilise uniquement des données **synthétiques**, deux prédicteurs jouets et
la bibliothèque standard. Il écrit une suite, les entrées sans gold, les prédictions et
les scores. Il ne mesure aucune performance sur les corpus du projet.

Pour utiliser l’environnement du dépôt : `uv sync --locked`, puis préfixer les commandes
Python par `uv run --locked`. Le scoring, le rejeu et la comparaison n’ont aucune
dépendance externe. On peut copier uniquement le dossier `benchmark/` dans un autre
répertoire et y exécuter `python3 -S -m benchmark ...`. Le chargement depuis Hugging Face
nécessite `datasets` ; lire une configuration YAML nécessite `PyYAML`. Ces dépendances
sont déjà déclarées et verrouillées à la racine. Aucun fichier de prompt n’est requis.

## Créer une suite reproductible

```bash
uv run --locked python -m benchmark freeze --config config.yaml --output logs/benchmarks/esc-suite.json
uv run --locked python -m benchmark inputs --suite logs/benchmarks/esc-suite.json --output logs/benchmarks/esc-inputs.jsonl
```

`freeze` lit le fichier YAML directement, sans le validateur de prompts du runner. Il
reprend le dataset actif, son **split exact**, le plafond de documents, le seed, l’ordre
du split, les shuffles, les candidats, les annotations et les folds. Il embarque les
tokens, spans, IDs et données nécessaires au scoring. Son empreinte SHA-256 porte sur
le JSON canonique du contenu complet ; modifier la suite invalide son empreinte.
Réutiliser le **même fichier de suite** pour tous les modèles comparés.

Pour un export local, fournir `--rows split.jsonl` : ce fichier doit contenir **le split
entier dans son ordre original**, avant tout échantillonnage. La configuration peut
également être en JSON. Ne pas substituer silencieusement un export corrigé aux données
historiques. L’audit reproductible des dépôts configurés s’exécute séparément :

```bash
uv run --locked python -m benchmark.audit --config config.yaml --output logs/benchmarks/dataset-audit.json
```

L’audit enregistre les révisions Hugging Face et vérifie les spans sur tous les splits.
La suite reste l’entrée figée faisant foi : un dépôt distant peut évoluer depuis un
ancien run ; l’import historique vérifiera alors les golds et les périmètres.

## Interface de prédictions de relations

L’API Python principale est `benchmark.evaluate.evaluate(suite, predictions, model=...)`.
Le format JSONL contient une ligne par document, y compris les prédictions vides :

```json
{"id":"doc1","triples":[["e0","causes","e1"],["e1","causedby","e0"]]}
{"id":"doc2","triples":[]}
```

Les extrémités sont les IDs de mentions du dataset, pas leurs surfaces ni leurs indices
dans une liste. Les triples sont `[source, label, cible]`. Les deux orientations sont
distinctes : l’adaptateur doit émettre les décisions qu’il souhaite faire évaluer ; le
scorer n’ajoute pas automatiquement la relation inverse. Pour un protocole déjà binaire,
les labels attendus sont `causal` et `norel`, et les paires finales sont ordonnées
lexicographiquement. Les prédictions `final` doivent être déjà sous leur forme finale.

```bash
python3 -m benchmark.examples.external --inputs logs/benchmarks/esc-inputs.jsonl --output logs/benchmarks/model-a.predictions.jsonl
python3 -m benchmark score --suite logs/benchmarks/esc-suite.json --predictions logs/benchmarks/model-a.predictions.jsonl --model model-a --output logs/benchmarks/model-a.result.json
```

L’exemple `external.py` fournit une fonction `classify(tokens, spans, source, target)` à
remplacer par un encodeur, un classifieur local ou un modèle décisionnel. Son comportement
par défaut est une baseline jouet qui répond toujours `norel`. Pour un JEV renvoyant une
distribution de choix, l’adaptateur peut par exemple faire :

```python
label_of = {"A": "causes", "B": "causedby", "C": "norel"}
probabilities = jev_response["answers"]["q"]["probabilities"]
label = label_of[max(probabilities, key=probabilities.get)]
```

Les critères A/B/C de la requête doivent correspondre exactement à cette convention.
Conserver la configuration du producteur, ses poids/version, le seuil et les distributions
à côté du résultat. Ne jamais choisir un seuil sur les annotations du split de test.

Pour réutiliser le vote historique, fournir `--representation passes` avec :

```json
{"id":"doc1","passes":[[["e0","causes","e1"]],[],[["e0","causes","e1"]]]}
```

Le nombre de passes doit correspondre à la suite. Ce sont les triples **déjà parsés et
prétraités**, après réunion des blocs et application des éventuels filtres/whitelists
du producteur. Le benchmark reprend ensuite les votes implicites `norel`, le dernier
label émis par paire et par passe, les égalités seedées et les conversions binaires
historiques. Il n’interprète pas une réponse textuelle et n’exécute aucun outil LLM.

Des documents absents, supplémentaires ou dupliqués provoquent une erreur ; une liste
vide explicite reste une prédiction valide. Les labels hors convention et les IDs
inconnus restent scorés, comme historiquement ; aucune normalisation ni suppression
silencieuse n’est appliquée. Le champ `per_label` et le compteur de phrases inconnues
permettent de les inspecter. Tous les fichiers de sortie refusent l’écrasement.

## Mercury Decide et Span-01 Lite

L’exemple optionnel appelle directement la
[Decisions API OpenRouter](https://openrouter.ai/respan/span-01-lite%3Afree), sans reprendre
le pipeline LLM. Les deux modèles ont été testés sur un exemple synthétique. Mercury
accepte un `choice` A/B/C ; Span exige ici un état textuel et deux questions `noul`.
Pour Span, le plus grand des deux scores strictement supérieur au seuil détermine la
direction ; un score insuffisant ou une égalité produit `norel`. Ces deux probabilités
oui/non sont indépendantes et ne forment pas une distribution A/B/C.

```bash
# OPENROUTER_API_KEY doit être défini ; OPENROUTER est également reconnu.
python3 -m benchmark.examples.openrouter_decisions --smoke --output logs/benchmarks/api-smoke.json
python3 -m benchmark.examples.openrouter_decisions --inputs logs/benchmarks/esc-inputs.jsonl --model inception/mercury-decide:free --output logs/benchmarks/mercury.predictions.jsonl
python3 -m benchmark.examples.openrouter_decisions --inputs logs/benchmarks/esc-inputs.jsonl --model respan/span-01-lite:free --threshold 0.5 --output logs/benchmarks/span.predictions.jsonl
```

Les commandes `--inputs` classent **tous les candidats de tous les documents** : elles
effectuent de nouveaux appels autorisés au producteur. Elles produisent aussi les
réponses/probabilités brutes et les métadonnées. L’exemple produit des labels dirigés ;
pour une suite binaire, ajouter explicitement la conversion dans le producteur. Le
rejeu des anciennes prédictions ci-dessous n’effectue aucun appel API.

## Rejouer l’historique et démontrer la parité

```bash
python3 -m benchmark verify-history /archives/run_OLD.json /archives/run_OTHER.json --output logs/benchmarks/historical-parity.json
```

Deux niveaux sont distingués dans le résultat :

1. `checkpoint-to-report` si `docs.jsonl` accompagne le rapport : reconstruction des
   scores à partir des prédictions finales, du gold et des votes persistés, puis
   comparaison des lignes par paire et des métriques disponibles.
2. `report-rows-only` sinon : recalcul depuis `per_pair_predictions`, avec les scores
   globaux, par document, par langue et intra/inter. Cela vérifie l’arithmétique du
   rapport ; cela ne redémontre pas son parsing ou son vote d’origine.

Les résultats indiquent chaque écart, le nombre de comparaisons et les empreintes des
entrées. Tolérance absolue : $10^{-12}$, sans tolérance relative. La commande retourne
un code non nul en cas d’écart, d’absence d’entrée ou de format incompatible. Elle ne
modifie aucun fichier historique. Les anciens tests `tests.test_parity` restent utiles
si les traces agentere et les sources sont accessibles : ils vérifient aussi parsing,
prompts et vote. Un test ignoré faute de données ne constitue pas une réussite.

Pour comparer une baseline historique à un nouveau modèle sur la **même suite** :

```bash
python3 -m benchmark import-history --suite logs/benchmarks/esc-suite.json --report /archives/run_OLD.json --model historical-baseline --output logs/benchmarks/baseline.result.json
python3 -m benchmark compare logs/benchmarks/baseline.result.json logs/benchmarks/model-a.result.json --output logs/benchmarks/comparison.json
```

L’import exige la même configuration d’évaluation, les mêmes documents, le même gold,
les mêmes scopes intra/inter et les mêmes métriques. Les lignes négatives enregistrées
préservent aussi les paires rejetées par vote. La baseline importée représente les
**décisions finales**, pas des passes qui ne sont pas disponibles. L’import échoue si
une divergence existe. `compare` exige une empreinte de suite, une tâche, un protocole
et une représentation identiques ; il ne calcule pas une intersection de documents.

## Métriques et conventions conservées

| Élément | Comportement historique conservé |
|---|---|
| Documents | split configuré ; shuffle global seedé puis coupe `max_examples` ; `idx` reste l’indice du split |
| Folds | `doc_idx % n_folds` ; exclusion des démonstrations du même fold, **pas** une nouvelle partition train/test |
| Candidats | `pair_list` si disponible ; sinon tous les couples ordonnés de mentions distinctes ; shuffle seedé par ID |
| Paires effectivement scorées | union du gold, des prédictions et des paires votées ; aucun ajout de tous les vrais négatifs |
| Contraintes de candidats | instructions du producteur historique, pas un filtre ajouté au scorer |
| Labels | `causes`, `causedby`, `norel` ; `causal` en binaire ; variantes négatives `norel`, `none`, `no_rel`, `unknown` |
| Moyennes multi-classes | exclusion des variantes négatives ; inclusion des labels positifs hors convention observés |
| Binaire | tout label non négatif est positif ; métriques dirigées sauf conversion explicitement configurée |
| Intra/inter | phrase du premier token de la mention ; paires non résolues exclues des deux sous-ensembles |
| Dénominateurs nuls | P, R et F1 valent zéro |
| Échecs historiques | un document en échec dans un checkpoint reste scoré avec ses prédictions persistées, normalement vides |

Les scripts `baselines/decision_eci*.py` ont un autre protocole : dépôts `*-workere`,
paires intra-phrases non ordonnées, traitement distinct des paires non annotées et
échantillonnage propre. Leurs scores ne deviennent pas comparables par simple renommage
des labels. Reproduire leur sélection dans une suite distincte ou produire de nouvelles
prédictions sur les entrées figées ici. Aucun de ces scripts n’a été modifié.

## Extraction d’événements, séparée des relations

Les quatre dépôts configurés contiennent `tokens`, `mentions` et `spans` : leurs annotations
permettent une évaluation de la **détection de mentions d’événements non typées**. Les
exports historiques configurés n’ont pas de colonne `mention_type` : ce ne sont pas un
gold de NER générique (personnes, organisations, etc.) ni une baseline d’extraction typée.
Les anciens runs utilisaient les mentions gold ; leurs F1 de relations ne mesurent pas
la capacité à trouver ces mentions.

La tâche optionnelle suivante est **nouvelle**. Elle reprend `set_metrics` existant,
avec correspondance exacte de l’ensemble des indices de tokens de chaque span, y
compris les spans discontinus, et micro P/R/F1 sur tous les documents. Les doublons
sont dédupliqués comme des ensembles. Elle n’altère aucune métrique historique.

```bash
python3 -m benchmark inputs --suite logs/benchmarks/esc-suite.json --task event_mentions --output logs/benchmarks/event-inputs.jsonl
python3 -m benchmark score --suite logs/benchmarks/esc-suite.json --task event_mentions --predictions logs/benchmarks/event-predictions.jsonl --model external-extractor --output logs/benchmarks/event-result.json
```

```json
{"id":"doc1","spans":[[0],[5,6],[9,11]]}
{"id":"doc2","spans":[]}
```

Les indices sont globaux dans le document et commencent à zéro ; les listes sont triées,
sans répétition, et ne sont pas des intervalles `[début, fin]`. Les entrées de cette tâche
ne contiennent que tokens, frontières de phrases, langue et ID : aucun marquage gold.
Le protocole `event-mentions-exact-untyped/v1` est distinct des relations. L’alignement
de relations entre spans prédits et gold pour une évaluation de bout en bout reste hors
de ce protocole.

## Validation

```bash
uv run --locked python -m unittest tests.test_benchmark tests.test_units -v
uv run --locked python -m unittest tests.test_pipeline -v
uv run --locked python -m unittest tests.test_parity -v
```

Les nouveaux tests comparent les fonctions extraites au commit original `a253e5e`,
exercent 48 configurations seed/vote/binaire et vérifient couverture, empreintes,
imports compatibles, rejeu, erreurs de parité, extraction, producteurs décisionnels,
comparaison et CLI sans aucune dépendance installée. Les tests du pipeline utilisent
des modèles simulés, même lorsque leurs logs affichent des coûts ou un nom de modèle.
Leur chargement initial peut télécharger les datasets publics, sans appeler de LLM.
