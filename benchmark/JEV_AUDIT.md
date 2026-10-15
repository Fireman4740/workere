# Audit préalable JEV v1 — 15 octobre 2026

## Point de départ et isolation

Branche de travail `benchmark/multipipeline-framework`, issue de `dev` à `a253e5e`.
Le worktree principal conserve `dev` ; les autres worktrees ne sont pas modifiés.
L'extraction indépendante précédente est reprise par `eb9b826`. Les fichiers
`run.py`, `config.yaml`, `eci/runner.py`, les clients d'inférence historiques et
`baselines/decision_eci*.py` restent inchangés.

## Chaîne historique retrouvée

`run.py` charge la configuration via `eci/config.py`, puis le runner historique.
`config.yaml` contient choix de dataset/split, shuffle, labels, contraintes de paires,
resampling et options d'inférence. Les prompts/few-shot/CoT/outils/clients/cache LLM,
le runner asynchrone et MLflow appartiennent à l'inférence, pas au nouvel évaluateur.

Les loaders d'origine (`eci/data.py`, maintenant réexport de `benchmark/data.py`)
produisent des documents comprenant texte, tokens, mentions et spans parallèles,
phrases, candidats et relations gold. Le texte historique contient des marques
`<ID surface>` ; ce texte annoté n'est pas fourni au nouvel extracteur d'événements.
L'état modèle v1 utilise les tokens joints par des espaces et uniquement les deux
mentions de référence autorisées. Les offsets concernent ce texte canonique.

`Doc.pair_list` est conservé après le même chargement et le même shuffle. Si le
corpus ne fournit pas de liste, le loader génère les couples ordonnés de mentions
distinctes. Aucune suppression de négatifs n'est ajoutée. La classification Noul
est appelée pour chaque couple exact ; aucune direction causale n'est inventée.

Les cinq modules `data`, `labels`, `metrics`, `voting`, `report` sont repris à logique
identique ; les tests comparent leur source à `dev`, à l'exception des chemins
imports. L'évaluateur conserve les particularités historiques : union des paires
gold/prédites/votées pour les lignes scorées, labels hors convention mesurés,
normalisation et vote seulement dans les cas prévus, scopes inconnus exclus des
deux sous-scores intra/inter. Il ne remplace pas cela par une autre métrique.
Pour v1 toutes les paires candidates ont une prédiction explicite, y compris norel.

L'extraction d'événements exacte non typée et les relations end-to-end par spans
sont des tâches séparées et versionnées. Elles ne remplacent pas le score historique
avec mentions gold. CTB/ESC/MAVEN ont les spans nécessaires ; aucune garantie de
couverture d'une NER générique ou typée n'est déduite de ces annotations.

## Meilleur ESC `e77431d6` : ce qui est établi et ce qui manque

Sources locales : `notes/posthoc_sweep_e77431d6_dfe087fb.md`,
`notes/eci_sota_to_beat.md`, `notes/pass_variance.*`, `queues/baseline_coref.yaml`.
Ces notes décrivent le dépôt **corrigé**
`Nofing/EventStoryLine-0.9-standard-eci-workere`, 233 documents train,
DeepSeek-v4-flash, 7 passes, prompt `esl_standard_eci_lean_coref`, une démonstration
aléatoire par passe parmi un pool de 20 avec CoT, syntaxe désactivée dans le YAML.
Elles rapportent **46.6 de F1 binaire** pour la majorité à quatre votes. Les variantes
post-hoc à trois votes et avec propagation oracle de coréférence ont des protocoles
différents et ne sont pas reprises comme scores de la baseline de base.

Les fichiers originaux `.config.yaml`, rapport JSON, checkpoint/documents et traces
n'ont pas été retrouvés dans le checkout ni aux emplacements inspectés. La queue
et les notes ne démontrent pas une configuration exacte. **La parité historique
n'est donc pas démontrée et la migration n'est pas déclarée terminée.** L'importeur
refuse les différences de configuration, annotations, candidats et métriques.

Le `config.yaml` actuel cible encore le dépôt ESC standard non corrigé. Les YAML
JEV sélectionnent explicitement le dépôt corrigé et sa révision
`e6418e8747ac1c7fecf66d56ae41fbde7fd5967a` sans modifier la configuration historique.
Son split dev contient 25 documents et 19028 candidats ordonnés (2696 intra-phrase,
16332 inter-phrases). Un pilot de 50 documents distincts est impossible sur ce split.
Les essais smoke utilisent un document complet choisi par sa taille, pas ses labels :
`41_9ecbplus.xml`, 13 mentions, 156 candidats. Les métriques du smoke ne sont pas
comparables au résultat historique sur 233 documents train.

## Pourquoi les premiers essais semblaient proches du hasard

`notes/decision_models_eci.md` indique explicitement que **JEV n'avait pas été testé**,
faute de clé. Les observations proches du hasard concernent Laya. Le protocole était
intra-phrase, avec paires non ordonnées, phrase marquée par « », trois formulations
(dir3/bin2/nli2) et probabilités d'options. Ce n'est pas le protocole complet v1.

Les notes rapportent une forte préférence pour l'option A, une faible discrimination
entre paires d'une même phrase et une calibration vers le taux de base. Une refonte
de température n'a pas rendu cette expérience concluante. Ces résultats ne sont pas
reproduits ici : les fichiers bruts du pilot ne sont pas disponibles. On ne peut pas
attribuer rétrospectivement ce comportement à une erreur d'API ou à un défaut JEV.

Le nouveau client distingue erreurs API, réponses structurées invalides, limites de
contexte, budgets, candidats et prédictions. Il n'analyse pas du JSON généré librement.
Span exige un état textuel et Noul ; Mercury accepte aussi Choice. Ces contraintes sont
validées avant les appels. Les réponses brutes et versions servies permettent une
inspection du mapping ; les probabilités ne sont jamais jetées après seuil.

## Validation disponible

La non-régression du cœur est contrôlée contre le code de `dev`, sur fixtures, votes,
labels atypiques, scopes et modes historiques. Les tests du pipeline historique
utilisent un modèle scripté, sans API réelle. Les tests de parité sur rapports agentere
et de reconstruction des sources sont ignorés lorsque leurs fichiers manquent ;
ce sont des limites de preuve, pas des succès de migration.

Les vrais essais gratuits, leurs scores, analyses, coûts et artefacts sont publiés
dans le rapport durable `jev-v1/jev-v1-report.md`. Les sorties mock sont clairement
identifiées et ne servent pas de preuve de performance de JEV.
