# Baselines Qwen3.8-27B et référence Haiku — bilan au 3 novembre 2026

**Extension du bilan :** [autres datasets, méthodes et historique des 105 runs](project-experiments.md), avec les essais incomplets, échoués et annulés. [Inventaire CSV](project-runs.csv).

Ce document décrit les baselines exécutées dans cette session, leur protocole,
les résultats disponibles et les conditions de reprise. La baseline Qwen reste
**incomplète et en pause à la demande de l’utilisateur**. Cette publication
n’effectue aucun appel au modèle et ne relance aucun calcul.

## 1. Périmètre et état vérifié

L’objectif initial était d’évaluer Qwen3.8-27B sur **Causal-TimeBank train
(183 documents)** et **EventStoryLine train (233 documents)** avec notre recette
de vote documentaire à sept passes, sans modèle de décision. Les autres datasets
de la queue historique, notamment MAVEN-ERE, ne font pas partie de cette baseline.

La consigne suivante a donné priorité à CTB : terminer CTB avant de reprendre
ESC, puis arrêter. La dernière configuration active contient donc uniquement
`only: [r7_ctb]`.

L’état des fichiers distants a été revérifié le 3 novembre 2026 :

| Corpus | État de cette baseline Qwen | Suite autorisée |
|---|---|---|
| CTB | 179/183 documents réussis ; 4 documents en échec ; 10 passes valides manquantes | Pause ; attendre une instruction explicite de reprise |
| ESC train | 5/233 documents sauvegardés lors de l’ancienne exécution mixte ; aucune génération dans la dernière exécution CTB seule | Reste en pause ; aucune reprise automatique |

Pour CTB, les quatre documents incomplets ont déjà **18 passes valides** au
total. Le calcul restant minimal est donc `4 × 7 − 18 = 10` nouvelles réponses
valides. Les 179 documents réussis représentent 1 253 passes ; avec les passes
partielles, 1 271 des 1 281 passes prévues sont conservées. Les tentatives rejetées
ou tronquées ne comptent pas comme des passes valides.

| Document CTB incomplet | Passes valides conservées | Passes manquantes |
|---|---:|---:|
| AP900815-0044 | 6 | 1 |
| WSJ900813-0157 | 3 | 4 |
| WSJ910225-0066 | 6 | 1 |
| wsj_0585 | 3 | 4 |

Source : [instantané de progression](ctb179-progress.json), exécution
`b41468ee-d81e-4d4c-9df7-1f51309dbde9`. Le processus a quitté avec le code 1 et
aucun contrôleur de cette exécution n’était actif lors de la vérification.
L’interface de suivi conservait pourtant un statut `running` périmé : ce statut
n’est pas une preuve que le calcul continue.

## 2. Méthode d’inférence Qwen

La recette reprend les entrées CTB/ESC à sept passes de
[`queues/baseline_coref.yaml`](https://github.com/Fireman4740/workere/blob/f80a1a4678dde0105a329cb63e200ff6b1c2eed5/queues/baseline_coref.yaml), avec un
contrôleur Qwen dans
[`experiments/jev/qwen/run.py`](https://github.com/Fireman4740/workere/blob/f80a1a4678dde0105a329cb63e200ff6b1c2eed5/experiments/jev/qwen/run.py).

1. Une banque de 20 documents d’entraînement sert de réservoir de démonstrations.
2. Pour chaque démonstration, une explication CoT est synthétisée à partir de sa
   réponse de référence, puis réécrite pour retirer les formulations de
   connaissance préalable. Les annotations de référence servent à construire
   les démonstrations ; elles ne sont pas fournies comme réponse du document
   en cours d’évaluation.
3. Chaque passe utilise une démonstration différente. Les documents appartenant
   au même pli que le document évalué sont exclus de la sélection : 10 plis pour
   CTB et 5 pour ESC. Il ne s’agit pas d’un entraînement de modèles sur ces plis.
4. Sept sorties sont produites par document. Les relations absentes d’une sortie
   votent implicitement pour `norel`. Le vote est une pluralité, avec départage
   des égalités vers `norel`.
5. Le parseur et les métriques canoniques du dépôt évaluent les triplets de
   mentions. Aucun modèle de décision, aucune fusion avec d1 et aucun outil
   externe ne participent à cette baseline Qwen. Les options de whitelist
   oracle, de nombre de relations de référence et d’indices de coréférence de
   référence sont désactivées.

CTB utilise le prompt `causal_timebank_syntax` et cible les liens causaux dans
une même phrase. ESC utilise `esl_standard_eci_lean_coref`, dont la consigne
demande de couvrir les paires de mentions des événements causalement liés.

Les révisions de données sont fixées :

| Corpus | Dépôt Hugging Face | Révision | Documents | Paires candidates dirigées |
|---|---|---|---:|---:|
| CTB | `Nofing/CausalTimeBank-standard-eci-workere` | `1bba25094d1dda15f172c1f17bb88c26fb3d767d` | 183 | 19 442 |
| ESC | `Nofing/EventStoryLine-0.9-standard-eci-workere` | `e6418e8747ac1c7fecf66d56ae41fbde7fd5967a` | 233 | 122 130 |

Le plan complet représente 1 281 passes CTB et 1 631 passes ESC, soit 2 912
appels d’évaluation nominaux. La préparation de 20 démonstrations par corpus,
avec génération et réécriture, ajoute nominalement 80 appels : **2 992 appels
sans réessais et sans réutilisation de cache**. Ce total ne représente pas les
appels restant à faire après reprise.

## 3. Paramètres de la variante retenue

La configuration enregistrée est
[`experiments/jev/qwen/active.yaml`](https://github.com/Fireman4740/workere/blob/f80a1a4678dde0105a329cb63e200ff6b1c2eed5/experiments/jev/qwen/active.yaml), au
commit `f80a1a4678dde0105a329cb63e200ff6b1c2eed5`.

| Paramètre | Valeur utilisée |
|---|---|
| Modèle servi | `qwen3.8-27b` |
| Réglage de raisonnement envoyé | `reasoning_effort: xhigh` |
| Thinking | `enable_thinking: true`, `preserve_thinking: true` |
| Température | `0.0`, explicitement demandée par l’utilisateur |
| `top_p` / `top_k` / `min_p` | `0.95` / `20` / `0.0` |
| Pénalité de présence / répétition | `0.0` / `1.0` |
| Entrée maximale déclarée | 16 384 tokens, système et historique compris |
| Sortie maximale demandée | 32 768 tokens, raisonnement compris |
| Contexte total déclaré | 49 152 tokens |
| Concurrence HTTP globale | Jusqu’à 64 requêtes, limite partagée entre tâches |
| Timeout d’une requête | 86 400 secondes |
| Attente de disponibilité de l’API | Jusqu’à 1 800 secondes |

`xhigh` décrit le paramètre envoyé au serveur. Il ne constitue pas, à lui seul,
une mesure du raisonnement réalisé. Les réponses enregistrent séparément les
tokens de raisonnement et la raison d’arrêt. Ces réglages sont ceux exécutés ;
aucune recherche exhaustive d’hyperparamètres n’a établi leur optimalité.

Le service est un conteneur batch local à la box, accessible au contrôleur par
`http://127.0.0.1:18020/v1`. Les clés restent sur la machine d’exécution et ne
sont pas incluses dans les exports de résultats. Les entrées dépassant la
limite sont refusées ; aucune troncature automatique du prompt n’est prévue.

## 4. Historique des exécutions et des reprises

Les tentatives ci-dessous appartiennent à une même progression de baseline ;
elles ne constituent pas des répétitions indépendantes pour estimer une
variance.

| Exécution | Commit | Rôle et issue |
|---|---|---|
| `adcf6397-6bd8-4b2c-a5b6-cbf8b3a95566` | `99ea557` | Première variante sans thinking, température 0,7 ; annulée, aucun score complet |
| `8996f328-81a1-434c-8b72-e61fd9c86a33` | `b60e830` | Relance thinking/xhigh ; interruption du service avant évaluation exploitable |
| `cf753794-588e-41d9-8129-1b6d0c22978f` | `5a0404b` | Échec du moteur de génération ; pas de score d’évaluation complet |
| `d0ce6e2d-a39b-4770-891a-2394c17d596d` | `e0059db` | Variante xhigh, température 0 ; 9 documents CTB réussis avant arrêt |
| `6233eb3f-8d87-4e56-a769-70c6cea0ac5a` | `15c9fe2` | Reprise des caches et passes valides ; 11 documents CTB réussis avant annulation |
| `a082a2f5-96b1-4c75-a89f-8bbed54d2ed2` | `38e222e` | Reprise mixte ; sauvegarde de 173 CTB et 5 ESC réussis ; interruption puis annulation |
| `02a7955d-ff08-4b38-82fe-8813a18a1db2` | `f80a1a4` | Passage à CTB seul ; annulé, aucune nouvelle réponse valide |
| `5ac5da92-f52d-448b-bfbc-cc25fb9f89ce` | `f80a1a4` | Annulé pendant l’attente de l’API, aucune nouvelle réponse valide |
| `b41468ee-d81e-4d4c-9df7-1f51309dbde9` | `f80a1a4` | Dernière exécution CTB seule ; 179 réussites, 4 échecs ; arrêt du processus avec code 1 |

Les reprises copient les caches et les checkpoints réussis. Une passe partielle
n’est reprise que si sa génération est terminée normalement, si sa réponse est
parsable et si ses messages d’entrée correspondent exactement au prompt attendu.
Le contrôleur vérifie l’identité de configuration avant réutilisation.

La dernière exécution a importé 173 documents terminés et réutilisé 48 passes
partielles. Elle a ensuite enregistré **61 nouvelles tentatives d’inférence,
dont 49 échecs `TruncatedResponse`**. Ces comptes viennent du résumé distant ;
la copie locale du journal de suivi était restée partielle. Les 12 tentatives
valides supplémentaires ont permis de terminer 6 documents de plus.

## 5. Résultats préliminaires comparables à Haiku

La comparaison appariée disponible est figée à l’instantané du **31 octobre
2026 à 13:04:48 UTC**. Elle porte sur les **mêmes 163 documents CTB réussis**,
avec les mêmes annotations et le même scorer. Les 20 documents encore
incomplets à cet instant sont exclus des deux systèmes. La métrique est le F1
binaire micro de présence d’une relation causale sur les paires dirigées.

| Système, sans fusion ni modèle de décision | Précision | Rappel | F1 binaire micro | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 66,67 % | 61,22 % | **63,83 %** | 240 | 120 | 152 |
| Haiku 5.5 | 68,07 % | 57,65 % | **62,43 %** | 226 | 106 | 166 |

Qwen gagne **1,40 point de F1** sur ce sous-ensemble. Il retrouve davantage de
relations, avec davantage de faux positifs. Aucun test de significativité
n’a été effectué.

Cette comparaison porte sur deux systèmes existants, et n’isole pas le seul
effet du modèle : Qwen utilise des démonstrations CoT synthétisées et une
exclusion par pli ; Haiku utilise des réponses de référence seules et exclut
le document courant. Haiku filtre aussi les triplets aux paires candidates et
aux étiquettes valides avant le vote. Enfin, le sous-ensemble dépend de la
complétion des requêtes ; il n’est pas un échantillon aléatoire.

Les empreintes des prédictions Haiku ont été contrôlées contre leurs métadonnées,
et les annotations positives vérifiées document par document contre CTB Qwen.
Le calcul emploie `benchmark.report.build` et les étiquettes canoniques.

- [Résultats exacts, documents, comptes et empreintes sources](paired-ctb163.json).
- [Métriques appariées par document](paired-ctb163-per-document.csv).

Les exécutions sources sont `a082a2f5-96b1-4c75-a89f-8bbed54d2ed2` pour Qwen et
`a7f7bbd5-b115-4926-850a-af6eb7f44776` pour Haiku. Cette comparaison à 163 documents
n’a pas été recalculée sur les 179 documents maintenant réussis.

### Référence Haiku sur les corpus complets

Le journal Haiku contient également le champ autonome
`vote_alone_majority_4_of_7`, sans fusion avec le modèle de décision :

| Corpus | Documents | Précision | Rappel | F1 |
|---|---:|---:|---:|---:|
| CTB | 183 | 65,68 % | 59,73 % | 62,57 % |
| ESC train | 233 | 36,52 % | 51,61 % | 42,78 % |

Source : [extraction du champ de référence Haiku](haiku-reference.json). Cette
règle de majorité 4/7 et ces corpus complets sont à distinguer de la comparaison
appariée précédente ; on ne peut pas soustraire directement ces F1 au F1 Qwen
sur 163 documents pour conclure à un gain contrôlé.

### Luna et autres références

Aucune prédiction Luna CTB n’avait été retrouvée dans les sources utilisées
pour la comparaison appariée. Aucun F1 Luna sur ces mêmes documents n’est donc
publié ici. Les résultats Luna sur d’autres sous-ensembles ESC ne remplacent pas
une comparaison sur les mêmes documents. Ce document ne revendique pas de
nouvel état de l’art : la baseline Qwen complète n’est pas terminée.

## 6. Résumé incomplet à 183 lignes : interprétation

Le dernier résumé Qwen contient `documents: 183`, `failures: 4` et
`complete: false`. Cela signifie que les 183 documents ont une ligne de sortie,
pas que les 183 évaluations ont réussi.

Le F1 binaire qui y figure est **60,22 %** (précision 67,50 %, rappel 54,36 %).
Ce score inclut les quatre documents en échec, pour lesquels le runner écrit
une liste de prédictions vide. Il ne représente ni un F1 sur les 179 seuls
documents réussis, ni une baseline CTB entièrement réussie. Il ne doit pas être
comparé directement au 63,83 % de l’instantané apparié sur 163 documents.

Le résumé incomplet est conservé intégralement, à l’exception du chemin absolu
de la machine, dans [ctb179-progress.json](ctb179-progress.json).

## 7. Limite rencontrée et reprise future

Le blocage observé est la génération de raisonnements atteignant la limite de
32 768 tokens de sortie : le serveur renvoie `finish_reason: length`, et le
contrôleur rejette cette réponse au lieu de la compter comme valide. Augmenter
la seule capacité du serveur ne changera pas la limite demandée par le client.

L’utilisateur a demandé la pause pour reconfigurer le conteneur avec davantage
de tokens de sortie. **Aucune nouvelle limite n’a été appliquée ni testée dans
cette publication.** Pour reprendre, il faudra :

1. Recevoir une instruction explicite de reprise et vérifier les nouvelles
   limites d’entrée, de sortie et de contexte effectivement servies.
2. Adapter aussi `max_tokens` et les limites déclarées côté client, en conservant
   la traçabilité du changement de budget de génération.
3. Préserver les 179 documents réussis et les 18 passes partielles du dernier
   run. La configuration actuelle pointe encore vers `a082a2f5`, antérieur :
   la relancer telle quelle ferait perdre la progression récente dans la reprise.
4. Adapter et valider la reprise : le garde-fou actuel refuse une réutilisation
   lorsque `max_tokens` change. Il faut une autorisation de compatibilité
   explicite et documentée, ou une nouvelle variante, plutôt que contourner
   silencieusement cette vérification.
5. Terminer uniquement CTB, contrôler les 183 réussites, recalculer les métriques
   puis comparer Haiku sur le même ensemble. ESC reste soumis à une instruction
   de reprise distincte.

Les dix réponses manquantes sont un minimum de sorties valides à obtenir, pas
une garantie sur le nombre réel d’appels ou sur la durée restante.

## 8. Code publié et pièces de provenance

Les branches scientifiques conservent les commits réellement utilisés :

| Branche | Commit publié |
|---|---|
| `orx/qwen3-8-27b-document-vote-baseline-esc-train233` | `e0059db617bacb9ed3d4280947c01da495adfb1d` |
| `orx/qwen3-8-xhigh-temperature-zero-resume-esc-train2` | `38e222e3b926fe3ced3818fcddb6e3f64d963358` |
| `orx/qwen-ctb-completion-only-esc-held-until-ctb-fini` | `f80a1a4678dde0105a329cb63e200ff6b1c2eed5` |
| `orx/h1-haiku-5-5-xhigh-x-d1-ultra-frozen-fusion-esc` | `36de1a08bc427fb8936a34dabbfeb0035ba47b8f` |

La branche Haiku contient aussi ses expériences de fusion ; les chiffres
retenus ici sont explicitement les sorties Haiku autonomes. La documentation
et les exports compacts sont ajoutés séparément dans
`publication/qwen38-baselines-2026-11-03`, dérivée de `f80a1a4`, sans modifier les
snapshots scientifiques et sans fusion dans `main`.

Les données publiées ici sont les paramètres, métriques, identifiants de
documents et empreintes de provenance. Les checkpoints bruts, prompts complets,
réponses CoT et bases de cache restent dans les sauvegardes de recherche ; les
exports de métriques ne suffisent pas à reconstruire ces réponses. Le fichier
[manifest.json](manifest.json) permet de vérifier les empreintes des pièces
publiées. Les comptes agrégés du tableau apparié permettent de vérifier le F1
par `2 × TP / (2 × TP + FP + FN)`.
