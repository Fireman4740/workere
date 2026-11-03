# Autres datasets, méthodes et historique complet des runs

Instantané du 2026-11-03T14:11:42.106548+00:00. **65 nœuds et 105 runs** ; les brouillons sans run sont conservés dans les fiches. Ce document étend le [bilan Qwen–Haiku](README.md) et inclut les autres sessions du projet.

## Lire cet inventaire

Les statuts enregistrés sont : 56 `done`, 26 `failed`, 21 `cancelled` et 2 `running`. Ce sont des états techniques à la date de capture. `done` ne garantit pas que le protocole scientifique ait produit un score valide ; une vérification hors ligne sans génération peut aussi finir `done`.

La baseline Qwen CTB est **en pause** : son état `running` du suivi est périmé, alors que le processus a quitté avec code 1. Ses 179 documents réussis et ses dix passes manquantes sont documentés dans le bilan principal. Le run N14 appartient à une autre campagne et apparaît en cours ; il n’a été ni lancé ni interrompu pour cette publication. Aucun calcul n’est relancé ici.

Un journal a été retrouvé pour 104/105 runs. Le run `9c35861e-10df-4bc9-8a72-fd993498e768` a échoué à la soumission, avant l’envoi des sources ; aucun journal d’exécution n’a été localisé. Cette absence est conservée dans l’inventaire.

## Datasets et sous-ensembles couverts

| Famille | Périmètres effectivement étudiés ou tentés | Précaution |
|---|---|---|
| ESC / EventStoryLine | Smoke d’un document, DEV-SCREEN10, DEV25, D15, CONF40, TEST-S40, DEV44, RES30, sous-lots DEV8/10/12/16/20 et ESC train233 | Ces sous-ensembles n’ont pas les mêmes documents ni le même rôle ; plusieurs ont été réutilisés pour le développement. |
| Causal-TimeBank | CTB183 ; attribution hors ligne sur unités verrouillées ; N6 sur sept documents ; N13 prévu CTB50 avec smoke initial de quatre ; N14 sur quatre documents | Les smokes N13 n’autorisent aucun F1 primaire CTB50. CTB183 Qwen reste incomplet. |
| MAVEN-ERE et autres corpus présents dans le code | Aucun run de cette famille identifié dans les 105 runs recensés | La présence d’une configuration ou d’une queue ne démontre pas une exécution. |

DEV, CONF et TEST désignent ici le rôle historique d’un sous-ensemble dans une campagne, pas une garantie actuelle de jeu indépendant. Les étapes de confirmation, de calibration supervisée, de diagnostic et d’audit sont distinguées dans les fiches.

## Familles de méthodes incluses

- Span, Mercury et d1-ultra : décision par paire, choix à trois classes, contexte ciblé, symétrisation, seuils, récupération de démonstrations et questions décomposées.
- Luna, Haiku et Qwen : lecteurs documentaires, votes répétés, rationales, représentations de mentions et changements de budget de génération.
- Fusion et juges : Luna/Haiku × d1, Harness², vérification DeepSeek, calibrations J2/J3/J4 et cascades/ensembles K. Ces méthodes peuvent inclure un modèle de décision, contrairement à la baseline Qwen seule.
- Couverture et ancrage : alias A/R, variantes O/P/M/MP, diversité complémentaire, preuve locale, placebos et attribution CTB.
- Campagnes N : validation des ajouts, ablations 1/3/7 lectures, démonstrations contrastives, réplications, factoriels et arrêt anticipé exact.

Les poids des modèles API ne sont généralement pas réentraînés, mais J3-B/J4 emploient des calibrateurs supervisés ; les seuils et démonstrations utilisent aussi des labels de développement. « Sans fine-tuning » ne signifie pas « sans supervision ».

## Comment sont rapportés les résultats

Les bilans historiques ci-dessous reprennent la synthèse technique consolidée du 3 novembre, dont les rapports sources et leurs empreintes figurent dans [la provenance](project-source-provenance.json). Les états des runs et l’index des preuves ont été rafraîchis pour cette publication ; N13, N14 et la continuation CTB ont été actualisés. Les tests historiques ne sont pas réexécutés ici, et aucun audit complet de toutes les réponses brutes n’est revendiqué.

Les F1 ne forment pas un classement global : datasets, documents, règles de vote, représentations, calibrations et états de validité diffèrent. Un score partiel est étiqueté comme tel. Un gate échoué, un timeout ou une réponse tronquée n’est ni un vote négatif ni un F1 nul. Les gains cités sont ceux des comparaisons appariées des rapports, avec leurs limites.

**Correction RES30 (J3)** : les F1 originaux des journaux sont invalides à cause d’une incohérence entre gold canonique et candidats symétriques. Les scores corrigés font foi : base 36,0698 ; J3-A 36,3525 ; J3-B 35,9878. Les fiches et l’inventaire signalent le remplacement des métriques originales ; aucun ancien F1 J3 n’est exporté dans l’index de preuves. Les noms de corpus, comptes et variantes ne doivent pas être supposés identiques sans vérifier la révision et le plan exécuté.

## Fiches des expériences et de toutes leurs tentatives


### Index des fiches

- [01. Span : validation initiale du modèle de décision](#experience-694058c6-1973-46a5-8bc5-3579eada24ad)
- [02. Mercury : remplacement du modèle, protocole inchangé](#experience-87cddd6f-7910-4913-8e2f-4fe3441ea33f)
- [03. Mercury Choice : décision à trois classes](#experience-2fbe084a-f105-4726-be67-815b8cf98c77)
- [04. Mercury Context : rendre explicites les occurrences ciblées](#experience-293c814f-d7a3-477f-80e2-dae63e9774b1)
- [05. Mercury sur ESC233 : passage à l'échelle interrompu](#experience-3afb71c4-e110-4a31-a9eb-899fc1419b68)
- [06. Prototype d1 DEV25+D15 : préparation non lancée](#experience-2893566d-a69d-4a26-bd2b-54d26c66073d)
- [07. R0 : baseline d1-ultra effective](#experience-9d81e9ba-b528-4f56-89de-049034bb3e07)
- [08. Luna DEV25+D15 : extraction documentaire répétée](#experience-06273e68-8c84-4758-8aa9-e7a2e23fa143)
- [09. R1c : une requête par paire non ordonnée](#experience-f17a10ce-51ad-440d-bddf-67fa9abb8ff2)
- [10. R1a : aligner la question sur les relations ESC](#experience-2750ffff-41c6-427e-a1ca-b177049d76fc)
- [11. R1b : baliser A et B dans le texte](#experience-f0710474-0d2e-42fc-9a96-d7db645dc0ba)
- [12. R1e : démonstrations récupérées pour d1](#experience-7bd4db91-8a31-47ae-bc87-db4d99a57bff)
- [13. R1d : plusieurs questions dans le même appel](#experience-15d2db3f-307f-477f-aa76-2fe008b320c9)
- [14. R2a : symétrisation et seuil appris hors document](#experience-4f2e2f6e-82c1-4d88-88d3-ba1047814c5e)
- [15. R2b : coréférence prédite puis propagation](#experience-d02eadb6-4a61-4c0c-a9e2-f5ac246a4ac1)
- [16. Confirmation d1 : symétrie et seuil 0,65 gelés](#experience-38c20615-90ba-436e-bd97-03400935eb8e)
- [17. Confirmation Luna : contrôle documentaire sur CONF40](#experience-18d28e3b-98a1-43d7-8f82-b1386f7b4fbb)
- [18. R3-T0 : fusion Luna × d1](#experience-514110b5-14e8-45f7-8347-2feef53129fa)
- [19. R3-T1 : rationales dans les démonstrations](#experience-8d3a10af-5925-4454-b53a-c9aea79c0424)
- [20. R3-T2 : groupes de coréférence explicitement produits par Luna](#experience-8a84f788-7bed-4d77-a7aa-7e69ab431e7d)
- [21. R4-A : préciser la sémantique des événements](#experience-c5adf0b0-2e62-43fd-912d-9bbb9cd0d0a7)
- [22. R4-B : récupération structurelle et négatifs explicites](#experience-aac30ccd-4e1f-4020-9c28-34e93c7a96bb)
- [23. H2-D : adaptation du harness de d1 sans labels](#experience-c5ead119-a20f-4a0f-b7bc-33b4760c8aee)
- [24. H2-L : adaptation du harness de Luna](#experience-e8b797ec-3933-4f23-92b3-3faf9c3a5dcc)
- [25. H2-F : adaptation du harness de la fusion](#experience-26772f64-c118-453a-b5fc-1fbbf3d9c14d)
- [26. J2 : nouveau juge dans la bande incertaine](#experience-38e7efda-ffe4-4da8-b53e-f38cbe627144)
- [27. J1R : coréférence par accord Luna × DeepSeek](#experience-8cc07740-6959-4f55-b15a-021f515d0aae)
- [28. J3-A : confirmation de la moyenne calibrée J2](#experience-0be04739-534b-4a5f-a098-0573be60b6cc)
- [29. J3-B : calibration logistique régularisée](#experience-0a091e08-ee86-4b08-a41e-944b8e5ee36c)
- [30. J4-A : fusion logistique supervisée des deux scores initiaux](#experience-4e3309d0-10d4-4c59-b098-1b4cd01a6639)
- [31. J4-B : ajouter des caractéristiques structurelles à la calibration](#experience-adf2136a-1962-47be-9e5f-387b85c52102)
- [32. K0 : demander directement chaque relation à Luna](#experience-f644c423-1a72-44a5-8b37-52e136998f86)
- [33. KA : cascade de types puis élimination](#experience-ec936f5a-5e3b-4ea3-841f-ed31d44e1852)
- [34. KB : proposer les causes d'une mention cible](#experience-b68899f8-464f-4f60-8dad-e9e29fb3db53)
- [35. KC : décider entre événements puis recopier aux mentions](#experience-a5ca0832-24c6-494e-95d9-ab8a0bead025)
- [36. KD : combiner les signaux hétérogènes du round K](#experience-28784112-0cb3-4466-9a0b-56f5aeba0b57)
- [37. K1 : jugement Luna par paire sur DEV10](#experience-5a28e40b-a7d0-445b-9ea2-7427f7764149)
- [38. H1 : Haiku documentaire sur ESC et CTB complets](#experience-519ac1f5-34cf-40a7-9f6d-636adcfa070d)
- [39. Qwen documentaire initial : baseline locale sans d1](#experience-069fa421-bef4-497d-a48a-9510c476a316)
- [40. Alias virtuels A/R : test d'ancrage par identifiants auxiliaires](#experience-fe522ecd-a36a-46df-8ab9-f531b29c0fca)
- [41. M0 hors ligne : planifier la couverture des paires](#experience-927184b6-9039-4e7a-ba0f-2faa6fa56b39)
- [42. P1 v1.1 : extraire des preuves sources avant transport local](#experience-3b57057e-71b0-4900-b8aa-2efd65d4f204)
- [43. M0 live : O/P/M/MP sur deux documents](#experience-2a1c6883-9130-4746-8064-2c19e131653f)
- [44. Placebos V0 : faisabilité d'éditions contrôlées automatiques](#experience-4b3cd218-0007-4211-b414-4840b7175316)
- [45. CTB attribution : concurrence entre liaisons argumentales](#experience-6d8bda08-e1bf-49e3-9b4f-f4592a01cee9)
- [46. M1 brouillon : nœud préparatoire non implémenté](#experience-34b19e46-8794-48e4-b445-1f1402975a4e)
- [47. M1 : diversité complémentaire contre lectures ordinaires](#experience-e06a915f-f97c-4f97-8310-bc3338d9b0e4)
- [48. Qwen xhigh, température zéro : reprise de baseline](#experience-b04d73c3-1dd9-42a9-a27b-d321278bb250)
- [49. N1 : filtrer les ajouts par preuve et ancrage](#experience-3b10f583-c944-4dfa-92ba-1a5ff8873bbd)
- [50. N2 : transport local v2, même pool et vérificateur commun](#experience-338d526f-2cd1-4e12-9783-10db01df8e32)
- [51. N5 : ablation exhaustive 1/3/7 lectures sur caches](#experience-a44ecab3-6c74-4895-84e0-344510820848)
- [52. N3 : paquet de démonstrations contrastives](#experience-c007d4b4-6142-4b2d-a3a1-7da0e3c8716b)
- [53. N4 : remplacer Luna par DeepSeek documentaire](#experience-3c1651c8-a305-44a9-a1ac-32096dc3359a)
- [54. N6 : question lexicale adaptée à CTB](#experience-95286508-31f4-4318-9ea6-cb7fa3bb7392)
- [55. N3 audit : récupération et reproduction sans génération](#experience-126ff578-40cb-4041-8f37-6b6aad85cf20)
- [56. N7 : réplication du paquet N3 sur deux calendriers](#experience-2b118e89-feae-4028-8a5f-041b35fadc7a)
- [57. N8 : trois lectures fraîches réellement incluses dans sept](#experience-2483cd7a-30a8-4836-8f97-06386dd31d05)
- [58. N9 : factoriel récupération × négatifs](#experience-c2fc283c-e85d-40e6-b231-00aa5759a96a)
- [59. N10 : six lectures contre tous les triples de N7](#experience-bfe65f5c-1775-4082-b398-94ba4b520a71)
- [60. N11 : arrêt anticipé exact des sept lectures](#experience-4254c7f2-ca34-437f-8f28-d61a016ca5ec)
- [61. N12 : changer la vue des occurrences](#experience-bdbab303-7898-4fb0-a9a9-676bda82b084)
- [62. N13 : représentations avec Qwen local, versions ESC puis CTB50](#experience-0594a3dc-b551-4b9a-9005-82ed671be484)
- [63. Qwen CTB seul : terminer la baseline avant ESC](#experience-03e141a3-c891-4cbc-95f1-eba83ea3685a)
- [64. N14 : factoriel prompt × exécution bornée, Qwen medium sur CTB4](#experience-91c2b8fe-5c86-4644-b1d9-ff6354054392)
- [65. J1 brouillon : accord de deux procédures DeepSeek](#experience-42d6a6ad-3405-478a-9ce7-c892bae94e8e)

<a id="experience-694058c6-1973-46a5-8bc5-3579eada24ad"></a>

### 01. Span : validation initiale du modèle de décision

Expérience `694058c6-1973-46a5-8bc5-3579eada24ad` — famille de données : **ESC**.

**Méthode.** Construire un état contenant le document entier et les deux mentions exactes : identifiants, surfaces, positions de tokens, offsets et phrase. Envoyer une question Noul oui/non ; classer causal si la probabilité vaut au moins 0,5. Le gold relationnel n'entre pas dans cet état.

**Données et protocole.** Un seul document ESC dev, 41_9ecbplus.xml ; 156 candidats ordonnés et huit positifs. Une décision par candidat, sans vote à sept lectures. C'est un test d'intégration en conditions réelles, pas une validation statistique.

**Contrôles rapportés.** 156 appels réussis ; sauvegarde des probabilités et réponses ; rejeu hors ligne du score identique. Le modèle servi est identifié dans le résumé. Les erreurs de direction sont séparées de la mesure binaire.

**Bilan et limites.** F1 binaire 0 : aucune relation positive prédite, donc huit faux négatifs. Le chemin de données fonctionne ; le modèle n'a pas réussi la détection positive sur ce document.

Branche : `orx/esc-free-noul-span-smoke` ; tête au moment de l’inventaire : `93bb77ff1fd1c748808263487d6240aa8c32a696`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `aa75d46a-5794-4b93-ab23-126075a69584` | done | `93bb77f` | 8m03s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`

<a id="experience-87cddd6f-7910-4913-8e2f-4fe3441ea33f"></a>

### 02. Mercury : remplacement du modèle, protocole inchangé

Expérience `87cddd6f-7910-4913-8e2f-4fe3441ea33f` — famille de données : **ESC**.

**Méthode.** Remplacer Span par Mercury en gardant le document, les 156 candidats, la question Noul, la représentation et le seuil 0,5. Le changement testé est le modèle de décision.

**Données et protocole.** Comparaison sur le même document et les mêmes huit positifs que Span. Précision, rappel, F1 binaire et sous-périmètres intra/inter-phrases.

**Contrôles rapportés.** 156 décisions livrées ; version servie enregistrée ; prédictions et métriques sauvegardées pour le rejeu. Aucun seuil choisi après observation du score.

**Bilan et limites.** Précision 43,75 %, rappel 87,5 %, F1 58,33. Meilleur smoke local, sans preuve de généralisation à ESC233.

Branche : `orx/esc-free-noul-mercury-smoke` ; tête au moment de l’inventaire : `6e48b7e25fad0c3bfea0d47192a62ed96bf1b538`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `654d1602-1b6e-47f3-a88e-aa992adbe7b4` | done | `6e48b7e` | 7m59s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`

<a id="experience-2fbe084a-f105-4726-be67-815b8cf98c77"></a>

### 03. Mercury Choice : décision à trois classes

Expérience `2fbe084a-f105-4726-be67-815b8cf98c77` — famille de données : **ESC**.

**Méthode.** Remplacer la question Noul par un choix CAUSES, CAUSED_BY ou NO_RELATION. Le modèle peut donc exprimer la direction. Le contexte et les candidats restent ceux du smoke Mercury.

**Données et protocole.** Même document et seuil déclaré ; le F1 binaire regroupe les deux classes causales. Les métriques directionnelles sont conservées séparément.

**Contrôles rapportés.** 156 appels ; réponses/probabilités archivées ; rejeu exact. Vérification que le résultat comparé est bien le binaire, et non une autre colonne de métriques.

**Bilan et limites.** TP=8, FP=24, FN=0 : F1 40,00. Le rappel augmente, mais les faux positifs font baisser le F1 par rapport au Noul Mercury.

Branche : `orx/mercury-choice-vs-noul` ; tête au moment de l’inventaire : `f9d6dc3eea69aa48f0ebd9cf207e73a70012c26b`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `5d1c4b87-783d-4f15-a87d-5bce934f83f2` | done | `f9d6dc3` | 15m36s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`

<a id="experience-293c814f-d7a3-477f-80e2-dae63e9774b1"></a>

### 04. Mercury Context : rendre explicites les occurrences ciblées

Expérience `293c814f-d7a3-477f-80e2-dae63e9774b1` — famille de données : **ESC**.

**Méthode.** Conserver Noul et ajouter au document une copie où les tokens A/B sont balisés, puis les phrases contenant A et B. Les événements restent les mêmes ; l'intervention porte sur leur présentation.

**Données et protocole.** Même smoke à 156 candidats ; modèle et seuil 0,5 inchangés. Comparaison appariée à Mercury Noul.

**Contrôles rapportés.** 156 appels ; sauvegarde et rejeu exact des sorties. Le constructeur d'état vérifie les styles reconnus et conserve les positions des mentions.

**Bilan et limites.** TP=8, FP=18, FN=0 : F1 47,06. L'explicitation ne surpasse pas le Noul initial sur ce cas unique.

Branche : `orx/mercury-explicit-context-vs-noul` ; tête au moment de l’inventaire : `bc98c9f8da6357fbb350983ffd1da61a269f1798`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `e8a58cf1-30a2-4562-b1a2-1047d16b8c94` | done | `bc98c9f` | 15m40s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`

<a id="experience-3afb71c4-e110-4a31-a9eb-899fc1419b68"></a>

### 05. Mercury sur ESC233 : passage à l'échelle interrompu

Expérience `3afb71c4-e110-4a31-a9eb-899fc1419b68` — famille de données : **ESC**.

**Méthode.** Appliquer la même décision Noul à tous les candidats ordonnés d'ESC train : 233 documents, 122 130 paires. Version de données, sélection et empreinte de suite figées ; modèle gratuit uniquement, cadence 20 requêtes/minute et cache partagé persistant.

**Données et protocole.** Objectif initial : score corpus complet et comparaison historique. Après HTTP429 puis arrêt demandé à une frontière documentaire, seules les unités entièrement terminées sont évaluables. Le diagnostic initial de 13 documents et le paquet final D15 sont deux instantanés distincts.

**Contrôles rapportés.** Validation des 4 253 réponses en cache après le premier échec ; respect de Retry-After et cooldown partagé ; test de préflight sans débordement du contexte. Les documents partiels sont exclus des calculs partiels, sans transformer les manquants en négatifs. Les anciennes prédictions DeepSeek n'ont pas été retrouvées.

**Bilan et limites.** Aucun résultat ESC233 complet. F1 33,00 sur l'instantané 13 documents ; 32,39 sur D15 final, diagnostique seulement. Ni l'un ni l'autre n'est comparable directement au chiffre historique sur 233 documents.

Branche : `orx/mercury-noul-on-historical-esc-train233` ; tête au moment de l’inventaire : `79169a769738b808ac83aa577119474c934798f9`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `586bbafb-473f-4e7d-94b2-e3eb1b381137` | cancelled | `79169a7` | 3h20m | disponible |
| `dfa23986-77a1-4a69-b0ea-b3701fc586bd` | failed | `960aa8a` | 3h54m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/historical-comparison/partial-evaluation-20261016T075639Z/partial-results-report.md`

<a id="experience-2893566d-a69d-4a26-bd2b-54d26c66073d"></a>

### 06. Prototype d1 DEV25+D15 : préparation non lancée

Expérience `2893566d-a69d-4a26-bd2b-54d26c66073d` — famille de données : **ESC**.

**Méthode.** Préparer le remplacement de Mercury par liquid/d1-ultra, avec appels parallèles par paire et seuil Noul 0,5. La cible est DEV25 et le diagnostic D15.

**Données et protocole.** Aucune exécution attachée à ce nœud. La réalisation effective se trouve dans R0, fiche suivante. Les deux nœuds ne sont pas deux comparaisons indépendantes.

**Contrôles rapportés.** État et notes du nœud relus ; absence de run dans l'inventaire. Une branche préparée ne démontre ni qualité mesurée ni réussite du déploiement.

**Bilan et limites.** Préparation remplacée par R0 ; aucun score propre.

Branche : `orx/d1-ultra-noul-baseline-dev25-d15` ; tête au moment de l’inventaire : `4730ae002c705ef6a5bfb634cc57096d380c625b`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

**Aucun run attaché** : préparation ou brouillon, pas de résultat d’exécution propre.

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`

<a id="experience-9d81e9ba-b528-4f56-89de-049034bb3e07"></a>

### 07. R0 : baseline d1-ultra effective

Expérience `9d81e9ba-b528-4f56-89de-049034bb3e07` — famille de données : **ESC**.

**Méthode.** Une probabilité causale par paire ordonnée, document entier et mentions exactes ; seuil 0,5. Modèle servi liquid/d1-ultra-20261015. Ajout de parallélisme, cache et registre atomique de budget ; ces mécanismes d'exécution ne constituent pas des variantes scientifiques.

**Données et protocole.** DEV25 : 19 028 candidats et 1 086 positifs. Sous-échantillon DEV-SCREEN : dix documents, 6 894 candidats, 436 positifs. D15 sert au diagnostic et pas à la sélection. AP pour le classement, micro-F1 binaire pour les décisions.

**Contrôles rapportés.** Correction des verrous SQLite par retry de transaction locale, sans rappel modèle dans cette boucle ; 17 tests rapportés dans la reprise. Données et question maintenues, sorties persistées, comparaison des variantes sur documents appariés.

**Bilan et limites.** DEV25 : F1 39,26, AP 39,87 ; DEV-SCREEN : F1 42,80. C'est le contrôle de R1 et R2. Les tentatives antérieures incomplètes ne sont pas des F1 supplémentaires.

Branche : `orx/r0-d1-ultra-noul-dev25-d15-reprise` ; tête au moment de l’inventaire : `08d8c6bbbb885f73cedb9035cd66ecb937fdc0b4`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `d0e4ab4a-1c61-4c15-871e-fec594d6bcdd` | done | `08d8c6b` | 9h07m | disponible |
| `cc6b7a91-6c5d-4030-a402-21272693157f` | failed | `eed0f32` | 35m56s | disponible |
| `30be8ec8-30b9-4c38-885f-f67e33e88f18` | cancelled | `6c75f7a` | 19m38s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/r0-baseline/report.md`

<a id="experience-06273e68-8c84-4758-8aa9-e7a2e23fa143"></a>

### 08. Luna DEV25+D15 : extraction documentaire répétée

Expérience `06273e68-8c84-4758-8aa9-e7a2e23fa143` — famille de données : **ESC**.

**Méthode.** Faire lire chaque document balisé à Luna batch en raisonnement high. Sept générations indépendantes, chacune avec un exemple gold d'un autre topic. La sortie est une liste de paires/labels ; les réponses sont agrégées par le vote historique. Ce système diffère de d1 par le modèle, la granularité et les démonstrations.

**Données et protocole.** 25 documents DEV et 15 D15, soit 280 requêtes documentaires. Contrôle principal DEV25 apparié à R0 ; D15 reste diagnostic. Le vote historique par labels et le F1 binaire ne doivent pas être confondus avec une probabilité calibrée.

**Contrôles rapportés.** Correction du transport vers une véritable API batch après HTTP404 ; reçus durables, reprise sans double soumission, cache distinct par document. 40/40 sous-évaluations terminées ; reconstruction hors ligne et contrôle des candidats.

**Bilan et limites.** DEV25 : F1 50,85, écart à d1 +11,59 points [7,51 ; 15,65]. AP 33,49, inférieure à celle de d1. Le gain n'isole pas l'effet propre de sept lectures ou de l'effort high.

Branche : `orx/llm-luna-batch-high-dev25-d15` ; tête au moment de l’inventaire : `0cfb03f72e2a9c077829cd6937c0e9d82d1155e3`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `7c9548f7-a36a-41f8-8a1b-9d31748b2810` | done | `0cfb03f` | 1h27m | disponible |
| `fa35ad3a-6863-435e-9bcc-69a602a5584f` | failed | `a170cb7` | 2m19s | disponible |
| `f426ae14-39da-4507-a0ef-b75ad1c2b280` | cancelled | `6c75f7a` | 19s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/luna-dev25/report.md`

<a id="experience-f17a10ce-51ad-440d-bddf-67fa9abb8ff2"></a>

### 09. R1c : une requête par paire non ordonnée

Expérience `f17a10ce-51ad-440d-bddf-67fa9abb8ff2` — famille de données : **ESC**.

**Méthode.** Choisir l'ordre canonique des deux mentions, demander une seule probabilité et la recopier aux deux candidats historiques orientés. Cela divise exactement par deux le nombre logique d'appels d1.

**Données et protocole.** DEV-SCREEN inchangé ; seuil 0,5 ; comparaison appariée R0. L'objectif d'efficacité inclut une limite de dégradation du F1, et le bilan de sélection R1 examine aussi le classement AP.

**Contrôles rapportés.** Contrôle du remappage vers les deux orientations et de l'identité des paires ; cache des reprises ; bootstrap par document, pas par orientation. Les intervalles couvrent des pertes dépassant un point : la proximité ponctuelle ne démontre pas la préservation de qualité.

**Bilan et limites.** F1 41,93 contre 42,80 pour R0 : −0,88 point ponctuel ; AP en baisse. Aucun finaliste retenu dans la campagne R1.

Branche : `orx/r1c-d1-ultra-unordered-screen` ; tête au moment de l’inventaire : `236694d268f566116cc769493eac30243ad6ae51`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `5bf3ecaa-c821-43e5-86b9-c9b735da0c0e` | done | `236694d` | 2h37m | disponible |
| `43ba76a3-cae0-49dc-b28d-60b5dbd070d6` | failed | `5e44074` | 1h33m | disponible |
| `ef9e55e1-34b6-4348-aa77-eb7b8d7210cb` | cancelled | `f2fea86` | 11m25s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/r1c-screen/report.md`

<a id="experience-2750ffff-41c6-427e-a1ca-b177049d76fc"></a>

### 10. R1a : aligner la question sur les relations ESC

Expérience `2750ffff-41c6-427e-a1ca-b177049d76fc` — famille de données : **ESC**.

**Méthode.** Élargir la définition de causalité dans la question : PLOT_LINK, facilitation, prévention, conséquence narrative soutenue et lien de sous-événement explicite. Conserver modèle, contexte et seuil de R0.

**Données et protocole.** Dix documents DEV-SCREEN ; une question par paire ordonnée. Sélection sur développement, avec AP comme critère principal de promotion, puis F1 et bootstrap apparié.

**Contrôles rapportés.** Reprise des mêmes requêtes depuis le cache après réparation budgétaire ; probabilités archivées. Les ensembles D15 et CONF40 ne servent pas au choix de cette variante.

**Bilan et limites.** F1 43,30, soit environ +0,50 point ; gain AP incertain. Aucun critère de promotion robuste franchi.

Branche : `orx/r1a-d1-ultra-esc-question-screen` ; tête au moment de l’inventaire : `0ae5f91cc2d0a5d81d3948b53abf7588c254eaf5`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `6841832e-ac6f-41fb-8215-730641ce52de` | done | `0ae5f91` | 6h59m | disponible |
| `322bd86a-0212-4fa8-9bed-f177535d4a5a` | failed | `b30de7e` | 33m29s | disponible |
| `fb899aae-c6c9-494a-8a19-01e61209938d` | cancelled | `db54913` | 10m58s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/r1a-screen/report.md`

<a id="experience-f0710474-0d2e-42fc-9a96-d7db645dc0ba"></a>

### 11. R1b : baliser A et B dans le texte

Expérience `f0710474-0d2e-42fc-9a96-d7db645dc0ba` — famille de données : **ESC**.

**Méthode.** Ajouter des balises autour des tokens exacts des deux occurrences visées dans le document de d1 ; préserver la question Noul et les candidats. L'hypothèse est un meilleur rattachement de la question aux mentions.

**Données et protocole.** DEV-SCREEN commun, seuil 0,5, métriques AP et F1 ; comparaison au R0 du même périmètre.

**Contrôles rapportés.** Vérification des positions et de la conservation du texte/identité des candidats, sauvegarde des scores et bootstrap apparié. L'amélioration d'un score de décision ne remplace pas le critère de classement annoncé.

**Bilan et limites.** F1 44,11, environ +1,31 point ; AP en baisse. L'intervalle du gain F1 inclut zéro : variante non promue.

Branche : `orx/r1b-d1-ultra-tagged-mentions-screen` ; tête au moment de l’inventaire : `ec7de232d726156ed12b63f9c7f6512b129712f1`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `6a71eda4-54ca-424f-9510-ba671d9fe740` | done | `ec7de23` | 6h34m | disponible |
| `bf51553c-23b4-4a3e-b120-37b63c1b15fc` | failed | `7cf978e` | 1h34m | disponible |
| `10bddad2-a06b-4019-83bb-85e222879c69` | cancelled | `0fbe9df` | 10m31s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/r1b-screen/report.md`

<a id="experience-7bd4db91-8a31-47ae-bc87-db4d99a57bff"></a>

### 12. R1e : démonstrations récupérées pour d1

Expérience `7bd4db91-8a31-47ae-bc87-db4d99a57bff` — famille de données : **ESC**.

**Méthode.** Ajouter quatre exemples récupérés par similarité : deux positifs et deux négatifs. Exclure le topic interrogé et les ensembles protégés de la banque de démonstrations. Les labels de ces exemples sont des annotations d'apprentissage autorisées, pas ceux de la cible.

**Données et protocole.** Même DEV-SCREEN, même d1 et seuil 0,5. Comparaison AP/F1 au contrôle ; aucune sélection sur D15.

**Contrôles rapportés.** Vérification de l'éligibilité de la banque et des exclusions ; cache/reprises conservés ; bootstrap apparié. Le coût du run terminal exclut les réponses déjà payées et relues dans le cache.

**Bilan et limites.** F1 43,26, environ +0,46 point ; gains de classement et F1 incertains. Aucun élargissement DEV-FULL de ce facteur.

Branche : `orx/r1e-d1-ultra-retrieved-demonstrations-screen` ; tête au moment de l’inventaire : `08922d583edd9d1edb4dc6c95505b7db46d318a6`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `41307f1d-6722-48be-903e-4aad11fa6ba2` | done | `08922d5` | 6h53m | disponible |
| `10a0a88a-e72c-4ab0-a7e2-a144ff81a78a` | failed | `db32720` | 1h19m | disponible |
| `2a62be6b-c7da-407a-96d4-782bd0bc8217` | cancelled | `1ed0cf3` | 8m40s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/r1e-screen/report.md`

<a id="experience-15d2db3f-307f-477f-aa76-2fe008b320c9"></a>

### 13. R1d : plusieurs questions dans le même appel

Expérience `15d2db3f-307f-477f-aa76-2fe008b320c9` — famille de données : **ESC**.

**Méthode.** Demander plusieurs décisions distinctes dans un appel : causalité, identité événementielle et autres propriétés de la paire. Prévoir un contrôle d'invariance de la probabilité causale avant le tri à grande échelle.

**Données et protocole.** Prétest sur les 156 paires du document connu : question causale seule versus appel multi-questions. L'évaluation DEV-SCREEN est conditionnelle à la réussite de ce prétest.

**Contrôles rapportés.** Les sorties des cinq questions sont distinctes, mais la probabilité causale change jusqu'à 0,21405. Le contrôle d'invariance échoue ; aucun score DEV-SCREEN ni calibrateur dérivé n'est produit. Une répétition dédiée serait nécessaire pour séparer interaction et stochasticité.

**Bilan et limites.** Prérequis non satisfait, essai arrêté. C'est un résultat du contrôle préalable, pas un F1 nul sur DEV ni la preuve que toutes les formulations multi-questions échouent.

Branche : `orx/r1d-d1-ultra-decomposed-questions-screen` ; tête au moment de l’inventaire : `5156f74ab05e7b536118d001a9b811423ed5348a`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `df51a1e6-f0a5-4e06-8685-a90d8685539f` | failed | `5156f74` | 1m32s | disponible |
| `73ae2296-7d4b-412f-8da4-247a1542752e` | cancelled | `f832a23` | 6m49s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`

<a id="experience-4f2e2f6e-82c1-4d88-88d3-ba1047814c5e"></a>

### 14. R2a : symétrisation et seuil appris hors document

Expérience `4f2e2f6e-82c1-4d88-88d3-ba1047814c5e` — famille de données : **ESC**.

**Méthode.** Réutiliser les probabilités R0. Pour chaque paire, moyenner d(a,b) et d(b,a). Comparer le seuil fixe 0,5 à un seuil choisi en leave-one-document-out : aucun label du document évalué pour choisir son seuil. Ajuster ensuite 0,65 sur DEV complet pour la future confirmation.

**Données et protocole.** DEV25 ; mêmes sorties brutes pour tous les post-traitements, donc zéro nouvel appel. Le réglage est interne au DEV ; les documents partagent néanmoins des topics.

**Contrôles rapportés.** Reconstruction déterministe des décisions, contrôle des folds et des seuils. Les IC complémentaires sont post hoc et ne réajustent pas le seuil dans chaque bootstrap. Correction d'un export YAML/date, sans nouvelle inférence.

**Bilan et limites.** F1 : R0 39,26 ; symétrie seule 39,87 ; symétrie+seuil hors document 42,04. Le +2,78 est un gain DEV incertain ; le transfert du seuil est testé dans la fiche 16.

Branche : `orx/r2a-d1-ultra-symmetry-lodo-dev25` ; tête au moment de l’inventaire : `f358a25f2578e5b6506e3aada8a0729c3ea0543d`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `0c4f8f08-d6d6-4cfa-9bc6-015b307ae418` | done | `f358a25` | 10s | disponible |
| `3ca1008a-072c-4df5-8195-55fa65887eec` | failed | `e2b0a07` | 10s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/r2a-symmetry/report.md`

<a id="experience-d02eadb6-4a61-4c0c-a9e2-f5ac246a4ac1"></a>

### 15. R2b : coréférence prédite puis propagation

Expérience `d02eadb6-4a61-4c0c-a9e2-f5ac246a4ac1` — famille de données : **ESC**.

**Méthode.** Poser séparément 9 514 questions d1 « même événement ? » sur les paires non ordonnées. Au seuil 0,5, former les composantes connexes ; agréger au maximum les scores causaux entre chaînes, puis diffuser. Les relations internes à une chaîne sont rejetées. Ajouter un oracle ECB+ pour mesurer le potentiel d'une identité correcte.

**Données et protocole.** DEV25. Deux évaluations différentes : précision/rappel des liens de coréférence contre ECB+, puis F1 causal sur toutes les paires. Seuil causal fixe ou choisi hors document comme R2a.

**Contrôles rapportés.** Questions de coréférence indépendantes de la question causale ; contrôle de fermeture et de propagation ; diagnostic avant/après fermeture. L'oracle utilise le gold de coréférence et n'est pas déployable.

**Bilan et limites.** Précision des liens : 26,00 % directs, 17,98 % après fermeture. F1 causal 29,27 au seuil 0,5 ; 37,70 avec seuil hors document. Oracle+seuil : 60,88. La sur-fusion prédite empêche de réaliser le potentiel oracle.

Branche : `orx/r2b-d1-ultra-predicted-coreference-dev25` ; tête au moment de l’inventaire : `3ba6739bf7b72133ae71c0526636775d6014ac76`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `26ac5081-c538-431e-994c-b6a7097f7245` | done | `3ba6739` | 2h01m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/r2b-predicted-coreference/report.md`

<a id="experience-38c20615-90ba-436e-bd97-03400935eb8e"></a>

### 16. Confirmation d1 : symétrie et seuil 0,65 gelés

Expérience `38c20615-90ba-436e-bd97-03400935eb8e` — famille de données : **ESC**.

**Méthode.** Geler la règle choisie sur DEV puis collecter une fois les scores d1 du nouveau CONF40. Appliquer aux mêmes scores la baseline 0,5 et la variante symétrique 0,65, sans régler à nouveau le seuil.

**Données et protocole.** 40 documents, 20 topics, 19 174 candidats ordonnés. Comparaison appariée avec bootstrap documentaire ; ce lot était réservé à cette étape de l'histoire du projet.

**Contrôles rapportés.** Empreinte du gel, complétude des candidats et reproduction officielle hors ligne. Les deux systèmes réutilisent les mêmes probabilités : la différence vient du post-traitement, pas d'une nouvelle génération.

**Bilan et limites.** Baseline F1 34,79 ; variante 28,29 ; −6,50 points [−9,94 ; −2,95]. Le réglage DEV ne se transfère pas. La variante retire 1 010 FP mais aussi 309 TP ; elle n'est pas promue.

Branche : `orx/final-d1-ultra-frozen-symmetry-conf40` ; tête au moment de l’inventaire : `81a17b23726ee6754bd9602786dcb8787087d541`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `945b655d-4595-4726-8d72-d1714bb2f5e6` | done | `81a17b2` | 4h10m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/final-report.md`

<a id="experience-18d28e3b-98a1-43d7-8f82-b1386f7b4fbb"></a>

### 17. Confirmation Luna : contrôle documentaire sur CONF40

Expérience `18d28e3b-98a1-43d7-8f82-b1386f7b4fbb` — famille de données : **ESC**.

**Méthode.** Conserver Luna high, sept lectures et une démonstration hors topic par lecture. Aucune adaptation sur les labels CONF40. Comparer au d1 original et au post-traitement d1 gelé.

**Données et protocole.** Les mêmes 40 documents que la confirmation d1 ; 280 réponses Luna. Micro-F1 global, intra/inter et différences appariées.

**Contrôles rapportés.** 40 lots complets ; reconstruction des votes et du score officiel. Les contrastes Luna sont secondaires, sans correction de multiplicité dans ce premier bilan.

**Bilan et limites.** F1 36,45 ; +1,65 point [−3,35 ; 6,33] face au d1 original. Le gain net observé sur DEV n'est pas confirmé avec la même ampleur. L'IC contenant zéro ne démontre pas l'équivalence.

Branche : `orx/final-luna-batch-high-conf40-reference` ; tête au moment de l’inventaire : `3470a896384c40faba0aa6f0ad92c692963cb547`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `ab42114f-3687-4493-b89a-eb619790fa7b` | done | `3470a89` | 40m09s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/bilan-complet-experiences/rapport-complet.md`
- `jev-v1/esc-improvement-plan/final-luna/report.md`

<a id="experience-514110b5-14e8-45f7-8347-2feef53129fa"></a>

### 18. R3-T0 : fusion Luna × d1

Expérience `514110b5-14e8-45f7-8347-2feef53129fa` — famille de données : **ESC**.

**Méthode.** Moyenner les deux orientations du vote Luna et les deux scores d1 : f=(v_ab+v_ba+d_ab+d_ba)/4. Positif si f≥0,39. La famille de règles et le seuil sont choisis sur POOL80, réunion de DEV25+D15+CONF40 (22 topics), en laissant un topic à l'écart, puis figés avant TEST-S40. D15 et CONF40 sont donc réutilisés à ce stade, après leur rôle réservé dans le plan précédent.

**Données et protocole.** 40 documents TEST-S40, 17 958 candidats ordonnés ; deux documents par topic, exclusions de D15, CONF40 et banque de démonstrations. Comparer fusion, Luna seule et d1 seul sur les mêmes documents.

**Contrôles rapportés.** Recalcul hors ligne depuis sorties et contrôles ; règle de fusion sans refit sur le test. Bootstrap apparié par document ; lecture des résultats de tous les documents, sans remplacer le micro-F1 par une moyenne documentaire.

**Bilan et limites.** Fusion 48,64 ; d1 38,59 ; Luna 46,02. Écart +10,05 [6,66 ; 13,65] face à d1, mais +2,62 [−1,37 ; 7,69] face à Luna : la supériorité sur les deux contrôles n'est pas établie.

Branche : `orx/r3-t0-test-s40-controls-frozen-luna-x-d1-fusion` ; tête au moment de l’inventaire : `d39d3349fe90c2396307d189b32d46ee7f6dabe5`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `64e91e64-d6e8-4a51-94fb-eab40c38eec7` | done | `d39d334` | 59m03s | disponible |
| `1906c172-b273-4b3e-94c8-1b7500d8782d` | cancelled | `dccc8fe` | 16m51s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/round3/round3-results-report.md`

<a id="experience-8d3a10af-5925-4454-b53a-c9aea79c0424"></a>

### 19. R3-T1 : rationales dans les démonstrations

Expérience `8d3a10af-5925-4454-b53a-c9aea79c0424` — famille de données : **ESC**.

**Méthode.** Garder les mêmes documents de démonstration que T0, mais ajouter à leur réponse gold une explication produite une fois par Luna. La cible ne reçoit pas son gold ; l'intervention est le contenu des démonstrations.

**Données et protocole.** TEST-S40 apparié à Luna T0 ; sept lectures high et contrôle gold-answer-only. Comparaison de la majorité binaire et du vote historique par labels.

**Contrôles rapportés.** 40/40 documents complets ; rapprochement des calendriers de démonstrations. Les deux règles de vote diffèrent légèrement et sont rapportées distinctement.

**Bilan et limites.** Majorité binaire : F1 44,34 contre 46,02, soit −1,67 [−3,85 ; 0,85]. Vote historique par labels : 44,27. Aucun gain confirmé des rationales.

Branche : `orx/r3-t1-luna-rationale-demonstrations-test-s40` ; tête au moment de l’inventaire : `f9767cac359edc12cd3c839278929294f2fc39a1`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `fcef631c-414b-4c2e-9edf-31779de3f6d7` | done | `4c516e1` | 41m10s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/round3/round3-results-report.md`

<a id="experience-8a84f788-7bed-4d77-a7aa-7e69ab431e7d"></a>

### 20. R3-T2 : groupes de coréférence explicitement produits par Luna

Expérience `8a84f788-7bed-4d77-a7aa-7e69ab431e7d` — famille de données : **ESC**.

**Méthode.** Demander une seule fois par document les groupes de mentions du même événement, sans causalité. Propager ensuite au maximum les scores causaux entre groupes, sans suppression interne et sans fermeture additionnelle au-delà des groupes fournis.

**Données et protocole.** Gate intrinsèque fixé à 60 % de précision des liens de coréférence. Le run vise 120 documents ; 116 sont complets, dont tous ceux de TEST-S40, D15 et CONF40. DEV25 reste incomplet.

**Contrôles rapportés.** Séparer échec global du processus et complétude des sous-ensembles. Comparer les liens aux chaînes ECB+ ; les scores causaux après échec du gate sont exploratoires, pas une promotion.

**Bilan et limites.** Précision TEST-S40 54,86 %, sous le gate. Fusion+groupes F1 48,88, +0,24 [−1,25 ; 1,58] en diagnostic. Aucun avantage validé ; pas de métrique DEV25 complète inventée.

Branche : `orx/r3-t2-luna-explicit-event-coreference-chains` ; tête au moment de l’inventaire : `4cb91fe84d5275ce7d07828e887a10a13883dc13`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `7a6bd979-4a1a-4d07-9c28-ae0d488a92b4` | failed | `968ae69` | 44m32s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/round3/round3-results-report.md`

<a id="experience-c5adf0b0-2e62-43fd-912d-9bbb9cd0d0a7"></a>

### 21. R4-A : préciser la sémantique des événements

Expérience `c5adf0b0-2e62-43fd-912d-9bbb9cd0d0a7` — famille de données : **ESC**.

**Méthode.** Modifier seulement les consignes Luna : identité stricte, distinction phases/sous-événements, motivation, réaction, facilitation, prévention et cohérence des participants. Maintenir démonstrations, sept lectures et fusion 0,39.

**Données et protocole.** DEV44, 44 documents de 22 topics, 21 832 candidats orientés ; contrôle de fusion 47,323. Développement déjà consommé, pas confirmation.

**Contrôles rapportés.** Sélection de documents sans labels, d1 en cache ; résultat ROUND4_RESULT. Bootstrap documentaire et par topic ; IC 97,5 % également conservés pour les deux comparaisons R4 principales.

**Bilan et limites.** F1 44,318, −3,006 points. L'IC documentaire à 95 % [−7,157 ; 0,298] inclut zéro ; aucune amélioration à retenir.

Branche : `orx/r4-a-luna-event-semantics-dev44` ; tête au moment de l’inventaire : `e1c9baff072de16d8fb30ab261636070beb2f1b0`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `ca86b400-acac-407c-b440-4bdda95463f5` | done | `e1c9baf` | 1h01m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `synthese-globale-2026-10-29/recapitulatif-actualise-2026-10-30.md`

<a id="experience-aac30ccd-4e1f-4020-9c28-34e93c7a96bb"></a>

### 22. R4-B : récupération structurelle et négatifs explicites

Expérience `aac30ccd-4e1f-4020-9c28-34e93c7a96bb` — famille de données : **ESC**.

**Méthode.** Pour chaque cible, calculer sept caractéristiques publiques : nombre de mentions normalisé, fraction de paires intra-phrase, densité par phrase, répétition des surfaces et densités de mots causaux/temporels/modaux. Classer les démonstrations par cosinus, hors topic. Ajouter jusqu'à quatre négatifs gold des exemples, privilégiant même occurrence, voisin causal commun et même phrase.

**Données et protocole.** DEV44 identique à R4-A ; sept lectures, banque de 20 démonstrations et fusion 0,39. L'intervention combine sélection des exemples et négatifs : leurs effets ne sont pas séparés ici.

**Contrôles rapportés.** Code de profile, negative_pairs et select_demonstrations relu ; exclusions de topic, départage déterministe par hash, candidats et contrôles vérifiés. Bootstrap apparié et correction à deux contrastes disponible.

**Bilan et limites.** F1 48,429 contre 47,323 ; +1,106 point, IC95 documentaire [−0,815 ; 3,656]. Signal de développement incertain ; il motive N3/N7/N9 sans être une victoire confirmée.

Branche : `orx/r4-b-luna-structural-contrastive-demos-dev44` ; tête au moment de l’inventaire : `93e284dc815f6d396bd11aa06c5216f65ab5da4c`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `6ce6a18b-692d-4cc3-86f5-b06701dd1596` | done | `93e284d` | 1h05m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `synthese-globale-2026-10-29/recapitulatif-actualise-2026-10-30.md`

<a id="experience-c5ead119-a20f-4a0f-b7bc-33b4760c8aee"></a>

### 23. H2-D : adaptation du harness de d1 sans labels

Expérience `c5ead119-a20f-4a0f-b7bc-33b4760c8aee` — famille de données : **ESC**.

**Méthode.** Une boucle Propose–Probe–Compose édite instructions et script de post-traitement de l'exécuteur d1. Éditeur/compositeur Luna ; bibliothèque d'expérience par topic ; k=1. Les annotations n'interviennent qu'après adaptation pour mesurer le F1.

**Données et protocole.** Huit documents DEV44 de quatre topics prévus ; sept disposent de la sortie finale. Comparaison du harness sélectionné au harness de base, bootstrap documentaire sur ces sept documents seulement.

**Contrôles rapportés.** Les sorties de base correspondent exactement au cache R3. Traçage des étapes, éditions et fallbacks ; un document manque au stade final. Les échecs 401 et de transport antérieurs ne sont pas des résultats scientifiques.

**Bilan et limites.** Sur sept documents : F1 39,46→45,87, +6,41 [1,44 ; 14,30]. L'éditeur a surtout retrouvé la symétrisation OR déjà connue. Résultat partiel et petit échantillon, pas nouvelle amélioration confirmée de la fusion.

Branche : `orx/h2-d-harness-test-time-harnessing-d1-ultra-decis` ; tête au moment de l’inventaire : `9ebcf9720d0eb57db7f29c8a5265f34aef053798`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `705c9838-0dcf-4875-bc97-dbbe3025937d` | failed | `9ebcf97` | 2h19m | disponible |
| `63f70f4f-5ab9-4822-858f-2c610f92ed69` | cancelled | `8377da9` | 6m10s | disponible |
| `f159c4f9-58f7-4fb4-8898-072ec5eb663f` | cancelled | `7947a31` | 1m14s | disponible |
| `9c35861e-10df-4bc9-8a72-fd993498e768` | failed | `7947a31` | 2m25s | non localisé |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/harness2-pilots/harness2-pilots-report.md`

<a id="experience-e8b797ec-3933-4f23-92b3-3faf9c3a5dcc"></a>

### 24. H2-L : adaptation du harness de Luna

Expérience `e8b797ec-3933-4f23-92b3-3faf9c3a5dcc` — famille de données : **ESC**.

**Méthode.** Même mécanisme, avec Luna documentaire sept lectures comme exécuteur. Les leviers sont le prompt, les règles, les skills/mémoires et un post-traitement des votes. Sélection sans accès aux labels de la cible.

**Données et protocole.** Huit documents communs au pilote H2-D ; comparer base et final sélectionné sur les huit complets.

**Contrôles rapportés.** Base identique au cache R3 ; journaux des étapes INV/DOM/AGG et décisions KEEP_BASE. L'adaptation et l'évaluation sont deux étapes distinctes ; aucune sélection du meilleur harness selon le F1 cible.

**Bilan et limites.** F1 43,40→43,83, +0,43 [−5,79 ; 10,10]. Les modifications de domaine augmentent parfois la prudence et réduisent le rappel. Gain non établi.

Branche : `orx/h2-l-harness-test-time-harnessing-luna-llm-execu` ; tête au moment de l’inventaire : `ec2779f76b308e3f0414264d5ce6f6a94eb738e2`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `6de8dcd9-212c-47e3-a408-a296bcdbe675` | done | `ec2779f` | 2h13m | disponible |
| `36635dc6-82a4-487c-8e3d-d4a741e573cc` | failed | `5031786` | 30s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/harness2-pilots/harness2-pilots-report.md`

<a id="experience-26772f64-c118-453a-b5fc-1fbbf3d9c14d"></a>

### 25. H2-F : adaptation du harness de la fusion

Expérience `26772f64-c118-453a-b5fc-1fbbf3d9c14d` — famille de données : **ESC**.

**Méthode.** Donner à l'éditeur la fusion complète et son script de score, avec récursion k=2 et sélection sans labels entre deux finales. Router éditeurs, compositeur, sélecteur et réflecteur vers des modèles distincts ; conserver les fallbacks vers la base en cas d'étape invalide.

**Données et protocole.** Huit documents du même pilote ; base fusion 45,56. Rapporter aussi le sous-ensemble de sept documents H2-D sans le substituer à l'échantillon principal.

**Contrôles rapportés.** Base reproduite exactement depuis les caches ; validation des blocs de réponse des agents. Trois documents perdent la sélection à cause des limites de clé/budget ; fallbacks et taux de sessions acceptées sont conservés.

**Bilan et limites.** F1 final 47,02, +1,46 [−4,23 ; 8,12]. Pilote techniquement incomplet dans ses sélections, sans gain robuste sur la fusion.

Branche : `orx/h2-f-harness-on-frozen-luna-x-d1-fusion-k-2-rout` ; tête au moment de l’inventaire : `3728959d99169a57d841f34d7a0e3370d51efa87`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `718cb169-5d34-4619-bdc4-0ac967936084` | done | `3728959` | 9h39m | disponible |
| `30ddb304-00dc-4709-b76b-bf1a310e24a3` | cancelled | `9823c4e` | 27m11s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/harness2-pilots/harness2-pilots-report.md`

<a id="experience-38e7efda-ffe4-4da8-b53e-f38cbe627144"></a>

### 26. J2 : nouveau juge dans la bande incertaine

Expérience `38e7efda-ffe4-4da8-b53e-f38cbe627144` — famille de données : **ESC**.

**Méthode.** Rejuger uniquement les paires de score fusionné dans [0,25 ; 0,55). DeepSeek v4.1 flash, effort max, deux échantillons mélangés ; 1 089 paires non ordonnées, découpées en blocs de 64 maximum. Combiner score initial et score du juge par poids de {0 ; 0,25 ; 0,5 ; 0,75 ; 1}.

**Données et protocole.** DEV44, 44 documents/22 topics. Primaire : poids choisi hors topic, seuil maintenu à 0,39. Secondaire : ajustement conjoint poids/seuil hors topic. Le reste des décisions reste inchangé.

**Contrôles rapportés.** Contrôle exhaustif des paires attendues : omission ou duplication fait échouer le jugement, sans défaut négatif. Première tentative interrompue sur une paire omise, puis réparation du transport/cache. Tests ciblés et scores sauvegardés ; contrôle de l'effort demandé et des tokens de raisonnement retournés.

**Bilan et limites.** Primaire F1 45,74 contre 47,32, soit −1,58 [−3,06 ; −0,36]. Secondaire réglé 49,31, exploratoire ; cette piste secondaire est ensuite évaluée dans J3 et n'est plus en attente de confirmation.

Branche : `orx/j2-deepseek-verifier-on-the-uncertain-fusion-ban` ; tête au moment de l’inventaire : `ccdaf3a5c8a4a0aeed21d301437f95bc49cf8cca`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `582ddbdd-9f2b-4878-bd7d-650e2874869b` | done | `ccdaf3a` | 3h32m | disponible |
| `42c1426c-7292-41a2-bfb4-01524a3b76e7` | failed | `a76f4b5` | 1h47m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/new-signal-judges/j1-j2-results-report.md`

<a id="experience-8cc07740-6959-4f55-b15a-021f515d0aae"></a>

### 27. J1R : coréférence par accord Luna × DeepSeek

Expérience `8cc07740-6959-4f55-b15a-021f515d0aae` — famille de données : **ESC**.

**Méthode.** Prendre les liens de groupes Luna déjà calculés et demander à DeepSeek max de confirmer l'identité événementielle. Fermer les liens conservés en composantes ; exiger 85 % de précision intrinsèque avant toute propagation causale.

**Données et protocole.** DEV44 ; 442 liens Luna, 315 liens gold, 43 chunks DeepSeek. La question est d'abord la fiabilité des groupes, pas le F1 causal.

**Contrôles rapportés.** Récupération d'un reçu batch Luna déjà payé, sans nouvelle génération ; validation des chunks et de l'identité servie. Le gate empêche explicitement l'évaluation causale quand la précision est insuffisante.

**Bilan et limites.** Liens d'accord après fermeture : 193/362 corrects, précision 53,31 %. Gate 85 % échoué. Aucun F1 causal calculé : l'échec porte sur le préalable intrinsèque.

Branche : `orx/j1r-luna-x-deepseek-max-high-precision-coreferen` ; tête au moment de l’inventaire : `ca89eb086319761575ef3b65c039be0f6d1ef51f`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `87a81f5a-c099-4c0c-a201-35d99c7d6a9e` | done | `ca89eb0` | 1h14m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/new-signal-judges/j1-coref-agreement-result.md`

<a id="experience-0be04739-534b-4a5f-a098-0573be60b6cc"></a>

### 28. J3-A : confirmation de la moyenne calibrée J2

Expérience `0be04739-534b-4a5f-a098-0573be60b6cc` — famille de données : **ESC**.

**Méthode.** Ajuster sur DEV44 puis figer le mélange f/J2 : poids 0,5 et seuil 0,56 dans la bande ; hors bande, décision d'origine. Collecter les signaux d'un nouveau lot RES30 sans régler la règle sur ses labels.

**Données et protocole.** 30 documents, 20 topics, 21 014 candidats ordonnés ; 767 paires non ordonnées dans la bande, deux jugements. Inférence partagée avec J3-B ; IC 97,5 % pour les deux comparaisons prévues.

**Contrôles rapportés.** Après le run, découverte d'un gold canonique dans un seul sens comparé à des candidats symétriques. Correction déterministe du gold dans les deux sens, sans changement de prédictions ni nouveau seuil ; double vérification par métriques canoniques officielles et bootstrap. Les F1 bruts du journal sont invalides.

**Bilan et limites.** Résultat corrigé : base 36,0698, candidat 36,3525, +0,2827 point ; IC97,5 par topic [−3,4041 ; 2,8997]. Gain non confirmé. Utiliser le rapport corrigé et non les F1 initiaux.

Branche : `orx/j3-a-res30-confirmation-of-frozen-j2-calibrated` ; tête au moment de l’inventaire : `7ddf032e8f0af4645e128de244b8bff061f3f6f6`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `8f7f88dd-6bed-456b-aad0-dd42fdd2643d` | done | `7ddf032` | 12h12m | disponible |
| `b14da22c-f858-4f44-b690-d63246d47127` | cancelled | `e06758a` | 9h25m | disponible |
| `a744a5b7-a6b5-43f2-bd1c-a7c5782f76b5` | failed | `53cd793` | 7s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/new-signal-judges/res30/res30-results-report.md`

<a id="experience-0a091e08-ee86-4b08-a41e-944b8e5ee36c"></a>

### 29. J3-B : calibration logistique régularisée

Expérience `0a091e08-ee86-4b08-a41e-944b8e5ee36c` — famille de données : **ESC**.

**Méthode.** Apprendre sur DEV44 une régression logistique L2 sur les logits standardisés du score fusionné et du juge. Geler standardisation, coefficients et seuil 0,24 avant RES30. Préserver les décisions initiales hors bande.

**Données et protocole.** Même RES30 et mêmes inférences que J3-A ; aucune nouvelle collecte propre à B. Branche avec apprentissage supervisé des scores, même si les modèles API ne sont pas entraînés.

**Contrôles rapportés.** Contrôle du gel avant labels de réserve et du partage d'inférence. Même correction de symétrie du gold que J3-A ; recalcul indépendant hors ligne sans refit.

**Bilan et limites.** F1 corrigé 35,9878 contre 36,0698, −0,0820 point ; IC97,5 par topic [−4,0831 ; 2,8881]. Pas de gain, pas de promotion.

Branche : `orx/j3-b-res30-confirmation-of-regularized-score-cal` ; tête au moment de l’inventaire : `5c22dcba8598d1f63c3f4b1dc9a2b8df7f4a4142`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `0516bc85-715b-47b6-a144-aa276fdeb4f0` | done | `5c22dcb` | 12h04m | disponible |
| `6f724d67-e29a-4ca5-9359-a4a07c5d3687` | cancelled | `a3aebed` | 9h22m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/new-signal-judges/res30/res30-results-report.md`

<a id="experience-4e3309d0-10d4-4c59-b098-1b4cd01a6639"></a>

### 30. J4-A : fusion logistique supervisée des deux scores initiaux

Expérience `4e3309d0-10d4-4c59-b098-1b4cd01a6639` — famille de données : **ESC**.

**Méthode.** Régression logistique L2=1 sur logits Luna et d1 symétrisés, sans score DeepSeek. Pour chaque fold topic, apprendre moyennes/échelles, coefficients et seuil uniquement sur les autres topics. Ajouter un témoin qui règle seulement le seuil.

**Données et protocole.** DEV44, 22 folds leave-one-topic-out, 10 916 paires canoniques. Développement exposé ; pas de RES30 et pas d'accès à la réserve.

**Contrôles rapportés.** Six tests ciblés rapportés ; contrôle des statistiques d'apprentissage, des folds et des seuils. Reconstruction canonique indépendante des TP/FP/FN et IC ; garde réseau pour un calcul entièrement sur caches.

**Bilan et limites.** F1 46,759 contre 47,323 : −0,564 point ; IC97,5 par topic [−2,576 ; 1,838]. Aucune amélioration ; zéro nouvel appel API.

Branche : `orx/j4-a-dev44-supervised-logistic-fusion-cached-sco` ; tête au moment de l’inventaire : `58e0a09a3c20bd8e63c1b3f551481a84e2e6ac60`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `136a525c-465e-4e65-ac14-22a55b3893e9` | done | `58e0a09` | 11s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/new-signal-judges/j4-offline-fusion/j4-offline-fusion-report.md`

<a id="experience-adf2136a-1962-47be-9e5f-387b85c52102"></a>

### 31. J4-B : ajouter des caractéristiques structurelles à la calibration

Expérience `adf2136a-1962-47be-9e5f-387b85c52102` — famille de données : **ESC**.

**Méthode.** Même apprentissage que J4-A, plus désaccord des modèles, asymétrie des orientations, distances de phrase/tokens, même phrase et même surface. Pas d'identifiant topic ni de coréférence gold comme feature.

**Données et protocole.** Mêmes 22 folds DEV44 et mêmes contrôles ; seule la liste de caractéristiques change. Comparaison exploratoire à la fusion gelée et au témoin de seuil.

**Contrôles rapportés.** Même contrôle training-only de la normalisation et du choix du seuil ; six tests de la famille J4 et reconstruction indépendante. Les petits écarts d'arrondi entre plateformes ne changent pas décisions/F1/IC.

**Bilan et limites.** F1 45,185, −2,138 points ; IC97,5 [−4,397 ; 0,232]. Les features ajoutées n'améliorent pas ce protocole.

Branche : `orx/j4-b-dev44-logistic-fusion-with-pair-structure-c` ; tête au moment de l’inventaire : `093d4e22cf749bd5fe0df6c2aa058953aae7208e`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `040544e6-cb82-472f-b1be-16ec3ceb4e4c` | done | `093d4e2` | 6s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/new-signal-judges/j4-offline-fusion/j4-offline-fusion-report.md`

<a id="experience-f644c423-1a72-44a5-8b37-52e136998f86"></a>

### 32. K0 : demander directement chaque relation à Luna

Expérience `f644c423-1a72-44a5-8b37-52e136998f86` — famille de données : **ESC**.

**Méthode.** Un appel batch high par paire non ordonnée ; sortie causal, direction et probability. Cette dernière était supposée représenter une probabilité causale, mais le modèle l'utilise parfois comme confiance dans sa réponse négative.

**Données et protocole.** Un document DEV44 déjà évalué, 23_9ecbplus.xml ; 120 paires. Smoke technique et estimation de coût avant le round à dix documents.

**Contrôles rapportés.** Validation live du transport, du parsing et de la facturation ; inspection des réponses montrant causal=false avec probability=99. Le primaire utilise le booléen ; les analyses probabilistes non corrigées sont non interprétables.

**Bilan et limites.** F1 75,68 contre 94,44 pour la fusion sur ce document facile. Ce n'est pas une performance corpus ni une estimation stable de l'écart.

Branche : `orx/k0-luna-per-pair-decision-1-doc-test` ; tête au moment de l’inventaire : `0534f92f20f796917e6ba369280a47f532cfddcd`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `f526f74c-454f-48f5-bf1f-73e6bd67adf2` | done | `0534f92` | 54m33s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `synthese-globale-2026-10-29/recapitulatif-actualise-2026-10-30.md`

<a id="experience-ec936f5a-5e3b-4ea3-841f-ed31d44e1852"></a>

### 33. KA : cascade de types puis élimination

Expérience `ec936f5a-5e3b-4ea3-841f-ed31d44e1852` — famille de données : **ESC**.

**Méthode.** Sur la bande de fusion [0,2 ; 0,7), interroger huit types : IDENTITY, PART_OF, TEMPORAL_ONLY, DIFFERENT_INCIDENT, DIRECT_CAUSE, ENABLEMENT, REACTION, PREVENTION. Éliminer progressivement des types puis choisir entre les deux derniers. Garder la décision de fusion hors bande.

**Données et protocole.** Dix documents DEV44, 417 paires dans la bande ; 3 851 appels chat. Comparaison appariée au F1 fusion 50,00 avec bootstrap documentaire.

**Contrôles rapportés.** Traçage des étapes et des types finaux ; candidats et contrôles communs à KB/KC/KD. Pas d'apprentissage des règles à partir des labels de ces dix documents.

**Bilan et limites.** F1 44,89, −5,11 [−9,77 ; −1,20]. 305/417 paires finissent dans un type causal : hausse des faux positifs. La cascade n'agit pas comme un filtre conservateur.

Branche : `orx/ka-type-elimination-cascade-on-fusion-band-dev10` ; tête au moment de l’inventaire : `5edbde6e22867bc40e083381c33ff9c3f12a559b`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `6b39e7f7-371f-4b1a-b86b-fd0bc526a4a1` | done | `5edbde6` | 31m39s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `synthese-globale-2026-10-29/recapitulatif-actualise-2026-10-30.md`

<a id="experience-b68899f8-464f-4f60-8dad-e9e29fb3db53"></a>

### 34. KB : proposer les causes d'une mention cible

Expérience `b68899f8-464f-4f60-8dad-e9e29fb3db53` — famille de données : **ESC**.

**Méthode.** Un appel par mention, répété trois fois : demander quelles mentions la causent, la facilitent, la motivent ou la préviennent. Ajouter à la fusion les paires proposées dans au moins deux réponses sur trois ; seuils 1/3 et 3/3 secondaires.

**Données et protocole.** Les mêmes dix documents du round K ; sortie documentaire centrée sur la cible, 645 appels batch. Comparer sur tous les candidats, y compris les non proposés.

**Contrôles rapportés.** Réparation d'une connexion SQLite partagée entre threads par clients appropriés aux documents. Cache conservé ; contrôle des doublons, des identifiants et de la règle d'union.

**Bilan et limites.** F1 46,74 contre 50,00, −3,26 [−10,90 ; 4,83]. Ajouts non orientés : 25 TP pour 125 FP. Les vrais ajouts ne compensent pas le bruit.

Branche : `orx/kb-event-centric-cause-proposer-dev10` ; tête au moment de l’inventaire : `9b341c4f84f04e906fb1b4cca37bed895f5e9cec`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `8d22731d-0379-490f-b136-f7af0c15365e` | done | `9b341c4` | 3h15m | disponible |
| `27fd79e9-7c2e-47ef-9c01-6099286403e3` | failed | `f3c141a` | 31m37s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `synthese-globale-2026-10-29/recapitulatif-actualise-2026-10-30.md`

<a id="experience-a5ca0832-24c6-494e-95d9-ab8a0bead025"></a>

### 35. KC : décider entre événements puis recopier aux mentions

Expérience `a5ca0832-24c6-494e-95d9-ab8a0bead025` — famille de données : **ESC**.

**Méthode.** Produire trois regroupements de mentions ; conserver les liens d'identité vus au moins deux fois, puis composantes connexes. Interroger chaque paire d'événements avec toutes ses mentions et diffuser la décision à toutes les paires de mentions. Les paires internes aux groupes deviennent non causales.

**Données et protocole.** Même DEV10 que KA/KB ; mesurer intrinsèquement les groupes contre ECB+, puis F1 causal contre les contrôles. 1 383 appels batch.

**Contrôles rapportés.** Réparation SQLite avant l'exécution finale ; contrôle des groupes, de la diffusion et des paires originales. Les variantes fondées sur le champ probability de K0 sont explicitement exclues d'une lecture probabiliste naïve.

**Bilan et limites.** Précision de regroupement 38,13 % ; F1 causal 39,42 contre 50,00, soit −10,58 points. La diffusion amplifie les erreurs d'identité.

Branche : `orx/kc-event-level-decisions-copied-to-mentions-dev1` ; tête au moment de l’inventaire : `8408f74817e7ac68eeee8899c4cb7d52eda77ead`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `52be4709-d8f6-4cee-a36f-0b99a386c96d` | done | `8408f74` | 3h15m | disponible |
| `1083dd4e-0940-47f1-bcfc-06ab2aab848f` | cancelled | `ab529c4` | 32m49s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `synthese-globale-2026-10-29/recapitulatif-actualise-2026-10-30.md`

<a id="experience-28784112-0cb3-4466-9a0b-56f5aeba0b57"></a>

### 36. KD : combiner les signaux hétérogènes du round K

Expérience `28784112-0cb3-4466-9a0b-56f5aeba0b57` — famille de données : **ESC**.

**Méthode.** Rejouer les caches Luna documentaire, d1, KC et KB. Primaire : moyenne Luna/d1/score KC au seuil 0,39, puis union avec les propositions KB à 2/3. Secondaires : moyenne seule et votes binaires 2/4 ou 3/4. Règles fixes, sans apprentissage.

**Données et protocole.** DEV10 identique ; aucune inférence nouvelle. K1 n'est pas inclus dans cet ensemble.

**Contrôles rapportés.** 2 028 cache hits, zéro nouvel appel et zéro coût API frais ; rapprochement de chaque signal et des conventions de confiance de K0. Bootstrap apparié contre la fusion.

**Bilan et limites.** Primaire F1 42,42 contre 50,00. Vote 3/4 : 49,13, encore inférieur. Aucun ensemble testé ne dépasse la fusion sur ce périmètre.

Branche : `orx/kd-diverse-no-training-ensemble-fusion-signals-k` ; tête au moment de l’inventaire : `8cecbcd69a8d9cef1622cc2dc8eebe2f5b0c8943`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `41d741cb-5a24-43bf-a423-1a9364d5866f` | done | `8cecbcd` | 7s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `synthese-globale-2026-10-29/recapitulatif-actualise-2026-10-30.md`

<a id="experience-5a28e40b-a7d0-445b-9ea2-7427f7764149"></a>

### 37. K1 : jugement Luna par paire sur DEV10

Expérience `5a28e40b-a7d0-445b-9ea2-7427f7764149` — famille de données : **ESC**.

**Méthode.** Étendre K0 aux dix documents du round K : un booléen causal par paire non ordonnée, effort high. Pour les secondaires, convertir la confiance dans la réponse en score causal selon le booléen, puis tester une combinaison avec la fusion.

**Données et protocole.** Même DEV10, contrôle fusion 50,00. Ce run est désormais terminé ; les anciens rapports le donnant « encore en file » sont dépassés.

**Contrôles rapportés.** Batch livré, clients par document et cache de la réparation KB ; lecture du JUDGE_RESULT final. Le booléen reste le primaire ; la confiance auto-déclarée n'est pas une probabilité calibrée.

**Bilan et limites.** F1 Luna par paire 40,41 ; combinaison figée avec fusion 42,55. Aucun avantage de cette formulation générique par paire.

Branche : `orx/k1-luna-per-pair-decision-dev10` ; tête au moment de l’inventaire : `4d091c044b76b554a0cc8e8c91d2c39279bec56a`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `591a2c21-7768-4201-b4db-23e6210ae362` | done | `4d091c0` | 24h01m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `synthese-globale-2026-10-29/recapitulatif-actualise-2026-10-30.md`

<a id="experience-519ac1f5-34cf-40a7-9f6d-636adcfa070d"></a>

### 38. H1 : Haiku documentaire sur ESC et CTB complets

Expérience `519ac1f5-34cf-40a7-9f6d-636adcfa070d` — famille de données : **ESC et CTB**.

**Méthode.** Remplacer Luna par Haiku 5.5 batch, xhigh ; conserver sept passages, une démonstration par passage et les prompts propres au corpus. Calculer aussi d1 Noul et appliquer la fusion 0,39 sans refit. Sur ESC, démonstration hors topic ; sur CTB, autre document.

**Données et protocole.** ESC train233, 122 130 paires ordonnées ; CTB183, 19 442 candidats dans le résumé FUSION_RESULT effectivement relu. Une synthèse antérieure indiquait 19 410 ; ce chiffre est corrigé ici au profit du journal source. Comparaisons fusion/d1/lecteur seul appariées à l’intérieur de chaque corpus ; toute équivalence avec une autre révision doit être vérifiée.

**Contrôles rapportés.** Reprise après verrou SQLite ; sorties Haiku et CTB d1 réutilisées depuis le cache. Résumé effectif par corpus, modèle servi, TP/FP/FN et bootstrap documentaire relus. DeepSeek historique n'est qu'une référence non appariée, prédictions originales absentes.

**Bilan et limites.** ESC : fusion 43,27, Haiku seul 42,78, d1 35,31 ; gain fusion/Haiku incertain. CTB : Haiku seul 62,57, fusion 47,69 ; perte −14,88 [−18,60 ; −11,20]. La fusion ESC ne se transfère pas telle quelle à CTB.

Branche : `orx/h1-haiku-5-5-xhigh-x-d1-ultra-frozen-fusion-esc` ; tête au moment de l’inventaire : `36de1a08bc427fb8936a34dabbfeb0035ba47b8f`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `a7f7bbd5-b115-4926-850a-af6eb7f44776` | done | `36de1a0` | 3h47m | disponible |
| `0fe86c46-fe57-4e60-bbb7-f797cf5df371` | failed | `36de1a0` | 3h11m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/rapport-global-projet/rapport-complet-2026-10-27.md`
- `jev-v1/haiku-fusion-plan/h1-fusion-result.json`

<a id="experience-069fa421-bef4-497d-a48a-9510c476a316"></a>

### 39. Qwen documentaire initial : baseline locale sans d1

Expérience `069fa421-bef4-497d-a48a-9510c476a316` — famille de données : **ESC et CTB**.

**Méthode.** Exécuter Qwen3.8-27B sur le serveur local exposant une API compatible, avec les prompts documentaires ESC/CTB, sept lectures et vote historique. Pas de modèle de décision ni de fusion d1. Les versions ultérieures de la baseline doivent être distinguées de ce premier contrat.

**Données et protocole.** Objectif ESC233+CTB183 ; plusieurs essais d'installation, de transport ou de génération ont été interrompus. Un point de contrôle comporte neuf documents CTB entièrement terminés, sans résultat complet ESC.

**Contrôles rapportés.** Manifeste des données, prompts, paramètres, contextes et checkpoints ; vérification de complétude avant calcul partiel. Les réponses tronquées ne sont pas des négatifs. Les runs échoués/annulés sont des incidents et pas des scores de modèle.

**Bilan et limites.** Pas de baseline complète de ce nœud. Les résultats CTB partiels sont conditionnés à la complétion et ne permettent pas de classer Qwen face à H1 sur CTB183.

Branche : `orx/qwen3-8-27b-document-vote-baseline-esc-train233` ; tête au moment de l’inventaire : `e0059db617bacb9ed3d4280947c01da495adfb1d`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `d0ce6e2d-a39b-4770-891a-2394c17d596d` | cancelled | `e0059db` | 4h26m | disponible |
| `cf753794-588e-41d9-8129-1b6d0c22978f` | failed | `5a0404b` | 22m23s | disponible |
| `8996f328-81a1-434c-8b72-e61fd9c86a33` | failed | `b60e830` | 5m12s | disponible |
| `adcf6397-6bd8-4b2c-a5b6-cbf8b3a95566` | cancelled | `99ea557` | 13h30m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/qwen38-baseline/launch-protocol.md`
- `jev-v1/qwen38-baseline/preliminary-d0ce6e2d/preliminary-report.md`

<a id="experience-fe522ecd-a36a-46df-8ab9-f531b29c0fca"></a>

### 40. Alias virtuels A/R : test d'ancrage par identifiants auxiliaires

Expérience `fe522ecd-a36a-46df-8ab9-f531b29c0fca` — famille de données : **ESC**.

**Méthode.** Comparer des handles auxiliaires d'alias au témoin apparié, sans changer les faits du document. Utiliser un nouveau schéma de sortie et un vote au moins 4/7 ; ce protocole ne reprend pas automatiquement la fusion historique 0,39.

**Données et protocole.** Smoke sur un document DEV44, sept réponses par bras. Gate de coût : différence d'usage de tokens au plus 1 %, aucune troncature. Première version plafonnée à 4 096 tokens, puis réparation bornée à 16 384 dans les deux bras avec messages inchangés.

**Contrôles rapportés.** Kit initial : 37 tests reproduits, 25 régressions ajoutées, puis six régressions Luna rapportées. Contrôles des omissions de documents, types, cardinalités et coût non vide. Le premier score mécanique malgré gate échoué est rejeté ; le scorer est conditionné au gate PASS.

**Bilan et limites.** Version valide : 14 réponses complètes ; A F1 16,00 contre R 18,18, mêmes deux TP et trois FP nets de plus. Aucun handle utilisé dans les arêtes. Signal local défavorable, pas résultat sur DEV44 complet.

Branche : `orx/alias-ar-gate-and-smoke-on-frozen-luna-dev44` ; tête au moment de l’inventaire : `3d6120abad654eaf543ed055f364251be95a46cb`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `b81b023d-05a7-4146-8174-e4eed9b940ee` | done | `3d6120a` | 1h08m | disponible |
| `04704436-7ff7-4456-bc8c-44d306547de3` | failed | `0fbf1e5` | 7h08m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `alias-followup-2026-10-27/rapport-resultats-smoke.md`

<a id="experience-927184b6-9039-4e7a-ba0f-2faa6fa56b39"></a>

### 41. M0 hors ligne : planifier la couverture des paires

Expérience `927184b6-9039-4e7a-ba0f-2faa6fa56b39` — famille de données : **ESC**.

**Méthode.** Construire quatre bras : O extraction ordinaire, P permutation bijective des IDs, M partition des candidats autorisés en sept blocs, MP combinaison. Conserver le texte et les candidats originaux ; inverser la permutation à la lecture des sorties et compter les expositions par paire.

**Données et protocole.** Pas de modèle dans ce nœud. Planification vérifiée sur les 61 065 paires non ordonnées d'ESC233 ; huit documents DEV44 prévus dans quatre strates de taille, soit 224 prompts rendus.

**Contrôles rapportés.** 15 tests rapportés et 150 graphes aléatoires : déterminisme, bijection, union exacte des masques et exposition unique des paires. Préflight des contextes par borne UTF-8+1 024, non par tokenizer exact. Un écart de taille de bloc jusqu'à deux est conservé et expliqué par les restes de strates.

**Bilan et limites.** Validation d'infrastructure et de couverture PASS ; zéro appel modèle et aucun F1. La qualité de M/P/MP est testée dans le smoke distinct.

Branche : `orx/m0-couverture-des-paires-pilote-o-p-m-mp-sur-dev` ; tête au moment de l’inventaire : `53f71274662e8ba0bfc948e2a1fb5acdf93d083f`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `19f7f9a9-4a9e-44b8-a151-dfe578971508` | done | `53f7127` | 11s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `couverture-sortie/m0-pilot/rapport-etape-m0-2026-10-27.md`

<a id="experience-3b57057e-71b0-4900-b8aa-2efd65d4f204"></a>

### 42. P1 v1.1 : extraire des preuves sources avant transport local

Expérience `3b57057e-71b0-4900-b8aa-2efd65d4f204` — famille de données : **ESC**.

**Méthode.** Partir des positifs de la fusion gelée et extraire des certificats causaux textuels : source, preuve citée et liaison aux occurrences. Le plan compare ensuite identité locale A0, cadres A1, témoin transporté A2 et groupes historiques A3, avec vérification commune, sans récursion des ajouts.

**Données et protocole.** Extraction initiale sur DEV44 : 44 requêtes, plafond 8 192 tokens incluant raisonnement. Le gate de volume exige 50 possibilités exploratoires avant la comparaison finale des bras.

**Contrôles rapportés.** Après échec, récupération GET-only des six lots déjà soumis, sans nouvelle génération. Contrôles des citations littérales, couverture des mentions, reçus et coût. 39 réponses complètes ; cinq troncatures ; 26 certificats structurels utilisables, sans certification sémantique indépendante.

**Bilan et limites.** Volume insuffisant, même en réparant les emplacements tronqués/invalides selon le diagnostic. Pas de registre final de candidats, pas de comparaison A0–A3, pas de F1. Le run de récupération terminé ne transforme pas l'essai scientifique en succès.

Branche : `orx/p1-local-causal-proof-transport-a0-a1-a2-a3-dev4` ; tête au moment de l’inventaire : `3663d645fb1e31e2dbb8246220ffb91ce08c62ae`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `ca2990dd-5338-418c-bfdb-abd220a32782` | done | `3663d64` | 1h00m | disponible |
| `e390a333-f7da-4844-aaca-aac99607ac8f` | failed | `edb98b6` | 4h35m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `transport-preuve-locale/pilote-batch/echec-e390/collecte-finale/bilan-final-extraction.md`

<a id="experience-2a1c6883-9130-4746-8064-2c19e131653f"></a>

### 43. M0 live : O/P/M/MP sur deux documents

Expérience `2a1c6883-9130-4746-8064-2c19e131653f` — famille de données : **ESC**.

**Méthode.** Exécuter sept lectures dans chaque bras. Pour chaque bras, ajouter à P0 uniquement les paires Z0, négatives sous P0 et à zéro vote Luna historique, proposées au moins une fois. Conserver les décisions P0 ; aucune fusion directe de V/E avec 0,39.

**Données et protocole.** Deux documents prédéfinis, 1 246 paires non ordonnées, 56 requêtes. Même calendrier de démonstrations hors topic ; contraste primaire M−O. Une paire voit sept sorties O/P et une seule exposition M/MP : les dénominateurs sont explicites.

**Contrôles rapportés.** 56 sorties complètes, dont quatre listes vides valides ; une paire hors masque filtrée. Reconstruction réponse→registre→ajouts avant gold ; couverture, permutation inverse et usage tokens contrôlés.

**Bilan et limites.** P0 F1 36,92 ; +O 44,12 ; +M 37,14 ; +MP 42,86 ; +P 34,29. Primaire M−O −6,97 points. Union O+MP à 49,32 est post hoc et consomme plus de calcul ; elle motive M1, sans promotion M0.

Branche : `orx/m0-smoke-live-o-p-m-mp-2-docs-dev-56-requ-tes` ; tête au moment de l’inventaire : `4811a1b9663e55e957d4587f6a59b90d2c678f76`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `08eaa19b-af21-400d-94c9-20e96d0dbb93` | done | `4811a1b` | 7h06m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `couverture-sortie/m0-pilot/rapport-resultats-m0-2026-10-27.md`

<a id="experience-4b3cd218-0007-4211-b414-4840b7175316"></a>

### 44. Placebos V0 : faisabilité d'éditions contrôlées automatiques

Expérience `4b3cd218-0007-4211-b414-4840b7175316` — famille de données : **ESC**.

**Méthode.** Pour une relation sélectionnée, demander une transformation cible et trois placebos comparables, en protégeant mentions et faits requis. Un auditeur automatique devait valider les éditions, puis l'extracteur mesurer la réponse sous transformation. Pas de nouvelle annotation humaine.

**Données et protocole.** Douze paires sélectionnées sans gold ; six disposent des snapshots requis, sur quatre documents. Six appels éditeur effectifs sous plafond 2 USD.

**Contrôles rapportés.** Schéma, IDs et snapshots figés ; réponses stop et identifiants uniques. Premier échec au garde budgétaire avant POST, correction comptable sans modification des messages. Audit des hashes et des refus ; pas de changement opportuniste des cas après refus.

**Bilan et limites.** Six refus explicites de construire un ensemble valide ; zéro auditeur, zéro extraction transformée, zéro signature V0 et aucun F1. L'effet du contraste n'est pas mesuré : le générateur ne satisfait pas les contraintes de l'expérience.

Branche : `orx/agent4-v0-auto-only-placebo-feasibility-cap-2-us` ; tête au moment de l’inventaire : `e4f8086c031644ce0655a50601b9a1d383a847f8`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `fbd22a5d-8571-465a-8acf-f99e99eefd40` | done | `e4f8086` | 6h07m | disponible |
| `5100bfee-0c84-42d6-97ba-942d0218cc23` | failed | `b3e572c` | 37s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `empreinte-placebo-2026-10-27/auto-only-v0/rapport-final-pilote-automatique.md`

<a id="experience-6d8bda08-e1bf-49e3-9b4f-f4592a01cee9"></a>

### 45. CTB attribution : concurrence entre liaisons argumentales

Expérience `6d8bda08-e1bf-49e3-9b4f-f4592a01cee9` — famille de données : **CTB**.

**Méthode.** Réutiliser les votes Haiku H1 et construire des supports lexicaux déterministes. F ajoute selon les faits provisoires ; A ajoute une compétition tenant compte de la portée et des arguments. Garder mêmes candidats/supports et un témoin F ramené à la même couverture que A.

**Données et protocole.** Échantillons verrouillés de 20 unités de développement et 40 de test ; test à 370 paires. 32 seeds fixes pour l'égalisation de couverture ; ablation sans arêtes ; bootstrap par document. Calcul CPU hors ligne.

**Contrôles rapportés.** Règles figées avant évaluation ; aucune nouvelle inférence, aucun label humain de vérité d'attribution. Une seule unité déclenche la concurrence. La première tentative a échoué sur le chemin de uv avant calcul, pas sur l'hypothèse.

**Bilan et limites.** Sur le test : référence 69,14 ; F 65,88 ; A 66,67. A et F à couverture égale donnent exactement les mêmes décisions sur les 32 seeds. Aucun apport propre de la concurrence montré.

Branche : `orx/ctb-cached-attribution-competition-f-vs-a-offlin` ; tête au moment de l’inventaire : `167a1c2cd643acbdabfc91bb33b463ef82c4c1aa`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `e788ecb4-039e-40d4-89d3-9dcd230fdec0` | done | `167a1c2` | 6s | disponible |
| `9da90ea5-13b0-4bec-bc00-6433b4c42e9b` | failed | `167a1c2` | 2s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `agent3-auto-sans-annotation/rapport-pilote-automatique.md`

<a id="experience-34b19e46-8794-48e4-b445-1f1402975a4e"></a>

### 46. M1 brouillon : nœud préparatoire non implémenté

Expérience `34b19e46-8794-48e4-b445-1f1402975a4e` — famille de données : **ESC**.

**Méthode.** Annonce de C15 contre H7O+7MP à partir de M0. Le nœud reste sur le code M0, sans implémentation M1 effective.

**Données et protocole.** Aucun run attaché. L'expérience M1 exécutée est le nœud distinct de la fiche suivante.

**Contrôles rapportés.** Comparaison des branches et du point d'entrée conservée dans le rapport M1 ; absence de duplication de génération.

**Bilan et limites.** Aucun score propre. Le titre d'une expérience ne démontre pas que son implémentation ou son exécution a eu lieu.

Branche : `orx/m1-diversit-compl-mentaire-c15-vs-h7o-7mp-dev20` ; tête au moment de l’inventaire : `4811a1b9663e55e957d4587f6a59b90d2c678f76`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

**Aucun run attaché** : préparation ou brouillon, pas de résultat d’exécution propre.

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `m1-diversity/m1-final-report.md`

<a id="experience-e06a915f-f97c-4f97-8310-bc3338d9b0e4"></a>

### 47. M1 : diversité complémentaire contre lectures ordinaires

Expérience `e06a915f-f97c-4f97-8310-bc3338d9b0e4` — famille de données : **ESC**.

**Méthode.** C=15 lectures ordinaires ; H=7 lectures ordinaires communes +7 MP. Générer une seule fois les sept communes : 7+8+7=22 requêtes uniques/document. Ajouter uniquement à Z0 ; maintenir P0. Le nombre 15 du contrôle provient d'une allocation de coût préenregistrée, pas d'un réglage après résultats.

**Données et protocole.** 20 documents de 20 topics, 4 347 paires non ordonnées ; 440 requêtes. Comparaison H−C, ajouts TP/FP, exposition valide par paire, 20 000 bootstrap topics et retraits de topics. Pas de sélection du meilleur passage.

**Contrôles rapportés.** Gel/hashes avant évaluation ; inversion des permutations, déduplication, compteurs E_planned/submitted/delivered/valid. 18 troncatures au plafond 10 000, mises en quarantaine. Le ratio de coût pondéré H/C=0,8483 ne respecte pas ±5 % ; le protocole conserve INCONCLUSIF pour le test à coût égal.

**Bilan et limites.** F1 P0 42,50, C 42,39, H 41,09 ; H−C −1,302 [−3,646 ; 0,170]. H ajoute 17 TP et 97 FP. Pas de gain validé ; la complémentarité de certains TP n'assure pas leur utilité nette.

Branche : `orx/m1-diversity-c15-vs-h7o7mp-dev20-execution-a939` ; tête au moment de l’inventaire : `9cc4587780530f7061089867715de3e88d2e30cf`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `0be48b97-e191-455c-ad13-db1214fd22cd` | done | `9cc4587` | 1h24m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `m1-diversity/m1-final-report.md`

<a id="experience-b04d73c3-1dd9-42a9-a27b-d321278bb250"></a>

### 48. Qwen xhigh, température zéro : reprise de baseline

Expérience `b04d73c3-1dd9-42a9-a27b-d321278bb250` — famille de données : **ESC et CTB**.

**Méthode.** Déclarer le contrat local thinking xhigh et température 0, conserver prompts ESC/CTB, sept lectures, autres paramètres d'échantillonnage et checkpoints compatibles. Les namespaces de cache évitent de mélanger les configurations incompatibles.

**Données et protocole.** Objectif ESC233+CTB183 ; analyse préliminaire limitée aux documents entièrement terminés. Les changements demandés de température/thinking appartiennent à cette version, pas au premier Qwen.

**Contrôles rapportés.** Vérification des corps de prompts avant reprise ; sources et paramètres effectifs dans le journal. Interruptions/annulations conservées. Les documents complets, les lectures isolées et les scores définitifs ne sont pas des unités interchangeables.

**Bilan et limites.** Exécution mixte ensuite arrêtée au profit de CTB seul. Pas de score final des deux corpus attribué à ce nœud ; consulter les bilans préliminaires et la continuation CTB.

Branche : `orx/qwen3-8-xhigh-temperature-zero-resume-esc-train2` ; tête au moment de l’inventaire : `38e222e3b926fe3ced3818fcddb6e3f64d963358`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `a082a2f5-96b1-4c75-a89f-8bbed54d2ed2` | cancelled | `38e222e` | 41h23m | disponible |
| `6233eb3f-8d87-4e56-a769-70c6cea0ac5a` | cancelled | `15c9fe2` | 14m21s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/qwen38-baseline/thinking-xhigh-relaunch.md`
- `jev-v1/qwen38-baseline/preliminary-2026-10-31/comparison-report.md`

<a id="experience-3b10f583-c944-4dfa-92ba-1a5ff8873bbd"></a>

### 49. N1 : filtrer les ajouts par preuve et ancrage

Expérience `3b10f583-c944-4dfa-92ba-1a5ff8873bbd` — famille de données : **ESC**.

**Méthode.** Reprendre 120 candidats distincts ordinaires/MP de M1 ; demander support causal et rattachement exact aux mentions. Les citations doivent être littérales et couvrir les occurrences requises. Comparer C filtré aux ajouts ordinaires communs +MP filtrés H, sans retirer P0.

**Données et protocole.** Même DEV20 que M1, donc exposé. 45 réponses de vérification ; critères gelés : gain au moins un point, utilité marginale positive et stabilité au retrait de chaque topic.

**Contrôles rapportés.** Audit indépendant reconstruit décisions et métriques depuis les réponses, 45/45 complètes, aucune troncature. Contrôle du gel avant gold, des empreintes et du rapprochement coût batch. La validité d'une citation n'est pas une validation humaine du lien causal.

**Bilan et limites.** P0 F1 42,497 ; C 42,950 ; H 42,340. H−C −0,610, IC95 topic [−1,824 ; 0,463]. NO_GO opérationnel ; ajouts H 5 TP/22 FP, précision marginale 18,52 %.

Branche : `orx/n1-automatic-evidence-and-mention-anchoring-on-m` ; tête au moment de l’inventaire : `c39ed37fca230d97427f90e1d9a11f89e870bbf8`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `82e1d497-7441-4115-b5a3-6cd499689b67` | done | `c39ed37` | 1h13m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-sans-entrainement-2026-10-28/n1-ancrage/bilan-verifie.md`

<a id="experience-338d526f-2cd1-4e12-9783-10db01df8e32"></a>

### 50. N2 : transport local v2, même pool et vérificateur commun

Expérience `338d526f-2cd1-4e12-9783-10db01df8e32` — famille de données : **ESC**.

**Méthode.** Réemployer les 26 certificats structurels P1 ; vérifier séparément leur source. Générer lexicalement neuf substitutions et les soumettre aux quatre politiques : A0 identité d'occurrence, A1 compatibilité des cadres, A2 réutilisation du témoin causal, A3 groupes T2. Un vérificateur final commun décide l'admission ; pas de récursion.

**Données et protocole.** 62 nouvelles réponses, neuf candidats sur cinq documents. Gate de rendement : au moins cinq ajouts sur trois topics. Le registre de candidats est gelé avant décision, et aucune lecture du gold n'est requise si les prédictions restent identiques à P0.

**Contrôles rapportés.** 62/62 réponses complètes ; audit indépendant des corps, IDs, citations, hashes et quatre bras. 24 sources sur 26 déclarées soutenues par le modèle ; cela n'est pas une précision sémantique certifiée. Les mêmes neuf candidats et le même vérificateur empêchent de confondre mécanisme et richesse du dossier.

**Bilan et limites.** Admissions pré-vérificateur 1/1/0/1 pour A0/A1/A2/A3 ; zéro ajout final partout. ΔF1 exactement nul par identité des prédictions ; précision des ajouts indéfinie. Mécanismes non départagés, pool lexical trop peu productif.

Branche : `orx/n2-automatic-local-proof-transport-v2-feasibilit` ; tête au moment de l’inventaire : `4d4a27c4d3e1c971fa974d789f0febde52eda9b1`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `e3d9885b-2297-48d0-84d7-fbffe8745723` | done | `4d4a27c` | 5h12m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-sans-entrainement-2026-10-28/n2-transport/bilan-verifie.md`

<a id="experience-a44ecab3-6c74-4895-84e0-344510820848"></a>

### 51. N5 : ablation exhaustive 1/3/7 lectures sur caches

Expérience `a44ecab3-6c74-4895-84e0-344510820848` — famille de données : **ESC**.

**Méthode.** Rejouer les sept réponses Luna historiques de CONF40. Évaluer les sept singletons, les 35 triples et l'ensemble de sept ; ne choisir aucun meilleur sous-ensemble. Recalculer les votes et la fusion historique avec d1 identique et seuil 0,39.

**Données et protocole.** 40 documents, 20 topics, déjà exposés. Rapporter séparément moyenne des F1 et F1 des comptes moyens, car ces opérations ne commutent pas. Bootstrap topics des différences fondées sur les comptes attendus.

**Contrôles rapportés.** Probabilités reconstruites sur sept lectures identiques aux métadonnées, zéro écart. Entrées immuables, script sans client modèle, zéro appel frais. Le bootstrap utilise 2 000 tirages seed 20261028.

**Bilan et limites.** F1 moyen : 1 lecture 38,362 ; 3 lectures 40,779 ; 7 lectures 40,928. Perte moyenne à trois −0,149 point, économie de 57,14 % des appels Luna uniquement. Non-infériorité future non établie ; N8 la teste avec un vrai préfixe frais.

Branche : `orx/n5-cached-luna-1-3-7-pass-fusion-ablation` ; tête au moment de l’inventaire : `780c6eefff8556a1d277c8272cac010da4950839`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `ac747a2d-8ea7-4066-990c-f94539c3cc54` | done | `780c6ee` | 7s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-sans-entrainement-2026-10-28/n5-passes/rapport.md`

<a id="experience-c007d4b4-6142-4b2d-a3a1-7da0e3c8716b"></a>

### 52. N3 : paquet de démonstrations contrastives

Expérience `c007d4b4-6142-4b2d-a3a1-7da0e3c8716b` — famille de données : **ESC**.

**Méthode.** Comparer trois lectures ordinaires, une démonstration positive hors topic par lecture, à trois lectures avec sélection structurelle et négatifs explicites. Même Luna high, sortie 32 768, d1 et fusion symétrique 0,39. Deux interventions restent empaquetées dans le bras contrastif.

**Données et protocole.** Douze documents de douze topics DEV exposés ; 72 requêtes, 4 656 candidats orientés. Primaire : delta F1 avec IC95 par topic ; gate ≥1 point et borne basse >0. Une seule réalisation par bras.

**Contrôles rapportés.** 72 sorties JSON strictes, aucune duplication/ID inconnu ; corps comparés aux hashes de préflight, exemples hors topic. Audit distinct reconstruit votes orientés, fusion, comptes, bootstrap et coût depuis les réponses brutes, sans appeler l'agrégateur du projet.

**Bilan et limites.** F1 45,473→49,513, +4,040 [−0,950 ; 10,428]. Gate non franchi malgré les retraits de topic favorables. Signal initial à répliquer, aucune promotion.

Branche : `orx/n3-fresh-paired-ordinary-versus-contrastive-demo` ; tête au moment de l’inventaire : `f2f58ed629268e44a20458a2e80174d12942c977`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `ff35cbc2-b157-4564-87c0-271081d105ff` | done | `f2f58ed` | 2h00m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-sans-entrainement-2026-10-28/n3-contrastes/bilan-verifie.md`

<a id="experience-3c1651c8-a305-44a9-a1ac-32096dc3359a"></a>

### 53. N4 : remplacer Luna par DeepSeek documentaire

Expérience `3c1651c8-a305-44a9-a1ac-32096dc3359a` — famille de données : **ESC**.

**Méthode.** Donner les mêmes messages/démonstrations aux deux lecteurs, trois slots par document ; Luna batch et DeepSeek direct après rejet du transport batch DeepSeek. Effort high et plafond de complétion 32 768. Primaire lecteur seul, majorité 2/3 ; fusion secondaire déclarée séparément.

**Données et protocole.** Douze documents/12 topics, 72 réponses attendues. Exiger toutes les réponses complètes avant d'ouvrir le gold. Le secondaire utilise un vote non orienté par lecture, distinct de la fusion historique orientée.

**Contrôles rapportés.** Audit brut : Luna 36/36 stop ; DeepSeek 12/36 stop et 24 length, sans contenu final. Raisonnement DeepSeek 99,788 % des tokens de complétion. Sorties gelées, metrics=null, aucun sous-ensemble complet scoré pour remplacer le primaire.

**Bilan et limites.** BLOCKED_VALIDITY ; aucun F1 comparatif. Le problème observé est le contrat de génération, pas une mauvaise qualité causale mesurée de DeepSeek.

Branche : `orx/n4-paired-deepseek-document-reader-versus-luna-n` ; tête au moment de l’inventaire : `f6f746a75a3e4088f2162fe03d01ac838de1cccf`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `c3a66705-3b67-4cde-869f-055cda7dab17` | done | `f6f746a` | 1h14m | disponible |
| `239d529e-1438-46bc-8417-7260f5c26fab` | failed | `c2f335c` | 8s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-sans-entrainement-2026-10-28/n4-lecteur/bilan-verifie.md`

<a id="experience-95286508-31f4-4318-9ea6-cb7fa3bb7392"></a>

### 54. N6 : question lexicale adaptée à CTB

Expérience `95286508-31f4-4318-9ea6-cb7fa3bb7392` — famille de données : **CTB**.

**Méthode.** Comparer avec d1 une question causale générique à une question exigeant un ancrage lexical CTB. Même contexte et une orientation canonique par paire ; seuil 0,5 dans les deux bras. Pas de transfert de la fusion ESC à 0,39.

**Données et protocole.** Sept documents CTB sélectionnés par métadonnées/hashes ; 73 paires, deux questions, 146 requêtes. Petit échantillon exposé, pas confirmation.

**Contrôles rapportés.** Huit tests rapportés ; reçus durables avant POST, modèles et probabilités archivés. Premier incident avant POST, puis 60 réponses valides et 86 HTTP429. Les 86 coûts non retournés restent provisionnés. Inspection accidentelle de 12 labels après gel divulguée ; aucune sélection modifiée.

**Bilan et limites.** INCONCLUSIVE_INTEGRITY ; pas de score primaire. Les réponses manquantes ne sont ni négatives ni des erreurs de causalité.

Branche : `orx/n6-ctb-lexical-anchoring-question-versus-generic` ; tête au moment de l’inventaire : `a98842f0fd399924e287e0963c82bb1eed42f909`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `9ad51ca1-9351-4428-bfe6-407ca4bd21be` | done | `a98842f` | 12s | disponible |
| `80ac313f-82c7-4b4b-bc4e-015f1fdb71e7` | failed | `8e0e883` | 7s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-sans-entrainement-2026-10-28/n6-ctb/rapport-n6.md`

<a id="experience-126ff578-40cb-4041-8f37-6b6aad85cf20"></a>

### 55. N3 audit : récupération et reproduction sans génération

Expérience `126ff578-40cb-4041-8f37-6b6aad85cf20` — famille de données : **ESC**.

**Méthode.** Récupérer les sorties de N3 puis recalculer indépendamment son pipeline : parser les 72 réponses, reconstruire votes/fusion, comptes, bootstrap topics, retraits de topics et coûts. Ne pas modifier le nœud scientifique parent.

**Données et protocole.** Exactement les mêmes sorties que N3. Il s'agit d'un test de reproductibilité, pas d'une seconde réalisation stochastique ni d'une confirmation sur de nouveaux documents.

**Contrôles rapportés.** 324 morceaux d'export, SHA de l'archive et 99 fichiers du manifeste vérifiés. Le script d'audit n'appelle ni modèle ni agrégateur original ; l'inventaire du run de récupération indique zéro génération.

**Bilan et limites.** Résultat N3 reproduit exactement ; son statut reste inconclusif. Le nombre d'exécutions ne doit pas faire compter le +4,040 deux fois.

Branche : `orx/n3-reproducibility-audit-of-frozen-paired-guidan` ; tête au moment de l’inventaire : `3e72fdd554e12ff162e0e393fe07f71b9e7e8d4d`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `cbb76514-9cb5-4671-b334-ad18984bbb8a` | done | `3e72fdd` | 13s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-sans-entrainement-2026-10-28/n3-contrastes/bilan-verifie.md`

<a id="experience-2b118e89-feae-4028-8a5f-041b35fadc7a"></a>

### 56. N7 : réplication du paquet N3 sur deux calendriers

Expérience `2b118e89-feae-4028-8a5f-041b35fadc7a` — famille de données : **ESC**.

**Méthode.** Conserver le paquet structure+négatifs N3 et le comparer au contrôle ordinaire sur deux calendriers frais, trois lectures par bras. Même d1 et fusion 0,39. Des prompts contrastifs identiques entre calendriers doivent tout de même produire des générations fraîches distinctes.

**Données et protocole.** Seize documents de seize topics, distincts des documents N3 ; 16×2 calendriers×2 bras×3 lectures=192 requêtes. Primaire : moyenne des deux écarts F1 ; gate ≥1 point, borne IC>0 et gain dans chacun des deux calendriers.

**Contrôles rapportés.** Quatre tests ciblés et revue préflight rapportés ; 192 slots, 135 corps distincts, 64 lots terminés. Gate api_call=true/cache_hit=false, IDs de réponses distincts ; gel avant gold. Audit indépendant rejoue toutes les réponses ; bootstrap 20 000 topics.

**Bilan et limites.** Écarts −3,247 puis +4,329 ; moyenne +0,541 [−1,614 ; 3,372]. Aucun gain robuste. La variation ne peut pas être attribuée uniquement au décodage : calendrier et ordre de soumission font aussi partie des conditions.

Branche : `orx/n7-two-schedule-replication-of-packaged-contrast` ; tête au moment de l’inventaire : `69b91eaa3b3028606086914b8f54bdaad18ee33b`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `350ed2fa-4dfa-4f8c-bed3-2edd259cb5ef` | done | `69b91ea` | 2h46m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-suite-2026-10-28/n7-replication/bilan-verifie.md`

<a id="experience-2483cd7a-30a8-4836-8f97-06386dd31d05"></a>

### 57. N8 : trois lectures fraîches réellement incluses dans sept

Expérience `2483cd7a-30a8-4836-8f97-06386dd31d05` — famille de données : **ESC**.

**Méthode.** Produire un seul calendrier frais de sept lectures ordinaires ; la variante à trois utilise exactement les trois premières réponses, pas un sous-ensemble choisi après les scores. Maintenir d1, fusion et seuil 0,39.

**Données et protocole.** Huit documents/topics ; 56 requêtes. Test de non-infériorité avec marge préfixée d'un point F1. Documents distincts de N7, mais topics pas nécessairement disjoints.

**Contrôles rapportés.** Quatre tests ciblés et intégration synthétique 56 sorties rapportés ; 56/56 réponses réelles complètes. Audit brut des 21 fichiers du manifeste et vérification du partage exact du préfixe ; bootstrap par topic.

**Bilan et limites.** F1 3 lectures 35,685 contre 7 lectures 36,364 ; Δ −0,679 [−3,174 ; 0,292]. La borne basse dépasse la perte tolérée : non-infériorité non établie. Ce n'est pas une preuve d'infériorité universelle.

Branche : `orx/n8-fresh-nested-three-versus-seven-luna-reads` ; tête au moment de l’inventaire : `16092e10a4156f12078fc8b3d590f8bfea9524ec`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `fb701e7a-0619-4fbb-9e2a-fa5775588b83` | done | `16092e1` | 7m06s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-suite-2026-10-28/n8-passes/bilan-verifie.md`

<a id="experience-c2fc283c-e85d-40e6-b231-00aa5759a96a"></a>

### 58. N9 : factoriel récupération × négatifs

Expérience `c2fc283c-e85d-40e6-b231-00aa5759a96a` — famille de données : **ESC**.

**Méthode.** Croiser R0/R1 (exemples aléatoires ou récupération structurelle) et N0/N1 (positifs seuls ou négatifs explicites). Les exemples sont identiques entre N0 et N1 d'un même choix R. Garder Luna high, trois lectures, d1 et fusion 0,39 ; mélanger bras et calendriers dans les lots par document.

**Données et protocole.** Dix documents/topics, deux calendriers : 10×2×4×3=240 générations. Contrastes déclarés : effets principaux, interaction et trois variantes face à R0N0, avec correction de multiplicité.

**Contrôles rapportés.** Huit tests et intégration synthétique à 240 sorties rapportés ; 240 vraies réponses complètes et IDs uniques. Audit autonome des réponses, requêtes, hashes, décisions et coûts. Les interruptions de suivi SSH n'ont entraîné ni nouvelle soumission ni nouvelle génération.

**Bilan et limites.** Face au témoin, négatifs seuls +0,503 ; structure seule −1,041 ; paquet complet −1,152 point en moyenne. Tous les IC corrigés incluent zéro. Aucun composant retenu ; les facteurs peuvent enfin être distingués, sans effet robuste démontré.

Branche : `orx/n9-factorial-structural-retrieval-and-explicit-n` ; tête au moment de l’inventaire : `342709d8b29933db40c64af365f656dc4900ee17`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `d9b0820c-37fb-47a5-bfb9-cf18ef8c2e16` | done | `342709d` | 13h46m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-composantes-2026-10-28/n9-factoriel/bilan-verifie.md`

<a id="experience-bfe65f5c-1775-4082-b398-94ba4b520a71"></a>

### 59. N10 : six lectures contre tous les triples de N7

Expérience `bfe65f5c-1775-4082-b398-94ba4b520a71` — famille de données : **ESC**.

**Méthode.** Réutiliser les deux calendriers de N7 pour obtenir six lectures par bras/document. Comparer les six à la moyenne de chacun des 20 sous-ensembles de trois ; refaire cette moyenne à chaque bootstrap. Comparer aussi six contrastives à six ordinaires.

**Données et protocole.** Seize documents/topics ; 192 réponses déjà payées. Trois contrastes exploratoires, correction Bonferroni, IC 98,3333 %. Ne pas choisir le meilleur triple ni requalifier l'analyse en confirmation de N7.

**Contrôles rapportés.** Six tests ciblés rapportés ; audit indépendant des 42 systèmes de prédictions (21 par bras), gel avant gold. Exécution sans client réseau ; les très petits écarts d'arrondi d'un Python ancien sont distingués des changements de décisions.

**Bilan et limites.** Six−moyenne des triples : ordinaire +0,161 [−1,539 ; 2,538], contrastif +0,544 [−0,246 ; 1,460]. Aucun candidat. Zéro appel frais ne signifie pas que six lectures seraient gratuites en production.

Branche : `orx/n10-cached-six-read-stabilization-versus-all-thr` ; tête au moment de l’inventaire : `1010fcd3d0195181267d021f37f21ae9d0ab25e5`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `8354656e-4a7e-4019-8a4a-8b649fb9d257` | done | `1010fcd` | 10s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-composantes-2026-10-28/n10-stabilite/bilan-verifie.md`

<a id="experience-4254c7f2-ca34-437f-8f28-d61a016ca5ec"></a>

### 60. N11 : arrêt anticipé exact des sept lectures

Expérience `4254c7f2-ca34-437f-8f28-d61a016ca5ec` — famille de données : **ESC**.

**Méthode.** Après k lectures, borner le score final à sept en supposant les futurs votes tous nuls ou tous positifs. Dénominateur toujours sept ; arrêter un document seulement quand toutes ses paires sont irrévocablement classées. Aucun changement de prompt ou de seuil.

**Données et protocole.** Rejeu de deux cohortes compatibles séparées : CONF40 historique et DEV8 N8 frais. Mesures : appels économisés, tokens historiques évités et égalité exacte des décisions. Gate de poursuite : au moins 20 % d'appels économisés.

**Contrôles rapportés.** Replay indépendant en arithmétique rationnelle, sans gold ; vérification des cas de frontière >=0,39 et de la reconstruction de toutes les décisions finales. Inventaire séparé de la disponibilité des caches pour un éventuel corpus complet.

**Bilan et limites.** CONF40 : 18/280 appels économisés, 6,43 % ; DEV8 : 9/56, 16,07 %. Zéro décision changée, gate de 20 % non atteint. Ce test mesure une économie exacte, pas un gain F1.

Branche : `orx/n11-cached-exact-stopping-and-full-run-readiness` ; tête au moment de l’inventaire : `b73126218ae6623db94eb45c805691ab11b5cf16`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `005dff89-b55e-42d8-b81f-63e160fb83ff` | done | `b731262` | 12s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-suite-2026-10-30/n11-arret-exact/bilan-verifie.md`

<a id="experience-bdbab303-7898-4fb0-a9a9-676bda82b084"></a>

### 61. N12 : changer la vue des occurrences

Expérience `bdbab303-7898-4fb0-a9a9-676bda82b084` — famille de données : **ESC**.

**Méthode.** Trois vues déterministes : original ; local, union des phrases de toutes les mentions et de leurs voisines ±1, avec marqueurs d'omission ; table, texte intégral suivi de ID/phrase/surface pour chaque occurrence. Aucun regroupement d'événements ni renumérotation. Garder les mêmes démonstrations dans les trois bras.

**Données et protocole.** Quatre documents ESC exposés, deux calendriers, trois lectures par vue : 72 requêtes, quatre lots mixtes de 18. Même fusion Luna×d1 à 0,39 ; smoke technique/exploratoire sans promotion.

**Contrôles rapportés.** Quatre tests de contrat, dont intégration synthétique à 72 sorties. Assertions : conservation de toutes les mentions/candidats, texte marqué d'origine exact, exemples hors topic et préfixes partagés. 72 réponses réelles complètes et uniques ; audit brut indépendant des votes, F1 et coûts.

**Bilan et limites.** Local : +4,566 puis +1,094 points, moyenne +2,830 ; table : moyenne −0,514. Petit signal sur quatre documents, sans généralisation. C'est une vue documentaire globale réduite, pas un prompt distinct par paire.

Branche : `orx/n12-occurrence-representation-paired-four-docume` ; tête au moment de l’inventaire : `bdb987d3acf93a1cf681d093583baee2d574e6b8`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `f5f58d8c-d3ce-4c2f-b2bd-d6dcc8ca9578` | done | `bdb987d` | 10h36m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-suite-2026-10-30/n12-representations/bilan-verifie-2026-10-31.md`

<a id="experience-0594a3dc-b551-4b9a-9005-82ed671be484"></a>

### 62. N13 : représentations avec Qwen local, versions ESC puis CTB50

Expérience `0594a3dc-b551-4b9a-9005-82ed671be484` — famille de données : **CTB50 ; ESC40 envisagé initialement**.

**Méthode.** Plan initial ESC : quatre documents smoke puis DEV40, trois vues N12, deux calendriers de trois lectures, majorité 2/3 non orientée, sans d1. Le protocole exécuté a ensuite été déplacé vers 50 documents CTB : candidats intra-phrase, démonstrations hors fold ; 900 requêtes prévues, dont 72 de contrôle sur quatre documents avant les 828 restantes. Dans la version CTB, la vue locale présente séparément chaque phrase contenant des mentions, dans son ordre et avec ses IDs originaux ; elle diffère donc de l'union ±1 phrase de N12 sur ESC. Ne pas fusionner ces plans malgré le titre historique du nœud.

**Données et protocole.** CTB50 : Qwen déclaré qwen3.8-27b, température 0. Premier smoke xhigh sortie 32 768 ; second xhigh sortie 65 536, contexte 81 920, entrée 16 384 et concurrence 4 ; puis relance high demandée, distincte. Chaque phase exige toutes ses réponses valides. Primaire : majorité par paire non orientée, une voix par lecture.

**Contrôles rapportés.** Tests de contrat présents : refus de fournisseur externe, 900 corps/72+828, déduplication des sens, tokenizer exact bloquant avant génération, contexte déclaré, exclusions de folds/démonstrations, gel avant gold et IDs uniques. Audits hors réseau des sorties réelles : 63/72 puis 69/72 valides ; le document difficile reste inclus. Les 828 requêtes principales restent bloquées dans ces deux smokes xhigh.

**Bilan et limites.** Le plan ESC40 initial n’a pas généré de résultat : arrêt pendant l’attente de la baseline. Sur CTB50, trois smokes de quatre documents échouent au gate : xhigh/32k 63/72 valides, xhigh/65k 69/72, high/65k 68/72. Les 828 appels de la phase suivante n’ont pas été lancés. Aucun F1 primaire CTB50. Le diagnostic post hoc high, sur trois documents complets (54 réponses, 84 paires, deux positifs), donne TP=0, FP=0, FN=2 et F1 global 0 pour toutes les vues et les deux calendriers. Le run high est done techniquement et BLOCKED_VALIDITY scientifiquement ; il n’est plus actif. Ce diagnostic partiel ne répare pas le gate.

Branche : `orx/n13-qwen-local-occurrence-representations-dev40` ; tête au moment de l’inventaire : `7c07872afbc1e511193d3b3dbbae61f8d4097b13`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `09873773-2908-459d-bced-dc436272fb32` | done ; BLOCKED_VALIDITY | `7c07872` | 2h00m | disponible |
| `2d6766a2-3d10-4a32-a19f-e8009e35efbc` | done ; BLOCKED_VALIDITY | `d2df260` | 1h40m | disponible |
| `e163b810-e9f7-419d-b769-168265cf4c49` | done ; BLOCKED_VALIDITY | `2e77e7c` | 10h46m | disponible |
| `d9abfff4-3ef2-4b8d-9b39-6d267a218614` | failed | `c7b2028` | 33h30m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-suite-2026-10-30/n13-qwen-local/bilan-lancement-2026-10-31.md`
- `campagne-suite-2026-10-30/n13-ctb50-2026-11-02/relance-65k/bilan-final.md`
- `campagne-suite-2026-10-30/n13-ctb50-2026-11-02/evaluation-partielle-2026-11-03/bilan.md`
- `campagne-suite-2026-10-30/n13-ctb50-2026-11-02/relance-high-2026-11-03/bilan-final.md`

<a id="experience-03e141a3-c891-4cbc-95f1-eba83ea3685a"></a>

### 63. Qwen CTB seul : terminer la baseline avant ESC

Expérience `03e141a3-c891-4cbc-95f1-eba83ea3685a` — famille de données : **CTB**.

**Méthode.** Arrêter le contrôleur mixte et ne reprendre que r7_ctb, avec xhigh, température 0, sept lectures et paramètres de la baseline inchangés. Réutiliser uniquement checkpoints et passes dont les corps/paramètres concordent exactement. ESC reste suspendu.

**Données et protocole.** CTB183 uniquement, sans ESC ; sept passes documentaires, reprises de checkpoints et de passes valides sur correspondance exacte des prompts.

**Contrôles rapportés.** 28 tests rapportés et 1 182 prompts source vérifiés identiques. Le garde de reprise autorise un sous-ensemble des datasets sans changer l'identité du corpus, du modèle ou de l'échantillonnage. Annulations liées aux redémarrages de service documentées sans les compter comme nouveaux résultats.

**Bilan et limites.** Pause demandée par l’utilisateur. 179/183 documents réussis, quatre en échec ; 18 passes partielles conservées et 10 réponses valides manquantes. Le processus distant a quitté avec code 1, malgré le statut running périmé. Le résumé à 183 lignes est incomplet et son F1 60,22 % inclut les quatre échecs, pas les seuls documents réussis.

Branche : `orx/qwen-ctb-completion-only-esc-held-until-ctb-fini` ; tête au moment de l’inventaire : `f80a1a4678dde0105a329cb63e200ff6b1c2eed5`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `b41468ee-d81e-4d4c-9df7-1f51309dbde9` | running périmé ; pause, processus terminé | `f80a1a4` | 43h05m | disponible |
| `5ac5da92-f52d-448b-bfbc-cc25fb9f89ce` | cancelled | `f80a1a4` | 10m10s | disponible |
| `02a7955d-ff08-4b38-82fe-8813a18a1db2` | cancelled | `f80a1a4` | 4m24s | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/qwen38-baseline/ctb-only-continuation-2026-11-01.json`
- `jev-v1/qwen38-baseline/publication-2026-11-03/ctb179-progress.json`

<a id="experience-91c2b8fe-5c86-4644-b1d9-ff6354054392"></a>

### 64. N14 : factoriel prompt × exécution bornée, Qwen medium sur CTB4

Expérience `91c2b8fe-5c86-4644-b1d9-ff6354054392` — famille de données : **CTB**.

**Méthode.** Quatre conditions appariées : ancien prompt/document entier (M00), prompt causal nettoyé/document entier (M10), anciennes règles/lots explicites de paires bornés (M01), les deux changements (M11). Trois représentations, deux calendriers et trois lectures. Paramètre medium envoyé dans la requête et le template ; différence medium/xhigh vérifiée dans le rendu du tokenizer. Les changements constituent des ensembles déclarés, sans attribution à un seul composant.

**Données et protocole.** Les mêmes quatre documents CTB de N13, dont WSJ910225-0066, 593 paires non orientées. Au maximum 72+72+594+594=1 332 appels, huit tâches simultanées. Entrée 16 384, sortie 65 536, contexte déclaré 81 920. Pas d’extension automatique à CTB50.

**Contrôles rapportés.** Six tests hors ligne rapportés PASS dans les notes de lancement ; gate de couverture complète pour les lots explicites, aucun remplacement des réponses manquantes par norel, décisions figées avant lecture du gold. Aucun de ces tests n’a été réexécuté pour cette publication.

**Bilan et limites.** En cours dans l’instantané de suivi ; journal de configuration disponible, aucun résumé final N14_RESULT observé. Aucun F1 final attribué. Aucun lancement, arrêt ou changement de ce run dans cette publication.

Branche : `orx/n14-medium-paired-prompt-and-bounded-execution-c` ; tête au moment de l’inventaire : `93b2a3357cfeda1d128538284205bf2f07fd57ed`. Chaque ligne de run conserve ci-dessous son propre commit, qui peut être différent.

| Run | Statut enregistré | Commit | Durée rapportée | Journal |
|---|---|---|---|---|
| `208d3236-fa28-438c-be67-62e6552f1495` | running | `93b2a33` | 3h34m | disponible |

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `campagne-suite-2026-10-30/n14-medium-factoriel-2026-11-03/protocole-et-lancement.md`

<a id="experience-42d6a6ad-3405-478a-9ce7-c892bae94e8e"></a>

### 65. J1 brouillon : accord de deux procédures DeepSeek

Expérience `42d6a6ad-3405-478a-9ce7-c892bae94e8e` — famille de données : **ESC**.

**Méthode.** Projet initial : intersection de trois groupements DeepSeek puis confirmation pairwise par DeepSeek, avec gate de précision 85 %. Ce n'est pas la variante Luna×DeepSeek finalement demandée.

**Données et protocole.** Nœud archivé, sans run. L'essai effectif est le nœud J1R.

**Contrôles rapportés.** Vérification de l'archive et de l'absence de lancement. Le code préparé ne vaut pas observation expérimentale.

**Bilan et limites.** Aucune métrique propre ; ne pas compter comme une expérience négative supplémentaire.

**Aucun run attaché** : préparation ou brouillon, pas de résultat d’exécution propre.

Sources de consolidation (chemins dans les artefacts de recherche, empreintes dans le fichier de provenance) :

- `jev-v1/new-signal-judges/j1-j2-results-report.md`

## Exports et vérifications

- [Tous les runs, CSV](project-runs.csv) : une ligne par tentative, y compris les échecs et annulations.
- [Tous les runs, JSON](project-runs.json) : identifiants complets, rattachement, statut enregistré, commit et avertissements d’interprétation.
- [Fiches des 65 expériences, JSON](project-experiments.json) : méthode, périmètre, bilan consolidé, branche et runs associés.
- [Index de preuves](project-log-evidence.json) : noms des événements de résumé, numéros de lignes sources et empreintes ; aucun contenu de journal n’est publié. L’absence d’un événement dans cet index ne prouve pas l’absence de résultat dans le journal complet.
- [Provenance des rapports](project-source-provenance.json) : empreintes des sources de consolidation.
- [Manifeste de publication](manifest.json) : empreintes de tous les fichiers publiés.

Les contrôles documentaires vérifient la présence de chaque run de l’instantané, l’unicité des identifiants, le rattachement de chaque run à une expérience, la concordance CSV/JSON, les liens relatifs et les empreintes. Ils ne remplacent pas un rejeu des inférences. Les données du suivi sont un instantané ; les prochains runs ne s’ajoutent pas automatiquement à ce commit.
