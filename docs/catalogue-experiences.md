# workere — catalogue détaillé des 117 expériences

**Mise à jour du 12 novembre 2026.** Ce catalogue rassemble une fiche pour chacun des 117 nœuds suivis et renvoie aux 190 exécutions, reprises comprises. Il complète la [synthèse destinée à la présentation](synthese-resultats.md).

Les fiches 1–64 conservent leur détail technique historique ; les états nouveaux de N13 et de la reprise CTB y sont signalés. Les fiches 65–117 couvrent les campagnes de novembre. Pour tout état courant, l’[inventaire du 12 novembre](../SOURCES.md#annexes-locales) prime sur les phrases datées des anciens essais. Les mentions de tests historiques décrivent les contrôles rapportés ; cette mise à jour ne prétend pas les avoir tous réexécutés.

Les valeurs E1/E2/E3/E5 suivent le replay strict structuré, qui corrige certains résumés antérieurs. Les pilotes W5/W6 finaux portent sur quatre documents chacun malgré leurs titres hérités ; W7/W8 ne disposent pas encore de résultat final. Voir le [registre des écarts](corrections-et-provenance.md).

Chaque fiche distingue l’intervention, les données, le résultat et les limites. Un run terminé, une reprise ou un échec technique n’est pas une confirmation indépendante. Les sources, journaux et commits sont indexés ; aucun modèle n’a été appelé pour cette mise à jour.

## 1. Contrat de la tâche et métriques

### 1.1 Entrées, candidats et cible

Dans les campagnes principales, les mentions d’événements et les paires candidates sont **fournies par le corpus**. Le système prédit l’existence d’une relation causale entre ces mentions. Il ne faut donc pas lire le F1 comme celui d’un extracteur d’événements de bout en bout. Les informations publiques incluent texte, positions des mentions et candidats ; les relations gold de la cible restent réservées à l’évaluation. Les démonstrations peuvent, elles, contenir les annotations des exemples autorisés.

ESC utilise souvent tous les couples ordonnés de mentions distinctes, avec candidats intra et inter-phrases. CTB conserve un périmètre de paires intra-phrase. Le score principal discuté ici est **binaire, sans direction** ; les labels `causes`/`causedby` peuvent exister dans les sorties et les votes historiques sans devenir le critère directionnel principal.

Le compte CTB dépend de la révision et du plan : H1 annonce 19 410 candidats, alors que l’audit du builder corrigé en compte 19 442. On conserve le compte du plan effectivement évalué ; on ne suppose pas ces corpus identiques. Voir [H1](../SOURCES.md#annexes-locales), [audit CTB](../SOURCES.md#annexes-locales) et [migration du benchmark](../SOURCES.md#annexes-locales).

### 1.2 Scores et agrégation

Avec TP vrais positifs, FP faux positifs et FN faux négatifs :

$$P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2TP}{2TP+FP+FN}.$$

Le micro-F1 additionne les comptes de tous les documents avant la division. Ce n’est pas la moyenne des F1 documentaires. L’AP mesure le classement sans choisir un seuil ; une hausse de F1 n’implique pas une hausse d’AP.

Si l’on compte les deux orientations d’une relation symétrique, les comptes peuvent être doubles de ceux du format canonique, avec le même F1. Ces orientations ne sont pas deux observations indépendantes. Le bug J3 a justement consisté à comparer des prédictions dans les deux sens à un gold conservé dans un seul sens ; les métriques corrigées font foi.

**d1/Noul.** Une question oui/non par paire ordonnée, avec document entier et occurrences exactes ; seuil initial 0,5. Les mots `probability` et `confidence` ne sont pas interchangeables, comme le montre K0.

**Lecteur documentaire.** Luna, Haiku ou Qwen lisent un document et produisent une liste éventuellement vide. Une lecture peut omettre une paire ; le traitement de cette omission dépend du protocole de vote, sous réserve que la réponse entière soit valide. Une troncature ou une erreur HTTP n’est jamais une réponse vide valide.

**Fusion ESC historique.** Avec $v$ les fractions de votes Luna et $d$ les scores d1 :

$$f(a,b)=\frac{v(a,b)+v(b,a)+d(a,b)+d(b,a)}4,\qquad \hat y=\mathbb{1}[f\ge0{,}39].$$

Le seuil 0,39 a été choisi avant TEST-S40. Les campagnes qui déclarent un autre vote — alias, N4 secondaire, Qwen/N13 — ne sont pas automatiquement équivalentes à cette fusion. Les règles de déduplication, dernier label et égalité restent celles du contrat de chaque essai. Voir [R3](../SOURCES.md#annexes-locales) et [N4](../SOURCES.md#annexes-locales).

### 1.3 Jeu de développement, confirmation et stochasticité

| Périmètre | Usage dans son étape historique | Limite actuelle |
|---|---|---|
| Smoke d’un document | Intégration API, parsing, coût, premier diagnostic | Ne valide pas une généralisation |
| DEV-SCREEN 10 / DEV25 | Tri et réglage des variantes R1/R2 | DEV25 concentré sur deux topics |
| D15 | Diagnostic Mercury/Luna, exclu de la sélection R1/R2 | Ensuite réutilisé dans POOL80 pour choisir la fusion R3 |
| CONF40 | Confirmation après gel de R2 et Luna | Ensuite intégré à POOL80, puis réutilisé dans N5/N11 |
| POOL80 / TEST-S40 | Choix hors topic puis évaluation de la fusion | Ne reste pas un test intact après analyses répétées |
| DEV44 / DEV10 | R4, juges, round K et pilotes | Développement exposé |
| RES30 | Confirmation historique de J2 dans J3 | Consommé après cette confirmation |
| DEV12/16/8/10 et CTB50 récents | Cohortes propres aux protocoles N | Pas de réserve indépendante revendiquée |

Un bootstrap apparié rééchantillonne **les mêmes documents ou topics pour les deux systèmes**, puis recalcule les comptes. Les milliers de paires ne sont pas tirées indépendamment. Le niveau de confiance et la correction de multiplicité changent selon les campagnes ; ils sont précisés dans les fiches. Un IC contenant zéro signifie que le gain n’est pas établi, pas que les systèmes sont équivalents.

Un « calendrier » indique une réalisation planifiée des lectures/démonstrations. Deux calendriers ne signifient pas toujours que seule la seed de décodage change. Les répétitions de générations, leur ordre de soumission et parfois les exemples du contrôle peuvent varier.

Les anciens 88 documents dits réservés apparaissent tous dans H1 : ils ne sont plus une réserve intacte à l’échelle du projet. Cela ne démontre pas la consultation de chacun de leurs labels. Voir [audit de réserve](../SOURCES.md#annexes-locales).

## 2. Ce que signifie « testé » dans ce projet

| Niveau | Contrôles concrets | Ce que cela établit |
|---|---|---|
| Tests unitaires | Spans/IDs, candidats, remappage, vote, seuil, erreurs de schéma | Le comportement du code sur les cas vérifiés |
| Tests d’intégration synthétiques | Faux transport, réponses construites, pipeline complet jusqu’au gel et aux métriques | La cohérence du pipeline, sans score scientifique |
| Préflight sur vraies entrées | Cohorte, exclusions, nombre exact de requêtes, budget et contexte | La conformité du plan avant génération |
| Smoke réel | Appels fournisseur et réponses enregistrées | La faisabilité locale, pas une généralisation |
| Gate de validité | Tous les IDs attendus, réponses complètes, modèle déclaré, schéma/candidats et fraîcheur | Le droit de calculer la comparaison prévue |
| Évaluation appariée | Mêmes documents et contrôles, sortie gelée avant gold | Une différence mesurée sous ce protocole |
| Audit indépendant | Repartir des requêtes/réponses et reconstruire votes, décisions et comptes | La reproductibilité du calcul, pas une réplication scientifique |
| Confirmation après gel | Règle choisie ailleurs, lot tenu à l’écart à cette date | Une épreuve de transfert sous les limites de ce lot |

Les gardes les plus utiles bloquent avant dépense ou avant scoring : dépassement du contexte, budget non réservé, réponse tronquée, ID inventé, paire manquante, réponse fournisseur réutilisée alors qu’une génération fraîche était exigée. Les tests de cache vérifient aussi qu’un retry de transaction SQLite ne rappelle pas un modèle payant.

Pour comprendre N11, noter $d_s=(d_{ab}+d_{ba})/2$ et $s_k$ la somme des votes symétrisés des $k$ lectures terminées, chaque vote étant compris entre zéro et un. Le score final à sept appartient à :

$$f_{\min}=\frac{s_k/7+d_s}{2},\qquad f_{\max}=\frac{(s_k+7-k)/7+d_s}{2}.$$

Si $f_{\min}\ge0{,}39$, la paire restera positive ; si $f_{\max}<0{,}39$, elle restera négative. Les tests énumèrent les futurs comptes de votes possibles pour vérifier ces cas de frontière. Il faut résoudre toutes les paires d’un document pour économiser sa prochaine lecture documentaire. [Protocole et audit N11](../SOURCES.md#annexes-locales).

La majorité des essais n’entraîne pas les poids des modèles. **J3-B et J4 apprennent toutefois des calibrateurs supervisés**, et les choix de seuils R2/J2 utilisent des labels de développement. Les démonstrations gold sont également une information supervisée dans le contexte. Les expressions « sans fine-tuning », « sans calibrateur appris » et « sans aucun label » décrivent donc des propriétés différentes.

## 3. Fiches techniques de tous les nœuds

Les numéros ci-dessous sont un index de lecture, pas une chronologie stricte des timestamps de soumission. L’inventaire inclut le brouillon J1 archivé et les préparations sans run ; cela explique pourquoi les 64 nœuds historiques ne correspondent pas à 64 comparaisons scientifiques indépendantes.


### Sommaire des fiches

- [01. Span : validation initiale du modèle de décision](#fiche-01)
- [02. Mercury : remplacement du modèle, protocole inchangé](#fiche-02)
- [03. Mercury Choice : décision à trois classes](#fiche-03)
- [04. Mercury Context : rendre explicites les occurrences ciblées](#fiche-04)
- [05. Mercury sur ESC233 : passage à l'échelle interrompu](#fiche-05)
- [06. Prototype d1 DEV25+D15 : préparation non lancée](#fiche-06)
- [07. R0 : baseline d1-ultra effective](#fiche-07)
- [08. Luna DEV25+D15 : extraction documentaire répétée](#fiche-08)
- [09. R1c : une requête par paire non ordonnée](#fiche-09)
- [10. R1a : aligner la question sur les relations ESC](#fiche-10)
- [11. R1b : baliser A et B dans le texte](#fiche-11)
- [12. R1e : démonstrations récupérées pour d1](#fiche-12)
- [13. R1d : plusieurs questions dans le même appel](#fiche-13)
- [14. R2a : symétrisation et seuil appris hors document](#fiche-14)
- [15. R2b : coréférence prédite puis propagation](#fiche-15)
- [16. Confirmation d1 : symétrie et seuil 0,65 gelés](#fiche-16)
- [17. Confirmation Luna : contrôle documentaire sur CONF40](#fiche-17)
- [18. R3-T0 : fusion Luna × d1](#fiche-18)
- [19. R3-T1 : rationales dans les démonstrations](#fiche-19)
- [20. R3-T2 : groupes de coréférence explicitement produits par Luna](#fiche-20)
- [21. R4-A : préciser la sémantique des événements](#fiche-21)
- [22. R4-B : récupération structurelle et négatifs explicites](#fiche-22)
- [23. H2-D : adaptation du harness de d1 sans labels](#fiche-23)
- [24. H2-L : adaptation du harness de Luna](#fiche-24)
- [25. H2-F : adaptation du harness de la fusion](#fiche-25)
- [26. J1 brouillon : accord de deux procédures DeepSeek](#fiche-26)
- [27. J2 : nouveau juge dans la bande incertaine](#fiche-27)
- [28. J1R : coréférence par accord Luna × DeepSeek](#fiche-28)
- [29. J3-A : confirmation de la moyenne calibrée J2](#fiche-29)
- [30. J3-B : calibration logistique régularisée](#fiche-30)
- [31. J4-A : fusion logistique supervisée des deux scores initiaux](#fiche-31)
- [32. J4-B : ajouter des caractéristiques structurelles à la calibration](#fiche-32)
- [33. K0 : demander directement chaque relation à Luna](#fiche-33)
- [34. KA : cascade de types puis élimination](#fiche-34)
- [35. KB : proposer les causes d'une mention cible](#fiche-35)
- [36. KC : décider entre événements puis recopier aux mentions](#fiche-36)
- [37. KD : combiner les signaux hétérogènes du round K](#fiche-37)
- [38. K1 : jugement Luna par paire sur DEV10](#fiche-38)
- [39. H1 : Haiku documentaire sur ESC et CTB complets](#fiche-39)
- [40. Qwen documentaire initial : baseline locale sans d1](#fiche-40)
- [41. Alias virtuels A/R : test d'ancrage par identifiants auxiliaires](#fiche-41)
- [42. M0 hors ligne : planifier la couverture des paires](#fiche-42)
- [43. P1 v1.1 : extraire des preuves sources avant transport local](#fiche-43)
- [44. M0 live : O/P/M/MP sur deux documents](#fiche-44)
- [45. Placebos V0 : faisabilité d'éditions contrôlées automatiques](#fiche-45)
- [46. CTB attribution : concurrence entre liaisons argumentales](#fiche-46)
- [47. M1 brouillon : nœud préparatoire non implémenté](#fiche-47)
- [48. M1 : diversité complémentaire contre lectures ordinaires](#fiche-48)
- [49. Qwen xhigh, température zéro : reprise de baseline](#fiche-49)
- [50. N1 : filtrer les ajouts par preuve et ancrage](#fiche-50)
- [51. N2 : transport local v2, même pool et vérificateur commun](#fiche-51)
- [52. N5 : ablation exhaustive 1/3/7 lectures sur caches](#fiche-52)
- [53. N3 : paquet de démonstrations contrastives](#fiche-53)
- [54. N4 : remplacer Luna par DeepSeek documentaire](#fiche-54)
- [55. N6 : question lexicale adaptée à CTB](#fiche-55)
- [56. N3 audit : récupération et reproduction sans génération](#fiche-56)
- [57. N7 : réplication du paquet N3 sur deux calendriers](#fiche-57)
- [58. N8 : trois lectures fraîches réellement incluses dans sept](#fiche-58)
- [59. N9 : factoriel récupération × négatifs](#fiche-59)
- [60. N10 : six lectures contre tous les triples de N7](#fiche-60)
- [61. N11 : arrêt anticipé exact des sept lectures](#fiche-61)
- [62. N12 : changer la vue des occurrences](#fiche-62)
- [63. N13 : représentations avec Qwen local, versions ESC puis CTB50](#fiche-63)
- [64. Qwen CTB seul : terminer la baseline avant ESC](#fiche-64)



- [65. N14 — consigne nettoyée et exécution bornée, factoriel](#fiche-65)
- [66. S1 — directions hint/demo et doses à la couche 32](#fiche-66)
- [67. N15 — contrat de génération court : medium ou sans réflexion](#fiche-67)
- [68. D0/E6 — MBR exact sur les caches N14](#fiche-68)
- [69. D0 — union des lectures brutes N14](#fiche-69)
- [70. S2-A — augmenter la dose demo](#fiche-70)
- [71. S2-B — direction opposée et contrôle aléatoire](#fiche-71)
- [72. S2-C — transfert de l’intervention à la couche 40](#fiche-72)
- [73. D0/E6 — MBR exact sur N15 complet](#fiche-73)
- [74. E1 — décodage, boucles et fermeture à 8k](#fiche-74)
- [75. E2 — arrêt anticipé par continuations en accord](#fiche-75)
- [76. E3 — retirer une composante d’hésitation](#fiche-76)
- [77. E4 — sonde avant et après réflexion](#fiche-77)
- [78. E5 — éditer des têtes associées à l’hésitation](#fiche-78)
- [79. R1-Qwen — cinq lectures échantillonnées et sonde](#fiche-79)
- [80. R2-Qwen — vote documentaire sur ESC40](#fiche-80)
- [81. R3-Qwen — élargir la consigne aux buts et à l’habilitation](#fiche-81)
- [82. N1-Qwen — diversité des démonstrations](#fiche-82)
- [83. N2-Qwen — relecture des propositions R1](#fiche-83)
- [84. P1-Qwen — même relecture sur N1](#fiche-84)
- [85. P3 — augmenter la réflexion CTB à 20k](#fiche-85)
- [86. P2 — réflexion ESC à 24k](#fiche-86)
- [87. P4 — vérifier les longues lectures P3](#fiche-87)
- [88. V1 — vérificateur dense par probabilités natives](#fiche-88)
- [89. V2 — vérificateur avec exemples positifs et négatifs récupérés](#fiche-89)
- [90. C1 — contexte et réflexion par blocs de paires](#fiche-90)
- [91. C2 — fiches d’événements et réflexion 4k](#fiche-91)
- [92. C3 natif — arbitrage direct ou avec preuve](#fiche-92)
- [93. C3 API — direct contre preuve textuelle](#fiche-93)
- [94. AP0 — replay C3 et gel de campagne](#fiche-94)
- [95. AP1 direct — témoin à budget de sortie égal](#fiche-95)
- [96. AP1 décomposé — faits, mécanisme et explication concurrente](#fiche-96)
- [97. AP2 — récupérer les liens rejetés unanimement](#fiche-97)
- [98. AP3 — stabilité et gel de la réparation](#fiche-98)
- [99. AP4 — extension de la politique figée](#fiche-99)
- [100. QG1 — sélection d’un graphe anonyme](#fiche-100)
- [101. QG2 — édition de graphes lien par lien](#fiche-101)
- [102. VC1 — valider filtrage et réparation sur CTB47](#fiche-102)
- [103. VC2 — étendre la réparation sur ESC40](#fiche-103)
- [104. VC3 — transfert des politiques sur MAVEN12](#fiche-104)
- [105. MV1 — votes de familles, première tentative](#fiche-105)
- [106. FULL CTB183 — plan de corpus complet](#fiche-106)
- [107. FULL ESC233 — plan de combinaison complète](#fiche-107)
- [108. FULL MAVEN710 — plan de transfert complet](#fiche-108)
- [109. MV2 — replay complet des votes de familles](#fiche-109)
- [110. W1 — modification de poids : conclusion et rappel](#fiche-110)
- [111. W2 — confirmation réservée de conclusion 0,10](#fiche-111)
- [112. W3 — diversité et majorité stricte sur caches W1/W2](#fiche-112)
- [113. W4 — steering, votes et vérification sur CTB47](#fiche-113)
- [114. W5 — pilote ESC réduit à quatre documents](#fiche-114)
- [115. W6 — pilote MAVEN réduit et format invalide](#fiche-115)
- [116. W7 — extension ESC50](#fiche-116)
- [117. W8 — extension MAVEN50 avec schéma de paire contraint](#fiche-117)

### A. Intégration, modèles de décision et premières validations


<a id="fiche-01"></a>

#### 01. Span : validation initiale du modèle de décision

Expérience `694058c6-1973-46a5-8bc5-3579eada24ad`.

**Implémentation et intervention.** Construire un état contenant le document entier et les deux mentions exactes : identifiants, surfaces, positions de tokens, offsets et phrase. Envoyer une question Noul oui/non ; classer causal si la probabilité vaut au moins 0,5. Le gold relationnel n'entre pas dans cet état.

**Protocole d’évaluation.** Un seul document ESC dev, 41_9ecbplus.xml ; 156 candidats ordonnés et huit positifs. Une décision par candidat, sans vote à sept lectures. C'est un test d'intégration en conditions réelles, pas une validation statistique.

**Tests et contrôles.** 156 appels réussis ; sauvegarde des probabilités et réponses ; rejeu hors ligne du score identique. Le modèle servi est identifié dans le résumé. Les erreurs de direction sont séparées de la mesure binaire.

**Résultat et portée.** F1 binaire 0 : aucune relation positive prédite, donc huit faux négatifs. Le chemin de données fonctionne ; le modèle n'a pas réussi la détection positive sur ce document.

Exécutions de référence : [run aa75d46a](../results/inventaire-executions.csv) (`aa75d46a-5794-4b93-ab23-126075a69584`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/93bb77ff1fd1c748808263487d6240aa8c32a696/pipelines/jev/state_builder.py) [pipelines/jev/pipeline.py](https://github.com/Fireman4740/workere/blob/93bb77ff1fd1c748808263487d6240aa8c32a696/pipelines/jev/pipeline.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 01.


<a id="fiche-02"></a>

#### 02. Mercury : remplacement du modèle, protocole inchangé

Expérience `87cddd6f-7910-4913-8e2f-4fe3441ea33f`.

**Implémentation et intervention.** Remplacer Span par Mercury en gardant le document, les 156 candidats, la question Noul, la représentation et le seuil 0,5. Le changement testé est le modèle de décision.

**Protocole d’évaluation.** Comparaison sur le même document et les mêmes huit positifs que Span. Précision, rappel, F1 binaire et sous-périmètres intra/inter-phrases.

**Tests et contrôles.** 156 décisions livrées ; version servie enregistrée ; prédictions et métriques sauvegardées pour le rejeu. Aucun seuil choisi après observation du score.

**Résultat et portée.** Précision 43,75 %, rappel 87,5 %, F1 58,33. Meilleur smoke local, sans preuve de généralisation à ESC233.

Exécutions de référence : [run 654d1602](../results/inventaire-executions.csv) (`654d1602-1b6e-47f3-a88e-aa992adbe7b4`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/6e48b7e25fad0c3bfea0d47192a62ed96bf1b538/pipelines/jev/state_builder.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 02.


<a id="fiche-03"></a>

#### 03. Mercury Choice : décision à trois classes

Expérience `2fbe084a-f105-4726-be67-815b8cf98c77`.

**Implémentation et intervention.** Remplacer la question Noul par un choix CAUSES, CAUSED_BY ou NO_RELATION. Le modèle peut donc exprimer la direction. Le contexte et les candidats restent ceux du smoke Mercury.

**Protocole d’évaluation.** Même document et seuil déclaré ; le F1 binaire regroupe les deux classes causales. Les métriques directionnelles sont conservées séparément.

**Tests et contrôles.** 156 appels ; réponses/probabilités archivées ; rejeu exact. Vérification que le résultat comparé est bien le binaire, et non une autre colonne de métriques.

**Résultat et portée.** TP=8, FP=24, FN=0 : F1 40,00. Le rappel augmente, mais les faux positifs font baisser le F1 par rapport au Noul Mercury.

Exécutions de référence : [run 5d1c4b87](../results/inventaire-executions.csv) (`5d1c4b87-783d-4f15-a87d-5bce934f83f2`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/f9d6dc3eea69aa48f0ebd9cf207e73a70012c26b/pipelines/jev/state_builder.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 03.


<a id="fiche-04"></a>

#### 04. Mercury Context : rendre explicites les occurrences ciblées

Expérience `293c814f-d7a3-477f-80e2-dae63e9774b1`.

**Implémentation et intervention.** Conserver Noul et ajouter au document une copie où les tokens A/B sont balisés, puis les phrases contenant A et B. Les événements restent les mêmes ; l'intervention porte sur leur présentation.

**Protocole d’évaluation.** Même smoke à 156 candidats ; modèle et seuil 0,5 inchangés. Comparaison appariée à Mercury Noul.

**Tests et contrôles.** 156 appels ; sauvegarde et rejeu exact des sorties. Le constructeur d'état vérifie les styles reconnus et conserve les positions des mentions.

**Résultat et portée.** TP=8, FP=18, FN=0 : F1 47,06. L'explicitation ne surpasse pas le Noul initial sur ce cas unique.

Exécutions de référence : [run e8a58cf1](../results/inventaire-executions.csv) (`e8a58cf1-30a2-4562-b1a2-1047d16b8c94`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/bc98c9f8da6357fbb350983ffd1da61a269f1798/pipelines/jev/state_builder.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 04.


<a id="fiche-05"></a>

#### 05. Mercury sur ESC233 : passage à l'échelle interrompu

Expérience `3afb71c4-e110-4a31-a9eb-899fc1419b68`.

**Implémentation et intervention.** Appliquer la même décision Noul à tous les candidats ordonnés d'ESC train : 233 documents, 122 130 paires. Version de données, sélection et empreinte de suite figées ; modèle gratuit uniquement, cadence 20 requêtes/minute et cache partagé persistant.

**Protocole d’évaluation.** Objectif initial : score corpus complet et comparaison historique. Après HTTP429 puis arrêt demandé à une frontière documentaire, seules les unités entièrement terminées sont évaluables. Le diagnostic initial de 13 documents et le paquet final D15 sont deux instantanés distincts.

**Tests et contrôles.** Validation des 4 253 réponses en cache après le premier échec ; respect de Retry-After et cooldown partagé ; test de préflight sans débordement du contexte. Les documents partiels sont exclus des calculs partiels, sans transformer les manquants en négatifs. Les anciennes prédictions DeepSeek n'ont pas été retrouvées.

**Résultat et portée.** Aucun résultat ESC233 complet. F1 33,00 sur l'instantané 13 documents ; 32,39 sur D15 final, diagnostique seulement. Ni l'un ni l'autre n'est comparable directement au chiffre historique sur 233 documents.

Exécutions de référence : [run 586bbafb](../results/inventaire-executions.csv) (`586bbafb-473f-4e7d-94b2-e3eb1b381137`) [run dfa23986](../results/inventaire-executions.csv) (`dfa23986-77a1-4a69-b0ea-b3701fc586bd`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/79169a769738b808ac83aa577119474c934798f9/pipelines/jev/state_builder.py) [pipelines/jev/client.py](https://github.com/Fireman4740/workere/blob/79169a769738b808ac83aa577119474c934798f9/pipelines/jev/client.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [partial-results-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 05.


<a id="fiche-06"></a>

#### 06. Prototype d1 DEV25+D15 : préparation non lancée

Expérience `2893566d-a69d-4a26-bd2b-54d26c66073d`.

**Implémentation et intervention.** Préparer le remplacement de Mercury par liquid/d1-ultra, avec appels parallèles par paire et seuil Noul 0,5. La cible est DEV25 et le diagnostic D15.

**Protocole d’évaluation.** Aucune exécution attachée à ce nœud. La réalisation effective se trouve dans R0, fiche suivante. Les deux nœuds ne sont pas deux comparaisons indépendantes.

**Tests et contrôles.** État et notes du nœud relus ; absence de run dans l'inventaire. Une branche préparée ne démontre ni qualité mesurée ni réussite du déploiement.

**Résultat et portée.** Préparation remplacée par R0 ; aucun score propre.

Aucune exécution dans l’inventaire pour ce nœud.

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/79169a769738b808ac83aa577119474c934798f9/pipelines/jev/state_builder.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 06.


<a id="fiche-07"></a>

#### 07. R0 : baseline d1-ultra effective

Expérience `9d81e9ba-b528-4f56-89de-049034bb3e07`.

**Implémentation et intervention.** Une probabilité causale par paire ordonnée, document entier et mentions exactes ; seuil 0,5. Modèle servi liquid/d1-ultra-20261015. Ajout de parallélisme, cache et registre atomique de budget ; ces mécanismes d'exécution ne constituent pas des variantes scientifiques.

**Protocole d’évaluation.** DEV25 : 19 028 candidats et 1 086 positifs. Sous-échantillon DEV-SCREEN : dix documents, 6 894 candidats, 436 positifs. D15 sert au diagnostic et pas à la sélection. AP pour le classement, micro-F1 binaire pour les décisions.

**Tests et contrôles.** Correction des verrous SQLite par retry de transaction locale, sans rappel modèle dans cette boucle ; 17 tests rapportés dans la reprise. Données et question maintenues, sorties persistées, comparaison des variantes sur documents appariés.

**Résultat et portée.** DEV25 : F1 39,26, AP 39,87 ; DEV-SCREEN : F1 42,80. C'est le contrôle de R1 et R2. Les tentatives antérieures incomplètes ne sont pas des F1 supplémentaires.

Exécutions de référence : [run d0e4ab4a](../results/inventaire-executions.csv) (`d0e4ab4a-1c61-4c15-871e-fec594d6bcdd`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/08d8c6bbbb885f73cedb9035cd66ecb937fdc0b4/pipelines/jev/state_builder.py) [pipelines/jev/spend.py](https://github.com/Fireman4740/workere/blob/08d8c6bbbb885f73cedb9035cd66ecb937fdc0b4/pipelines/jev/spend.py) [pipelines/jev/client.py](https://github.com/Fireman4740/workere/blob/08d8c6bbbb885f73cedb9035cd66ecb937fdc0b4/pipelines/jev/client.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 07.


<a id="fiche-08"></a>

#### 08. Luna DEV25+D15 : extraction documentaire répétée

Expérience `06273e68-8c84-4758-8aa9-e7a2e23fa143`.

**Implémentation et intervention.** Faire lire chaque document balisé à Luna batch en raisonnement high. Sept générations indépendantes, chacune avec un exemple gold d'un autre topic. La sortie est une liste de paires/labels ; les réponses sont agrégées par le vote historique. Ce système diffère de d1 par le modèle, la granularité et les démonstrations.

**Protocole d’évaluation.** 25 documents DEV et 15 D15, soit 280 requêtes documentaires. Contrôle principal DEV25 apparié à R0 ; D15 reste diagnostic. Le vote historique par labels et le F1 binaire ne doivent pas être confondus avec une probabilité calibrée.

**Tests et contrôles.** Correction du transport vers une véritable API batch après HTTP404 ; reçus durables, reprise sans double soumission, cache distinct par document. 40/40 sous-évaluations terminées ; reconstruction hors ligne et contrôle des candidats.

**Résultat et portée.** DEV25 : F1 50,85, écart à d1 +11,59 points [7,51 ; 15,65]. AP 33,49, inférieure à celle de d1. Le gain n'isole pas l'effet propre de sept lectures ou de l'effort high.

Exécutions de référence : [run 7c9548f7](../results/inventaire-executions.csv) (`7c9548f7-a36a-41f8-8a1b-9d31748b2810`)

Code ciblé : [pipelines/jev/llm.py](https://github.com/Fireman4740/workere/blob/0cfb03f72e2a9c077829cd6937c0e9d82d1155e3/pipelines/jev/llm.py) [pipelines/jev/batch.py](https://github.com/Fireman4740/workere/blob/0cfb03f72e2a9c077829cd6937c0e9d82d1155e3/pipelines/jev/batch.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 08.


<a id="fiche-09"></a>

#### 09. R1c : une requête par paire non ordonnée

Expérience `f17a10ce-51ad-440d-bddf-67fa9abb8ff2`.

**Implémentation et intervention.** Choisir l'ordre canonique des deux mentions, demander une seule probabilité et la recopier aux deux candidats historiques orientés. Cela divise exactement par deux le nombre logique d'appels d1.

**Protocole d’évaluation.** DEV-SCREEN inchangé ; seuil 0,5 ; comparaison appariée R0. L'objectif d'efficacité inclut une limite de dégradation du F1, et le bilan de sélection R1 examine aussi le classement AP.

**Tests et contrôles.** Contrôle du remappage vers les deux orientations et de l'identité des paires ; cache des reprises ; bootstrap par document, pas par orientation. Les intervalles couvrent des pertes dépassant un point : la proximité ponctuelle ne démontre pas la préservation de qualité.

**Résultat et portée.** F1 41,93 contre 42,80 pour R0 : −0,88 point ponctuel ; AP en baisse. Aucun finaliste retenu dans la campagne R1.

Exécutions de référence : [run 5bf3ecaa](../results/inventaire-executions.csv) (`5bf3ecaa-c821-43e5-86b9-c9b735da0c0e`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/236694d268f566116cc769493eac30243ad6ae51/pipelines/jev/state_builder.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 09.


<a id="fiche-10"></a>

#### 10. R1a : aligner la question sur les relations ESC

Expérience `2750ffff-41c6-427e-a1ca-b177049d76fc`.

**Implémentation et intervention.** Élargir la définition de causalité dans la question : PLOT_LINK, facilitation, prévention, conséquence narrative soutenue et lien de sous-événement explicite. Conserver modèle, contexte et seuil de R0.

**Protocole d’évaluation.** Dix documents DEV-SCREEN ; une question par paire ordonnée. Sélection sur développement, avec AP comme critère principal de promotion, puis F1 et bootstrap apparié.

**Tests et contrôles.** Reprise des mêmes requêtes depuis le cache après réparation budgétaire ; probabilités archivées. Les ensembles D15 et CONF40 ne servent pas au choix de cette variante.

**Résultat et portée.** F1 43,30, soit environ +0,50 point ; gain AP incertain. Aucun critère de promotion robuste franchi.

Exécutions de référence : [run 6841832e](../results/inventaire-executions.csv) (`6841832e-ac6f-41fb-8215-730641ce52de`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/0ae5f91cc2d0a5d81d3948b53abf7588c254eaf5/pipelines/jev/state_builder.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 10.


<a id="fiche-11"></a>

#### 11. R1b : baliser A et B dans le texte

Expérience `f0710474-0d2e-42fc-9a96-d7db645dc0ba`.

**Implémentation et intervention.** Ajouter des balises autour des tokens exacts des deux occurrences visées dans le document de d1 ; préserver la question Noul et les candidats. L'hypothèse est un meilleur rattachement de la question aux mentions.

**Protocole d’évaluation.** DEV-SCREEN commun, seuil 0,5, métriques AP et F1 ; comparaison au R0 du même périmètre.

**Tests et contrôles.** Vérification des positions et de la conservation du texte/identité des candidats, sauvegarde des scores et bootstrap apparié. L'amélioration d'un score de décision ne remplace pas le critère de classement annoncé.

**Résultat et portée.** F1 44,11, environ +1,31 point ; AP en baisse. L'intervalle du gain F1 inclut zéro : variante non promue.

Exécutions de référence : [run 6a71eda4](../results/inventaire-executions.csv) (`6a71eda4-54ca-424f-9510-ba671d9fe740`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/ec7de232d726156ed12b63f9c7f6512b129712f1/pipelines/jev/state_builder.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 11.


<a id="fiche-12"></a>

#### 12. R1e : démonstrations récupérées pour d1

Expérience `7bd4db91-8a31-47ae-bc87-db4d99a57bff`.

**Implémentation et intervention.** Ajouter quatre exemples récupérés par similarité : deux positifs et deux négatifs. Exclure le topic interrogé et les ensembles protégés de la banque de démonstrations. Les labels de ces exemples sont des annotations d'apprentissage autorisées, pas ceux de la cible.

**Protocole d’évaluation.** Même DEV-SCREEN, même d1 et seuil 0,5. Comparaison AP/F1 au contrôle ; aucune sélection sur D15.

**Tests et contrôles.** Vérification de l'éligibilité de la banque et des exclusions ; cache/reprises conservés ; bootstrap apparié. Le coût du run terminal exclut les réponses déjà payées et relues dans le cache.

**Résultat et portée.** F1 43,26, environ +0,46 point ; gains de classement et F1 incertains. Aucun élargissement DEV-FULL de ce facteur.

Exécutions de référence : [run 41307f1d](../results/inventaire-executions.csv) (`41307f1d-6722-48be-903e-4aad11fa6ba2`)

Code ciblé : [pipelines/jev/demonstrations.py](https://github.com/Fireman4740/workere/blob/08922d583edd9d1edb4dc6c95505b7db46d318a6/pipelines/jev/demonstrations.py) [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/08922d583edd9d1edb4dc6c95505b7db46d318a6/pipelines/jev/state_builder.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 12.


<a id="fiche-13"></a>

#### 13. R1d : plusieurs questions dans le même appel

Expérience `15d2db3f-307f-477f-aa76-2fe008b320c9`.

**Implémentation et intervention.** Demander plusieurs décisions distinctes dans un appel : causalité, identité événementielle et autres propriétés de la paire. Prévoir un contrôle d'invariance de la probabilité causale avant le tri à grande échelle.

**Protocole d’évaluation.** Prétest sur les 156 paires du document connu : question causale seule versus appel multi-questions. L'évaluation DEV-SCREEN est conditionnelle à la réussite de ce prétest.

**Tests et contrôles.** Les sorties des cinq questions sont distinctes, mais la probabilité causale change jusqu'à 0,21405. Le contrôle d'invariance échoue ; aucun score DEV-SCREEN ni calibrateur dérivé n'est produit. Une répétition dédiée serait nécessaire pour séparer interaction et stochasticité.

**Résultat et portée.** Prérequis non satisfait, essai arrêté. C'est un résultat du contrôle préalable, pas un F1 nul sur DEV ni la preuve que toutes les formulations multi-questions échouent.

Exécutions de référence : [run df51a1e6](../results/inventaire-executions.csv) (`df51a1e6-f0a5-4e06-8685-a90d8685539f`) [run 73ae2296](../results/inventaire-executions.csv) (`73ae2296-7d4b-412f-8da4-247a1542752e`)

Code ciblé : [pipelines/jev/state_builder.py](https://github.com/Fireman4740/workere/blob/5156f74ab05e7b536118d001a9b811423ed5348a/pipelines/jev/state_builder.py) [pipelines/jev/preflight.py](https://github.com/Fireman4740/workere/blob/5156f74ab05e7b536118d001a9b811423ed5348a/pipelines/jev/preflight.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 13.


<a id="fiche-14"></a>

#### 14. R2a : symétrisation et seuil appris hors document

Expérience `4f2e2f6e-82c1-4d88-88d3-ba1047814c5e`.

**Implémentation et intervention.** Réutiliser les probabilités R0. Pour chaque paire, moyenner d(a,b) et d(b,a). Comparer le seuil fixe 0,5 à un seuil choisi en leave-one-document-out : aucun label du document évalué pour choisir son seuil. Ajuster ensuite 0,65 sur DEV complet pour la future confirmation.

**Protocole d’évaluation.** DEV25 ; mêmes sorties brutes pour tous les post-traitements, donc zéro nouvel appel. Le réglage est interne au DEV ; les documents partagent néanmoins des topics.

**Tests et contrôles.** Reconstruction déterministe des décisions, contrôle des folds et des seuils. Les IC complémentaires sont post hoc et ne réajustent pas le seuil dans chaque bootstrap. Correction d'un export YAML/date, sans nouvelle inférence.

**Résultat et portée.** F1 : R0 39,26 ; symétrie seule 39,87 ; symétrie+seuil hors document 42,04. Le +2,78 est un gain DEV incertain ; le transfert du seuil est testé dans la fiche 16.

Exécutions de référence : [run 0c4f8f08](../results/inventaire-executions.csv) (`0c4f8f08-d6d6-4cfa-9bc6-015b307ae418`)

Code ciblé : [pipelines/jev/refinement.py](https://github.com/Fireman4740/workere/blob/f358a25f2578e5b6506e3aada8a0729c3ea0543d/pipelines/jev/refinement.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 14.


<a id="fiche-15"></a>

#### 15. R2b : coréférence prédite puis propagation

Expérience `d02eadb6-4a61-4c0c-a9e2-f5ac246a4ac1`.

**Implémentation et intervention.** Poser séparément 9 514 questions d1 « même événement ? » sur les paires non ordonnées. Au seuil 0,5, former les composantes connexes ; agréger au maximum les scores causaux entre chaînes, puis diffuser. Les relations internes à une chaîne sont rejetées. Ajouter un oracle ECB+ pour mesurer le potentiel d'une identité correcte.

**Protocole d’évaluation.** DEV25. Deux évaluations différentes : précision/rappel des liens de coréférence contre ECB+, puis F1 causal sur toutes les paires. Seuil causal fixe ou choisi hors document comme R2a.

**Tests et contrôles.** Questions de coréférence indépendantes de la question causale ; contrôle de fermeture et de propagation ; diagnostic avant/après fermeture. L'oracle utilise le gold de coréférence et n'est pas déployable.

**Résultat et portée.** Précision des liens : 26,00 % directs, 17,98 % après fermeture. F1 causal 29,27 au seuil 0,5 ; 37,70 avec seuil hors document. Oracle+seuil : 60,88. La sur-fusion prédite empêche de réaliser le potentiel oracle.

Exécutions de référence : [run 26ac5081](../results/inventaire-executions.csv) (`26ac5081-c538-431e-994c-b6a7097f7245`)

Code ciblé : [pipelines/jev/coreference.py](https://github.com/Fireman4740/workere/blob/3ba6739bf7b72133ae71c0526636775d6014ac76/pipelines/jev/coreference.py) [pipelines/jev/refinement.py](https://github.com/Fireman4740/workere/blob/3ba6739bf7b72133ae71c0526636775d6014ac76/pipelines/jev/refinement.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 15.


<a id="fiche-16"></a>

#### 16. Confirmation d1 : symétrie et seuil 0,65 gelés

Expérience `38c20615-90ba-436e-bd97-03400935eb8e`.

**Implémentation et intervention.** Geler la règle choisie sur DEV puis collecter une fois les scores d1 du nouveau CONF40. Appliquer aux mêmes scores la baseline 0,5 et la variante symétrique 0,65, sans régler à nouveau le seuil.

**Protocole d’évaluation.** 40 documents, 20 topics, 19 174 candidats ordonnés. Comparaison appariée avec bootstrap documentaire ; ce lot était réservé à cette étape de l'histoire du projet.

**Tests et contrôles.** Empreinte du gel, complétude des candidats et reproduction officielle hors ligne. Les deux systèmes réutilisent les mêmes probabilités : la différence vient du post-traitement, pas d'une nouvelle génération.

**Résultat et portée.** Baseline F1 34,79 ; variante 28,29 ; −6,50 points [−9,94 ; −2,95]. Le réglage DEV ne se transfère pas. La variante retire 1 010 FP mais aussi 309 TP ; elle n'est pas promue.

Exécutions de référence : [run 945b655d](../results/inventaire-executions.csv) (`945b655d-4595-4726-8d72-d1714bb2f5e6`)

Code ciblé : [pipelines/jev/confirmation.py](https://github.com/Fireman4740/workere/blob/81a17b23726ee6754bd9602786dcb8787087d541/pipelines/jev/confirmation.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [final-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 16.


<a id="fiche-17"></a>

#### 17. Confirmation Luna : contrôle documentaire sur CONF40

Expérience `18d28e3b-98a1-43d7-8f82-b1386f7b4fbb`.

**Implémentation et intervention.** Conserver Luna high, sept lectures et une démonstration hors topic par lecture. Aucune adaptation sur les labels CONF40. Comparer au d1 original et au post-traitement d1 gelé.

**Protocole d’évaluation.** Les mêmes 40 documents que la confirmation d1 ; 280 réponses Luna. Micro-F1 global, intra/inter et différences appariées.

**Tests et contrôles.** 40 lots complets ; reconstruction des votes et du score officiel. Les contrastes Luna sont secondaires, sans correction de multiplicité dans ce premier bilan.

**Résultat et portée.** F1 36,45 ; +1,65 point [−3,35 ; 6,33] face au d1 original. Le gain net observé sur DEV n'est pas confirmé avec la même ampleur. L'IC contenant zéro ne démontre pas l'équivalence.

Exécutions de référence : [run ab42114f](../results/inventaire-executions.csv) (`ab42114f-3687-4493-b89a-eb619790fa7b`)

Code ciblé : [pipelines/jev/confirm_luna.py](https://github.com/Fireman4740/workere/blob/3470a896384c40faba0aa6f0ad92c692963cb547/pipelines/jev/confirm_luna.py) [pipelines/jev/llm.py](https://github.com/Fireman4740/workere/blob/3470a896384c40faba0aa6f0ad92c692963cb547/pipelines/jev/llm.py)

Sources détaillées : [rapport-complet.md](../SOURCES.md#annexes-locales) · [report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 17.


### B. Fusion documentaire et consignes


<a id="fiche-18"></a>

#### 18. R3-T0 : fusion Luna × d1

Expérience `514110b5-14e8-45f7-8347-2feef53129fa`.

**Implémentation et intervention.** Moyenner les deux orientations du vote Luna et les deux scores d1 : f=(v_ab+v_ba+d_ab+d_ba)/4. Positif si f≥0,39. La famille de règles et le seuil sont choisis sur POOL80, réunion de DEV25+D15+CONF40 (22 topics), en laissant un topic à l'écart, puis figés avant TEST-S40. D15 et CONF40 sont donc réutilisés à ce stade, après leur rôle réservé dans le plan précédent.

**Protocole d’évaluation.** 40 documents TEST-S40, 17 958 candidats ordonnés ; deux documents par topic, exclusions de D15, CONF40 et banque de démonstrations. Comparer fusion, Luna seule et d1 seul sur les mêmes documents.

**Tests et contrôles.** Recalcul hors ligne depuis sorties et contrôles ; règle de fusion sans refit sur le test. Bootstrap apparié par document ; lecture des résultats de tous les documents, sans remplacer le micro-F1 par une moyenne documentaire.

**Résultat et portée.** Fusion 48,64 ; d1 38,59 ; Luna 46,02. Écart +10,05 [6,66 ; 13,65] face à d1, mais +2,62 [−1,37 ; 7,69] face à Luna : la supériorité sur les deux contrôles n'est pas établie.

Exécutions de référence : [run 64e91e64](../results/inventaire-executions.csv) (`64e91e64-d6e8-4a51-94fb-eab40c38eec7`)

Code ciblé : [experiments/jev/round3/fusion-rule.json](https://github.com/Fireman4740/workere/blob/d39d3349fe90c2396307d189b32d46ee7f6dabe5/experiments/jev/round3/fusion-rule.json) [experiments/jev/round3/materialize.py](https://github.com/Fireman4740/workere/blob/d39d3349fe90c2396307d189b32d46ee7f6dabe5/experiments/jev/round3/materialize.py)

Sources détaillées : [round3-results-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 18.


<a id="fiche-19"></a>

#### 19. R3-T1 : rationales dans les démonstrations

Expérience `8d3a10af-5925-4454-b53a-c9aea79c0424`.

**Implémentation et intervention.** Garder les mêmes documents de démonstration que T0, mais ajouter à leur réponse gold une explication produite une fois par Luna. La cible ne reçoit pas son gold ; l'intervention est le contenu des démonstrations.

**Protocole d’évaluation.** TEST-S40 apparié à Luna T0 ; sept lectures high et contrôle gold-answer-only. Comparaison de la majorité binaire et du vote historique par labels.

**Tests et contrôles.** 40/40 documents complets ; rapprochement des calendriers de démonstrations. Les deux règles de vote diffèrent légèrement et sont rapportées distinctement.

**Résultat et portée.** Majorité binaire : F1 44,34 contre 46,02, soit −1,67 [−3,85 ; 0,85]. Vote historique par labels : 44,27. Aucun gain confirmé des rationales.

Exécutions de référence : [run fcef631c](../results/inventaire-executions.csv) (`fcef631c-414b-4c2e-9edf-31779de3f6d7`)

Code ciblé : [pipelines/jev/llm.py](https://github.com/Fireman4740/workere/blob/4c516e147867c15c6d4bb72fd03e57705b4d7f9f/pipelines/jev/llm.py)

Sources détaillées : [round3-results-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 19.


<a id="fiche-20"></a>

#### 20. R3-T2 : groupes de coréférence explicitement produits par Luna

Expérience `8a84f788-7bed-4d77-a7aa-7e69ab431e7d`.

**Implémentation et intervention.** Demander une seule fois par document les groupes de mentions du même événement, sans causalité. Propager ensuite au maximum les scores causaux entre groupes, sans suppression interne et sans fermeture additionnelle au-delà des groupes fournis.

**Protocole d’évaluation.** Gate intrinsèque fixé à 60 % de précision des liens de coréférence. Le run vise 120 documents ; 116 sont complets, dont tous ceux de TEST-S40, D15 et CONF40. DEV25 reste incomplet.

**Tests et contrôles.** Séparer échec global du processus et complétude des sous-ensembles. Comparer les liens aux chaînes ECB+ ; les scores causaux après échec du gate sont exploratoires, pas une promotion.

**Résultat et portée.** Précision TEST-S40 54,86 %, sous le gate. Fusion+groupes F1 48,88, +0,24 [−1,25 ; 1,58] en diagnostic. Aucun avantage validé ; pas de métrique DEV25 complète inventée.

Exécutions de référence : [run 7a6bd979](../results/inventaire-executions.csv) (`7a6bd979-4a1a-4d07-9c28-ae0d488a92b4`)

Code ciblé : [pipelines/jev/coref.py](https://github.com/Fireman4740/workere/blob/968ae69028692f590b0f7fb1544c9ae4b75a8c00/pipelines/jev/coref.py) [experiments/jev/round3/materialize_t2.py](https://github.com/Fireman4740/workere/blob/968ae69028692f590b0f7fb1544c9ae4b75a8c00/experiments/jev/round3/materialize_t2.py)

Sources détaillées : [round3-results-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 20.


<a id="fiche-21"></a>

#### 21. R4-A : préciser la sémantique des événements

Expérience `c5adf0b0-2e62-43fd-912d-9bbb9cd0d0a7`.

**Implémentation et intervention.** Modifier seulement les consignes Luna : identité stricte, distinction phases/sous-événements, motivation, réaction, facilitation, prévention et cohérence des participants. Maintenir démonstrations, sept lectures et fusion 0,39.

**Protocole d’évaluation.** DEV44, 44 documents de 22 topics, 21 832 candidats orientés ; contrôle de fusion 47,323. Développement déjà consommé, pas confirmation.

**Tests et contrôles.** Sélection de documents sans labels, d1 en cache ; résultat ROUND4_RESULT. Bootstrap documentaire et par topic ; IC 97,5 % également conservés pour les deux comparaisons R4 principales.

**Résultat et portée.** F1 44,318, −3,006 points. L'IC documentaire à 95 % [−7,157 ; 0,298] inclut zéro ; aucune amélioration à retenir.

Exécutions de référence : [run ca86b400](../results/inventaire-executions.csv) (`ca86b400-acac-407c-b440-4bdda95463f5`)

Code ciblé : [pipelines/jev/round4_eval.py](https://github.com/Fireman4740/workere/blob/e1c9baff072de16d8fb30ab261636070beb2f1b0/pipelines/jev/round4_eval.py) [prompts/esl_standard_eci_event_semantics.yaml](https://github.com/Fireman4740/workere/blob/e1c9baff072de16d8fb30ab261636070beb2f1b0/prompts/esl_standard_eci_event_semantics.yaml)

Sources détaillées : [recapitulatif-actualise-2026-10-30.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 21.


<a id="fiche-22"></a>

#### 22. R4-B : récupération structurelle et négatifs explicites

Expérience `aac30ccd-4e1f-4020-9c28-34e93c7a96bb`.

**Implémentation et intervention.** Pour chaque cible, calculer sept caractéristiques publiques : nombre de mentions normalisé, fraction de paires intra-phrase, densité par phrase, répétition des surfaces et densités de mots causaux/temporels/modaux. Classer les démonstrations par cosinus, hors topic. Ajouter jusqu'à quatre négatifs gold des exemples, privilégiant même occurrence, voisin causal commun et même phrase.

**Protocole d’évaluation.** DEV44 identique à R4-A ; sept lectures, banque de 20 démonstrations et fusion 0,39. L'intervention combine sélection des exemples et négatifs : leurs effets ne sont pas séparés ici.

**Tests et contrôles.** Code de profile, negative_pairs et select_demonstrations relu ; exclusions de topic, départage déterministe par hash, candidats et contrôles vérifiés. Bootstrap apparié et correction à deux contrastes disponible.

**Résultat et portée.** F1 48,429 contre 47,323 ; +1,106 point, IC95 documentaire [−0,815 ; 3,656]. Signal de développement incertain ; il motive N3/N7/N9 sans être une victoire confirmée.

Exécutions de référence : [run 6ce6a18b](../results/inventaire-executions.csv) (`6ce6a18b-692d-4cc3-86f5-b06701dd1596`)

Code ciblé : [pipelines/jev/contrastive.py](https://github.com/Fireman4740/workere/blob/93e284dc815f6d396bd11aa06c5216f65ab5da4c/pipelines/jev/contrastive.py) [pipelines/jev/round4_eval.py](https://github.com/Fireman4740/workere/blob/93e284dc815f6d396bd11aa06c5216f65ab5da4c/pipelines/jev/round4_eval.py)

Sources détaillées : [recapitulatif-actualise-2026-10-30.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 22.


### C. Adaptation du harness


<a id="fiche-23"></a>

#### 23. H2-D : adaptation du harness de d1 sans labels

Expérience `c5ead119-a20f-4a0f-b7bc-33b4760c8aee`.

**Implémentation et intervention.** Une boucle Propose–Probe–Compose édite instructions et script de post-traitement de l'exécuteur d1. Éditeur/compositeur Luna ; bibliothèque d'expérience par topic ; k=1. Les annotations n'interviennent qu'après adaptation pour mesurer le F1.

**Protocole d’évaluation.** Huit documents DEV44 de quatre topics prévus ; sept disposent de la sortie finale. Comparaison du harness sélectionné au harness de base, bootstrap documentaire sur ces sept documents seulement.

**Tests et contrôles.** Les sorties de base correspondent exactement au cache R3. Traçage des étapes, éditions et fallbacks ; un document manque au stade final. Les échecs 401 et de transport antérieurs ne sont pas des résultats scientifiques.

**Résultat et portée.** Sur sept documents : F1 39,46→45,87, +6,41 [1,44 ; 14,30]. L'éditeur a surtout retrouvé la symétrisation OR déjà connue. Résultat partiel et petit échantillon, pas nouvelle amélioration confirmée de la fusion.

Exécutions de référence : [run 705c9838](../results/inventaire-executions.csv) (`705c9838-0dcf-4875-bc97-dbbe3025937d`)

Sources détaillées : [harness2-pilots-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 23.


<a id="fiche-24"></a>

#### 24. H2-L : adaptation du harness de Luna

Expérience `e8b797ec-3933-4f23-92b3-3faf9c3a5dcc`.

**Implémentation et intervention.** Même mécanisme, avec Luna documentaire sept lectures comme exécuteur. Les leviers sont le prompt, les règles, les skills/mémoires et un post-traitement des votes. Sélection sans accès aux labels de la cible.

**Protocole d’évaluation.** Huit documents communs au pilote H2-D ; comparer base et final sélectionné sur les huit complets.

**Tests et contrôles.** Base identique au cache R3 ; journaux des étapes INV/DOM/AGG et décisions KEEP_BASE. L'adaptation et l'évaluation sont deux étapes distinctes ; aucune sélection du meilleur harness selon le F1 cible.

**Résultat et portée.** F1 43,40→43,83, +0,43 [−5,79 ; 10,10]. Les modifications de domaine augmentent parfois la prudence et réduisent le rappel. Gain non établi.

Exécutions de référence : [run 6de8dcd9](../results/inventaire-executions.csv) (`6de8dcd9-212c-47e3-a408-a296bcdbe675`)

Sources détaillées : [harness2-pilots-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 24.


<a id="fiche-25"></a>

#### 25. H2-F : adaptation du harness de la fusion

Expérience `26772f64-c118-453a-b5fc-1fbbf3d9c14d`.

**Implémentation et intervention.** Donner à l'éditeur la fusion complète et son script de score, avec récursion k=2 et sélection sans labels entre deux finales. Router éditeurs, compositeur, sélecteur et réflecteur vers des modèles distincts ; conserver les fallbacks vers la base en cas d'étape invalide.

**Protocole d’évaluation.** Huit documents du même pilote ; base fusion 45,56. Rapporter aussi le sous-ensemble de sept documents H2-D sans le substituer à l'échantillon principal.

**Tests et contrôles.** Base reproduite exactement depuis les caches ; validation des blocs de réponse des agents. Trois documents perdent la sélection à cause des limites de clé/budget ; fallbacks et taux de sessions acceptées sont conservés.

**Résultat et portée.** F1 final 47,02, +1,46 [−4,23 ; 8,12]. Pilote techniquement incomplet dans ses sélections, sans gain robuste sur la fusion.

Exécutions de référence : [run 718cb169](../results/inventaire-executions.csv) (`718cb169-5d34-4619-bdc4-0ac967936084`)

Sources détaillées : [harness2-pilots-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 25.


### D. Juges et calibrations


<a id="fiche-26"></a>

#### 26. J1 brouillon : accord de deux procédures DeepSeek

Expérience `42d6a6ad-3405-478a-9ce7-c892bae94e8e`.

**Implémentation et intervention.** Projet initial : intersection de trois groupements DeepSeek puis confirmation pairwise par DeepSeek, avec gate de précision 85 %. Ce n'est pas la variante Luna×DeepSeek finalement demandée.

**Protocole d’évaluation.** Nœud archivé, sans run. L'essai effectif est J1R, fiche 28.

**Tests et contrôles.** Vérification de l'archive et de l'absence de lancement. Le code préparé ne vaut pas observation expérimentale.

**Résultat et portée.** Aucune métrique propre ; ne pas compter comme une expérience négative supplémentaire.

Aucune exécution dans l’inventaire pour ce nœud.

Sources détaillées : [j1-j2-results-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 26.


<a id="fiche-27"></a>

#### 27. J2 : nouveau juge dans la bande incertaine

Expérience `38e7efda-ffe4-4da8-b53e-f38cbe627144`.

**Implémentation et intervention.** Rejuger uniquement les paires de score fusionné dans [0,25 ; 0,55). DeepSeek v4.1 flash, effort max, deux échantillons mélangés ; 1 089 paires non ordonnées, découpées en blocs de 64 maximum. Combiner score initial et score du juge par poids de {0 ; 0,25 ; 0,5 ; 0,75 ; 1}.

**Protocole d’évaluation.** DEV44, 44 documents/22 topics. Primaire : poids choisi hors topic, seuil maintenu à 0,39. Secondaire : ajustement conjoint poids/seuil hors topic. Le reste des décisions reste inchangé.

**Tests et contrôles.** Contrôle exhaustif des paires attendues : omission ou duplication fait échouer le jugement, sans défaut négatif. Première tentative interrompue sur une paire omise, puis réparation du transport/cache. Tests ciblés et scores sauvegardés ; contrôle de l'effort demandé et des tokens de raisonnement retournés.

**Résultat et portée.** Primaire F1 45,74 contre 47,32, soit −1,58 [−3,06 ; −0,36]. Secondaire réglé 49,31, exploratoire ; cette piste secondaire est ensuite évaluée dans J3 et n'est plus en attente de confirmation.

Exécutions de référence : [run 582ddbdd](../results/inventaire-executions.csv) (`582ddbdd-9f2b-4878-bd7d-650e2874869b`)

Sources détaillées : [j1-j2-results-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 27.


<a id="fiche-28"></a>

#### 28. J1R : coréférence par accord Luna × DeepSeek

Expérience `8cc07740-6959-4f55-b15a-021f515d0aae`.

**Implémentation et intervention.** Prendre les liens de groupes Luna déjà calculés et demander à DeepSeek max de confirmer l'identité événementielle. Fermer les liens conservés en composantes ; exiger 85 % de précision intrinsèque avant toute propagation causale.

**Protocole d’évaluation.** DEV44 ; 442 liens Luna, 315 liens gold, 43 chunks DeepSeek. La question est d'abord la fiabilité des groupes, pas le F1 causal.

**Tests et contrôles.** Récupération d'un reçu batch Luna déjà payé, sans nouvelle génération ; validation des chunks et de l'identité servie. Le gate empêche explicitement l'évaluation causale quand la précision est insuffisante.

**Résultat et portée.** Liens d'accord après fermeture : 193/362 corrects, précision 53,31 %. Gate 85 % échoué. Aucun F1 causal calculé : l'échec porte sur le préalable intrinsèque.

Exécutions de référence : [run 87a81f5a](../results/inventaire-executions.csv) (`87a81f5a-c099-4c0c-a201-35d99c7d6a9e`)

Sources détaillées : [j1-coref-agreement-result.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 28.


<a id="fiche-29"></a>

#### 29. J3-A : confirmation de la moyenne calibrée J2

Expérience `0be04739-534b-4a5f-a098-0573be60b6cc`.

**Implémentation et intervention.** Ajuster sur DEV44 puis figer le mélange f/J2 : poids 0,5 et seuil 0,56 dans la bande ; hors bande, décision d'origine. Collecter les signaux d'un nouveau lot RES30 sans régler la règle sur ses labels.

**Protocole d’évaluation.** 30 documents, 20 topics, 21 014 candidats ordonnés ; 767 paires non ordonnées dans la bande, deux jugements. Inférence partagée avec J3-B ; IC 97,5 % pour les deux comparaisons prévues.

**Tests et contrôles.** Après le run, découverte d'un gold canonique dans un seul sens comparé à des candidats symétriques. Correction déterministe du gold dans les deux sens, sans changement de prédictions ni nouveau seuil ; double vérification par métriques canoniques officielles et bootstrap. Les F1 bruts du journal sont invalides.

**Résultat et portée.** Résultat corrigé : base 36,0698, candidat 36,3525, +0,2827 point ; IC97,5 par topic [−3,4041 ; 2,8997]. Gain non confirmé. Utiliser le rapport corrigé et non les F1 initiaux.

Exécutions de référence : [run 8f7f88dd](../results/inventaire-executions.csv) (`8f7f88dd-6bed-456b-aad0-dd42fdd2643d`)

Sources détaillées : [res30-results-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 29.


<a id="fiche-30"></a>

#### 30. J3-B : calibration logistique régularisée

Expérience `0a091e08-ee86-4b08-a41e-944b8e5ee36c`.

**Implémentation et intervention.** Apprendre sur DEV44 une régression logistique L2 sur les logits standardisés du score fusionné et du juge. Geler standardisation, coefficients et seuil 0,24 avant RES30. Préserver les décisions initiales hors bande.

**Protocole d’évaluation.** Même RES30 et mêmes inférences que J3-A ; aucune nouvelle collecte propre à B. Branche avec apprentissage supervisé des scores, même si les modèles API ne sont pas entraînés.

**Tests et contrôles.** Contrôle du gel avant labels de réserve et du partage d'inférence. Même correction de symétrie du gold que J3-A ; recalcul indépendant hors ligne sans refit.

**Résultat et portée.** F1 corrigé 35,9878 contre 36,0698, −0,0820 point ; IC97,5 par topic [−4,0831 ; 2,8881]. Pas de gain, pas de promotion.

Exécutions de référence : [run 0516bc85](../results/inventaire-executions.csv) (`0516bc85-715b-47b6-a144-aa276fdeb4f0`)

Sources détaillées : [res30-results-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 30.


<a id="fiche-31"></a>

#### 31. J4-A : fusion logistique supervisée des deux scores initiaux

Expérience `4e3309d0-10d4-4c59-b098-1b4cd01a6639`.

**Implémentation et intervention.** Régression logistique L2=1 sur logits Luna et d1 symétrisés, sans score DeepSeek. Pour chaque fold topic, apprendre moyennes/échelles, coefficients et seuil uniquement sur les autres topics. Ajouter un témoin qui règle seulement le seuil.

**Protocole d’évaluation.** DEV44, 22 folds leave-one-topic-out, 10 916 paires canoniques. Développement exposé ; pas de RES30 et pas d'accès à la réserve.

**Tests et contrôles.** Six tests ciblés rapportés ; contrôle des statistiques d'apprentissage, des folds et des seuils. Reconstruction canonique indépendante des TP/FP/FN et IC ; garde réseau pour un calcul entièrement sur caches.

**Résultat et portée.** F1 46,759 contre 47,323 : −0,564 point ; IC97,5 par topic [−2,576 ; 1,838]. Aucune amélioration ; zéro nouvel appel API.

Exécutions de référence : [run 136a525c](../results/inventaire-executions.csv) (`136a525c-465e-4e65-ac14-22a55b3893e9`)

Sources détaillées : [j4-offline-fusion-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 31.


<a id="fiche-32"></a>

#### 32. J4-B : ajouter des caractéristiques structurelles à la calibration

Expérience `adf2136a-1962-47be-9e5f-387b85c52102`.

**Implémentation et intervention.** Même apprentissage que J4-A, plus désaccord des modèles, asymétrie des orientations, distances de phrase/tokens, même phrase et même surface. Pas d'identifiant topic ni de coréférence gold comme feature.

**Protocole d’évaluation.** Mêmes 22 folds DEV44 et mêmes contrôles ; seule la liste de caractéristiques change. Comparaison exploratoire à la fusion gelée et au témoin de seuil.

**Tests et contrôles.** Même contrôle training-only de la normalisation et du choix du seuil ; six tests de la famille J4 et reconstruction indépendante. Les petits écarts d'arrondi entre plateformes ne changent pas décisions/F1/IC.

**Résultat et portée.** F1 45,185, −2,138 points ; IC97,5 [−4,397 ; 0,232]. Les features ajoutées n'améliorent pas ce protocole.

Exécutions de référence : [run 040544e6](../results/inventaire-executions.csv) (`040544e6-cb82-472f-b1be-16ec3ceb4e4c`)

Sources détaillées : [j4-offline-fusion-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 32.


### E. Décisions relation par relation


<a id="fiche-33"></a>

#### 33. K0 : demander directement chaque relation à Luna

Expérience `f644c423-1a72-44a5-8b37-52e136998f86`.

**Implémentation et intervention.** Un appel batch high par paire non ordonnée ; sortie causal, direction et probability. Cette dernière était supposée représenter une probabilité causale, mais le modèle l'utilise parfois comme confiance dans sa réponse négative.

**Protocole d’évaluation.** Un document DEV44 déjà évalué, 23_9ecbplus.xml ; 120 paires. Smoke technique et estimation de coût avant le round à dix documents.

**Tests et contrôles.** Validation live du transport, du parsing et de la facturation ; inspection des réponses montrant causal=false avec probability=99. Le primaire utilise le booléen ; les analyses probabilistes non corrigées sont non interprétables.

**Résultat et portée.** F1 75,68 contre 94,44 pour la fusion sur ce document facile. Ce n'est pas une performance corpus ni une estimation stable de l'écart.

Exécutions de référence : [run f526f74c](../results/inventaire-executions.csv) (`f526f74c-454f-48f5-bf1f-73e6bd67adf2`)

Code ciblé : [prompts/esl_luna_round_k.yaml](https://github.com/Fireman4740/workere/blob/0534f92f20f796917e6ba369280a47f532cfddcd/prompts/esl_luna_round_k.yaml)

Sources détaillées : [recapitulatif-actualise-2026-10-30.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 33.


<a id="fiche-34"></a>

#### 34. KA : cascade de types puis élimination

Expérience `ec936f5a-5e3b-4ea3-841f-ed31d44e1852`.

**Implémentation et intervention.** Sur la bande de fusion [0,2 ; 0,7), interroger huit types : IDENTITY, PART_OF, TEMPORAL_ONLY, DIFFERENT_INCIDENT, DIRECT_CAUSE, ENABLEMENT, REACTION, PREVENTION. Éliminer progressivement des types puis choisir entre les deux derniers. Garder la décision de fusion hors bande.

**Protocole d’évaluation.** Dix documents DEV44, 417 paires dans la bande ; 3 851 appels chat. Comparaison appariée au F1 fusion 50,00 avec bootstrap documentaire.

**Tests et contrôles.** Traçage des étapes et des types finaux ; candidats et contrôles communs à KB/KC/KD. Pas d'apprentissage des règles à partir des labels de ces dix documents.

**Résultat et portée.** F1 44,89, −5,11 [−9,77 ; −1,20]. 305/417 paires finissent dans un type causal : hausse des faux positifs. La cascade n'agit pas comme un filtre conservateur.

Exécutions de référence : [run 6b39e7f7](../results/inventaire-executions.csv) (`6b39e7f7-371f-4b1a-b86b-fd0bc526a4a1`)

Code ciblé : [prompts/esl_luna_round_k.yaml](https://github.com/Fireman4740/workere/blob/5edbde6e22867bc40e083381c33ff9c3f12a559b/prompts/esl_luna_round_k.yaml)

Sources détaillées : [recapitulatif-actualise-2026-10-30.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 34.


<a id="fiche-35"></a>

#### 35. KB : proposer les causes d'une mention cible

Expérience `b68899f8-464f-4f60-8dad-e9e29fb3db53`.

**Implémentation et intervention.** Un appel par mention, répété trois fois : demander quelles mentions la causent, la facilitent, la motivent ou la préviennent. Ajouter à la fusion les paires proposées dans au moins deux réponses sur trois ; seuils 1/3 et 3/3 secondaires.

**Protocole d’évaluation.** Les mêmes dix documents du round K ; sortie documentaire centrée sur la cible, 645 appels batch. Comparer sur tous les candidats, y compris les non proposés.

**Tests et contrôles.** Réparation d'une connexion SQLite partagée entre threads par clients appropriés aux documents. Cache conservé ; contrôle des doublons, des identifiants et de la règle d'union.

**Résultat et portée.** F1 46,74 contre 50,00, −3,26 [−10,90 ; 4,83]. Ajouts non orientés : 25 TP pour 125 FP. Les vrais ajouts ne compensent pas le bruit.

Exécutions de référence : [run 8d22731d](../results/inventaire-executions.csv) (`8d22731d-0379-490f-b136-f7af0c15365e`)

Code ciblé : [prompts/esl_luna_round_k.yaml](https://github.com/Fireman4740/workere/blob/9b341c4f84f04e906fb1b4cca37bed895f5e9cec/prompts/esl_luna_round_k.yaml)

Sources détaillées : [recapitulatif-actualise-2026-10-30.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 35.


<a id="fiche-36"></a>

#### 36. KC : décider entre événements puis recopier aux mentions

Expérience `a5ca0832-24c6-494e-95d9-ab8a0bead025`.

**Implémentation et intervention.** Produire trois regroupements de mentions ; conserver les liens d'identité vus au moins deux fois, puis composantes connexes. Interroger chaque paire d'événements avec toutes ses mentions et diffuser la décision à toutes les paires de mentions. Les paires internes aux groupes deviennent non causales.

**Protocole d’évaluation.** Même DEV10 que KA/KB ; mesurer intrinsèquement les groupes contre ECB+, puis F1 causal contre les contrôles. 1 383 appels batch.

**Tests et contrôles.** Réparation SQLite avant l'exécution finale ; contrôle des groupes, de la diffusion et des paires originales. Les variantes fondées sur le champ probability de K0 sont explicitement exclues d'une lecture probabiliste naïve.

**Résultat et portée.** Précision de regroupement 38,13 % ; F1 causal 39,42 contre 50,00, soit −10,58 points. La diffusion amplifie les erreurs d'identité.

Exécutions de référence : [run 52be4709](../results/inventaire-executions.csv) (`52be4709-d8f6-4cee-a36f-0b99a386c96d`)

Code ciblé : [prompts/esl_luna_round_k.yaml](https://github.com/Fireman4740/workere/blob/8408f74817e7ac68eeee8899c4cb7d52eda77ead/prompts/esl_luna_round_k.yaml)

Sources détaillées : [recapitulatif-actualise-2026-10-30.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 36.


<a id="fiche-37"></a>

#### 37. KD : combiner les signaux hétérogènes du round K

Expérience `28784112-0cb3-4466-9a0b-56f5aeba0b57`.

**Implémentation et intervention.** Rejouer les caches Luna documentaire, d1, KC et KB. Primaire : moyenne Luna/d1/score KC au seuil 0,39, puis union avec les propositions KB à 2/3. Secondaires : moyenne seule et votes binaires 2/4 ou 3/4. Règles fixes, sans apprentissage.

**Protocole d’évaluation.** DEV10 identique ; aucune inférence nouvelle. K1 n'est pas inclus dans cet ensemble.

**Tests et contrôles.** 2 028 cache hits, zéro nouvel appel et zéro coût API frais ; rapprochement de chaque signal et des conventions de confiance de K0. Bootstrap apparié contre la fusion.

**Résultat et portée.** Primaire F1 42,42 contre 50,00. Vote 3/4 : 49,13, encore inférieur. Aucun ensemble testé ne dépasse la fusion sur ce périmètre.

Exécutions de référence : [run 41d741cb](../results/inventaire-executions.csv) (`41d741cb-5a24-43bf-a423-1a9364d5866f`)

Sources détaillées : [recapitulatif-actualise-2026-10-30.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 37.


<a id="fiche-38"></a>

#### 38. K1 : jugement Luna par paire sur DEV10

Expérience `5a28e40b-a7d0-445b-9ea2-7427f7764149`.

**Implémentation et intervention.** Étendre K0 aux dix documents du round K : un booléen causal par paire non ordonnée, effort high. Pour les secondaires, convertir la confiance dans la réponse en score causal selon le booléen, puis tester une combinaison avec la fusion.

**Protocole d’évaluation.** Même DEV10, contrôle fusion 50,00. Ce run est désormais terminé ; les anciens rapports le donnant « encore en file » sont dépassés.

**Tests et contrôles.** Batch livré, clients par document et cache de la réparation KB ; lecture du JUDGE_RESULT final. Le booléen reste le primaire ; la confiance auto-déclarée n'est pas une probabilité calibrée.

**Résultat et portée.** F1 Luna par paire 40,41 ; combinaison figée avec fusion 42,55. Aucun avantage de cette formulation générique par paire.

Exécutions de référence : [run 591a2c21](../results/inventaire-executions.csv) (`591a2c21-7768-4201-b4db-23e6210ae362`)

Code ciblé : [prompts/esl_luna_round_k.yaml](https://github.com/Fireman4740/workere/blob/4d091c044b76b554a0cc8e8c91d2c39279bec56a/prompts/esl_luna_round_k.yaml)

Sources détaillées : [recapitulatif-actualise-2026-10-30.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 38.


### F. Lecteurs complets et nouvelles propositions


<a id="fiche-39"></a>

#### 39. H1 : Haiku documentaire sur ESC et CTB complets

Expérience `519ac1f5-34cf-40a7-9f6d-636adcfa070d`.

**Implémentation et intervention.** Remplacer Luna par Haiku 5.5 batch, xhigh ; conserver sept passages, une démonstration par passage et les prompts propres au corpus. Calculer aussi d1 Noul et appliquer la fusion 0,39 sans refit. Sur ESC, démonstration hors topic ; sur CTB, autre document.

**Protocole d’évaluation.** ESC train233, 122 130 paires ordonnées ; CTB183, 19 410 candidats du plan exécuté. Comparaisons fusion/d1/lecteur seul appariées à l'intérieur de chaque corpus. Ne pas substituer le compte CTB de 19 442 d'un ancien bilan à celui de ce plan.

**Tests et contrôles.** Reprise après verrou SQLite ; sorties Haiku et CTB d1 réutilisées depuis le cache. Résumé effectif par corpus, modèle servi, TP/FP/FN et bootstrap documentaire relus. DeepSeek historique n'est qu'une référence non appariée, prédictions originales absentes.

**Résultat et portée.** ESC : fusion 43,27, Haiku seul 42,78, d1 35,31 ; gain fusion/Haiku incertain. CTB : Haiku seul 62,57, fusion 47,69 ; perte −14,88 [−18,60 ; −11,20]. La fusion ESC ne se transfère pas telle quelle à CTB.

Exécutions de référence : [run a7f7bbd5](../results/inventaire-executions.csv) (`a7f7bbd5-b115-4926-850a-af6eb7f44776`)

Code ciblé : [experiments/jev/haiku/materialize.py](https://github.com/Fireman4740/workere/blob/36de1a08bc427fb8936a34dabbfeb0035ba47b8f/experiments/jev/haiku/materialize.py)

Sources détaillées : [rapport-complet-2026-10-27.md](../SOURCES.md#annexes-locales) · [h1-fusion-result.json](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 39.


<a id="fiche-40"></a>

#### 40. Qwen documentaire initial : baseline locale sans d1

Expérience `069fa421-bef4-497d-a48a-9510c476a316`.

**Implémentation et intervention.** Exécuter Qwen3.8-27B sur le serveur local exposant une API compatible, avec les prompts documentaires ESC/CTB, sept lectures et vote historique. Pas de modèle de décision ni de fusion d1. Les versions ultérieures de la baseline doivent être distinguées de ce premier contrat.

**Protocole d’évaluation.** Objectif ESC233+CTB183 ; plusieurs essais d'installation, de transport ou de génération ont été interrompus. Un point de contrôle comporte neuf documents CTB entièrement terminés, sans résultat complet ESC.

**Tests et contrôles.** Manifeste des données, prompts, paramètres, contextes et checkpoints ; vérification de complétude avant calcul partiel. Les réponses tronquées ne sont pas des négatifs. Les runs échoués/annulés sont des incidents et pas des scores de modèle.

**Résultat et portée.** Pas de baseline complète de ce nœud. Les résultats CTB partiels sont conditionnés à la complétion et ne permettent pas de classer Qwen face à H1 sur CTB183.

Exécutions de référence : [run d0ce6e2d](../results/inventaire-executions.csv) (`d0ce6e2d-a39b-4770-891a-2394c17d596d`) [run cf753794](../results/inventaire-executions.csv) (`cf753794-588e-41d9-8129-1b6d0c22978f`)

Code ciblé : [experiments/jev/qwen/run.py](https://github.com/Fireman4740/workere/blob/e0059db617bacb9ed3d4280947c01da495adfb1d/experiments/jev/qwen/run.py)

Sources détaillées : [launch-protocol.md](../SOURCES.md#annexes-locales) · [preliminary-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 40.


<a id="fiche-41"></a>

#### 41. Alias virtuels A/R : test d'ancrage par identifiants auxiliaires

Expérience `fe522ecd-a36a-46df-8ab9-f531b29c0fca`.

**Implémentation et intervention.** Comparer des handles auxiliaires d'alias au témoin apparié, sans changer les faits du document. Utiliser un nouveau schéma de sortie et un vote au moins 4/7 ; ce protocole ne reprend pas automatiquement la fusion historique 0,39.

**Protocole d’évaluation.** Smoke sur un document DEV44, sept réponses par bras. Gate de coût : différence d'usage de tokens au plus 1 %, aucune troncature. Première version plafonnée à 4 096 tokens, puis réparation bornée à 16 384 dans les deux bras avec messages inchangés.

**Tests et contrôles.** Kit initial : 37 tests reproduits, 25 régressions ajoutées, puis six régressions Luna rapportées. Contrôles des omissions de documents, types, cardinalités et coût non vide. Le premier score mécanique malgré gate échoué est rejeté ; le scorer est conditionné au gate PASS.

**Résultat et portée.** Version valide : 14 réponses complètes ; A F1 16,00 contre R 18,18, mêmes deux TP et trois FP nets de plus. Aucun handle utilisé dans les arêtes. Signal local défavorable, pas résultat sur DEV44 complet.

Exécutions de référence : [run b81b023d](../results/inventaire-executions.csv) (`b81b023d-05a7-4146-8174-e4eed9b940ee`)

Sources détaillées : [rapport-resultats-smoke.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 41.


<a id="fiche-42"></a>

#### 42. M0 hors ligne : planifier la couverture des paires

Expérience `927184b6-9039-4e7a-ba0f-2faa6fa56b39`.

**Implémentation et intervention.** Construire quatre bras : O extraction ordinaire, P permutation bijective des IDs, M partition des candidats autorisés en sept blocs, MP combinaison. Conserver le texte et les candidats originaux ; inverser la permutation à la lecture des sorties et compter les expositions par paire.

**Protocole d’évaluation.** Pas de modèle dans ce nœud. Planification vérifiée sur les 61 065 paires non ordonnées d'ESC233 ; huit documents DEV44 prévus dans quatre strates de taille, soit 224 prompts rendus.

**Tests et contrôles.** 15 tests rapportés et 150 graphes aléatoires : déterminisme, bijection, union exacte des masques et exposition unique des paires. Préflight des contextes par borne UTF-8+1 024, non par tokenizer exact. Un écart de taille de bloc jusqu'à deux est conservé et expliqué par les restes de strates.

**Résultat et portée.** Validation d'infrastructure et de couverture PASS ; zéro appel modèle et aucun F1. La qualité de M/P/MP est testée dans le smoke distinct.

Exécutions de référence : [run 19f7f9a9](../results/inventaire-executions.csv) (`19f7f9a9-4a9e-44b8-a151-dfe578971508`)

Sources détaillées : [rapport-etape-m0-2026-10-27.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 42.


<a id="fiche-43"></a>

#### 43. P1 v1.1 : extraire des preuves sources avant transport local

Expérience `3b57057e-71b0-4900-b8aa-2efd65d4f204`.

**Implémentation et intervention.** Partir des positifs de la fusion gelée et extraire des certificats causaux textuels : source, preuve citée et liaison aux occurrences. Le plan compare ensuite identité locale A0, cadres A1, témoin transporté A2 et groupes historiques A3, avec vérification commune, sans récursion des ajouts.

**Protocole d’évaluation.** Extraction initiale sur DEV44 : 44 requêtes, plafond 8 192 tokens incluant raisonnement. Le gate de volume exige 50 possibilités exploratoires avant la comparaison finale des bras.

**Tests et contrôles.** Après échec, récupération GET-only des six lots déjà soumis, sans nouvelle génération. Contrôles des citations littérales, couverture des mentions, reçus et coût. 39 réponses complètes ; cinq troncatures ; 26 certificats structurels utilisables, sans certification sémantique indépendante.

**Résultat et portée.** Volume insuffisant, même en réparant les emplacements tronqués/invalides selon le diagnostic. Pas de registre final de candidats, pas de comparaison A0–A3, pas de F1. Le run de récupération terminé ne transforme pas l'essai scientifique en succès.

Exécutions de référence : [run ca2990dd](../results/inventaire-executions.csv) (`ca2990dd-5338-418c-bfdb-abd220a32782`) [run e390a333](../results/inventaire-executions.csv) (`e390a333-f7da-4844-aaca-aac99607ac8f`)

Code ciblé : [experiments/jev/proof_transport_launch/controller.py](https://github.com/Fireman4740/workere/blob/3663d645fb1e31e2dbb8246220ffb91ce08c62ae/experiments/jev/proof_transport_launch/controller.py) [experiments/jev/proof_transport_launch/recover_only.py](https://github.com/Fireman4740/workere/blob/3663d645fb1e31e2dbb8246220ffb91ce08c62ae/experiments/jev/proof_transport_launch/recover_only.py)

Sources détaillées : [bilan-final-extraction.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 43.


<a id="fiche-44"></a>

#### 44. M0 live : O/P/M/MP sur deux documents

Expérience `2a1c6883-9130-4746-8064-2c19e131653f`.

**Implémentation et intervention.** Exécuter sept lectures dans chaque bras. Pour chaque bras, ajouter à P0 uniquement les paires Z0, négatives sous P0 et à zéro vote Luna historique, proposées au moins une fois. Conserver les décisions P0 ; aucune fusion directe de V/E avec 0,39.

**Protocole d’évaluation.** Deux documents prédéfinis, 1 246 paires non ordonnées, 56 requêtes. Même calendrier de démonstrations hors topic ; contraste primaire M−O. Une paire voit sept sorties O/P et une seule exposition M/MP : les dénominateurs sont explicites.

**Tests et contrôles.** 56 sorties complètes, dont quatre listes vides valides ; une paire hors masque filtrée. Reconstruction réponse→registre→ajouts avant gold ; couverture, permutation inverse et usage tokens contrôlés.

**Résultat et portée.** P0 F1 36,92 ; +O 44,12 ; +M 37,14 ; +MP 42,86 ; +P 34,29. Primaire M−O −6,97 points. Union O+MP à 49,32 est post hoc et consomme plus de calcul ; elle motive M1, sans promotion M0.

Exécutions de référence : [run 08eaa19b](../results/inventaire-executions.csv) (`08eaa19b-af21-400d-94c9-20e96d0dbb93`)

Sources détaillées : [rapport-resultats-m0-2026-10-27.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 44.


<a id="fiche-45"></a>

#### 45. Placebos V0 : faisabilité d'éditions contrôlées automatiques

Expérience `4b3cd218-0007-4211-b414-4840b7175316`.

**Implémentation et intervention.** Pour une relation sélectionnée, demander une transformation cible et trois placebos comparables, en protégeant mentions et faits requis. Un auditeur automatique devait valider les éditions, puis l'extracteur mesurer la réponse sous transformation. Pas de nouvelle annotation humaine.

**Protocole d’évaluation.** Douze paires sélectionnées sans gold ; six disposent des snapshots requis, sur quatre documents. Six appels éditeur effectifs sous plafond 2 USD.

**Tests et contrôles.** Schéma, IDs et snapshots figés ; réponses stop et identifiants uniques. Premier échec au garde budgétaire avant POST, correction comptable sans modification des messages. Audit des hashes et des refus ; pas de changement opportuniste des cas après refus.

**Résultat et portée.** Six refus explicites de construire un ensemble valide ; zéro auditeur, zéro extraction transformée, zéro signature V0 et aucun F1. L'effet du contraste n'est pas mesuré : le générateur ne satisfait pas les contraintes de l'expérience.

Exécutions de référence : [run fbd22a5d](../results/inventaire-executions.csv) (`fbd22a5d-8571-465a-8acf-f99e99eefd40`)

Sources détaillées : [rapport-final-pilote-automatique.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 45.


<a id="fiche-46"></a>

#### 46. CTB attribution : concurrence entre liaisons argumentales

Expérience `6d8bda08-e1bf-49e3-9b4f-f4592a01cee9`.

**Implémentation et intervention.** Réutiliser les votes Haiku H1 et construire des supports lexicaux déterministes. F ajoute selon les faits provisoires ; A ajoute une compétition tenant compte de la portée et des arguments. Garder mêmes candidats/supports et un témoin F ramené à la même couverture que A.

**Protocole d’évaluation.** Échantillons verrouillés de 20 unités de développement et 40 de test ; test à 370 paires. 32 seeds fixes pour l'égalisation de couverture ; ablation sans arêtes ; bootstrap par document. Calcul CPU hors ligne.

**Tests et contrôles.** Règles figées avant évaluation ; aucune nouvelle inférence, aucun label humain de vérité d'attribution. Une seule unité déclenche la concurrence. La première tentative a échoué sur le chemin de uv avant calcul, pas sur l'hypothèse.

**Résultat et portée.** Sur le test : référence 69,14 ; F 65,88 ; A 66,67. A et F à couverture égale donnent exactement les mêmes décisions sur les 32 seeds. Aucun apport propre de la concurrence montré.

Exécutions de référence : [run e788ecb4](../results/inventaire-executions.csv) (`e788ecb4-039e-40d4-89d3-9dcd230fdec0`)

Sources détaillées : [rapport-pilote-automatique.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 46.


<a id="fiche-47"></a>

#### 47. M1 brouillon : nœud préparatoire non implémenté

Expérience `34b19e46-8794-48e4-b445-1f1402975a4e`.

**Implémentation et intervention.** Annonce de C15 contre H7O+7MP à partir de M0. Le nœud reste sur le code M0, sans implémentation M1 effective.

**Protocole d’évaluation.** Aucun run attaché. L'expérience M1 exécutée est le nœud distinct de la fiche suivante.

**Tests et contrôles.** Comparaison des branches et du point d'entrée conservée dans le rapport M1 ; absence de duplication de génération.

**Résultat et portée.** Aucun score propre. Le titre d'une expérience ne démontre pas que son implémentation ou son exécution a eu lieu.

Aucune exécution dans l’inventaire pour ce nœud.

Sources détaillées : [m1-final-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 47.


<a id="fiche-48"></a>

#### 48. M1 : diversité complémentaire contre lectures ordinaires

Expérience `e06a915f-f97c-4f97-8310-bc3338d9b0e4`.

**Implémentation et intervention.** C=15 lectures ordinaires ; H=7 lectures ordinaires communes +7 MP. Générer une seule fois les sept communes : 7+8+7=22 requêtes uniques/document. Ajouter uniquement à Z0 ; maintenir P0. Le nombre 15 du contrôle provient d'une allocation de coût préenregistrée, pas d'un réglage après résultats.

**Protocole d’évaluation.** 20 documents de 20 topics, 4 347 paires non ordonnées ; 440 requêtes. Comparaison H−C, ajouts TP/FP, exposition valide par paire, 20 000 bootstrap topics et retraits de topics. Pas de sélection du meilleur passage.

**Tests et contrôles.** Gel/hashes avant évaluation ; inversion des permutations, déduplication, compteurs E_planned/submitted/delivered/valid. 18 troncatures au plafond 10 000, mises en quarantaine. Le ratio de coût pondéré H/C=0,8483 ne respecte pas ±5 % ; le protocole conserve INCONCLUSIF pour le test à coût égal.

**Résultat et portée.** F1 P0 42,50, C 42,39, H 41,09 ; H−C −1,302 [−3,646 ; 0,170]. H ajoute 17 TP et 97 FP. Pas de gain validé ; la complémentarité de certains TP n'assure pas leur utilité nette.

Exécutions de référence : [run 0be48b97](../results/inventaire-executions.csv) (`0be48b97-e191-455c-ad13-db1214fd22cd`)

Sources détaillées : [m1-final-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 48.


<a id="fiche-49"></a>

#### 49. Qwen xhigh, température zéro : reprise de baseline

Expérience `b04d73c3-1dd9-42a9-a27b-d321278bb250`.

**Implémentation et intervention.** Déclarer le contrat local thinking xhigh et température 0, conserver prompts ESC/CTB, sept lectures, autres paramètres d'échantillonnage et checkpoints compatibles. Les namespaces de cache évitent de mélanger les configurations incompatibles.

**Protocole d’évaluation.** Objectif ESC233+CTB183 ; analyse préliminaire limitée aux documents entièrement terminés. Les changements demandés de température/thinking appartiennent à cette version, pas au premier Qwen.

**Tests et contrôles.** Vérification des corps de prompts avant reprise ; sources et paramètres effectifs dans le journal. Interruptions/annulations conservées. Les documents complets, les lectures isolées et les scores définitifs ne sont pas des unités interchangeables.

**Résultat et portée.** Exécution mixte ensuite arrêtée au profit de CTB seul. Pas de score final des deux corpus attribué à ce nœud ; consulter les bilans préliminaires et la continuation CTB.

Exécutions de référence : [run a082a2f5](../results/inventaire-executions.csv) (`a082a2f5-96b1-4c75-a89f-8bbed54d2ed2`) [run 6233eb3f](../results/inventaire-executions.csv) (`6233eb3f-8d87-4e56-a769-70c6cea0ac5a`)

Code ciblé : [experiments/jev/qwen/run.py](https://github.com/Fireman4740/workere/blob/38e222e3b926fe3ced3818fcddb6e3f64d963358/experiments/jev/qwen/run.py)

Sources détaillées : [thinking-xhigh-relaunch.md](../SOURCES.md#annexes-locales) · [comparison-report.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 49.


### G. Campagnes N1–N13 et continuations


<a id="fiche-50"></a>

#### 50. N1 : filtrer les ajouts par preuve et ancrage

Expérience `3b10f583-c944-4dfa-92ba-1a5ff8873bbd`.

**Implémentation et intervention.** Reprendre 120 candidats distincts ordinaires/MP de M1 ; demander support causal et rattachement exact aux mentions. Les citations doivent être littérales et couvrir les occurrences requises. Comparer C filtré aux ajouts ordinaires communs +MP filtrés H, sans retirer P0.

**Protocole d’évaluation.** Même DEV20 que M1, donc exposé. 45 réponses de vérification ; critères gelés : gain au moins un point, utilité marginale positive et stabilité au retrait de chaque topic.

**Tests et contrôles.** Audit indépendant reconstruit décisions et métriques depuis les réponses, 45/45 complètes, aucune troncature. Contrôle du gel avant gold, des empreintes et du rapprochement coût batch. La validité d'une citation n'est pas une validation humaine du lien causal.

**Résultat et portée.** P0 F1 42,497 ; C 42,950 ; H 42,340. H−C −0,610, IC95 topic [−1,824 ; 0,463]. NO_GO opérationnel ; ajouts H 5 TP/22 FP, précision marginale 18,52 %.

Exécutions de référence : [run 82e1d497](../results/inventaire-executions.csv) (`82e1d497-7441-4115-b5a3-6cd499689b67`)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 50.


<a id="fiche-51"></a>

#### 51. N2 : transport local v2, même pool et vérificateur commun

Expérience `338d526f-2cd1-4e12-9783-10db01df8e32`.

**Implémentation et intervention.** Réemployer les 26 certificats structurels P1 ; vérifier séparément leur source. Générer lexicalement neuf substitutions et les soumettre aux quatre politiques : A0 identité d'occurrence, A1 compatibilité des cadres, A2 réutilisation du témoin causal, A3 groupes T2. Un vérificateur final commun décide l'admission ; pas de récursion.

**Protocole d’évaluation.** 62 nouvelles réponses, neuf candidats sur cinq documents. Gate de rendement : au moins cinq ajouts sur trois topics. Le registre de candidats est gelé avant décision, et aucune lecture du gold n'est requise si les prédictions restent identiques à P0.

**Tests et contrôles.** 62/62 réponses complètes ; audit indépendant des corps, IDs, citations, hashes et quatre bras. 24 sources sur 26 déclarées soutenues par le modèle ; cela n'est pas une précision sémantique certifiée. Les mêmes neuf candidats et le même vérificateur empêchent de confondre mécanisme et richesse du dossier.

**Résultat et portée.** Admissions pré-vérificateur 1/1/0/1 pour A0/A1/A2/A3 ; zéro ajout final partout. ΔF1 exactement nul par identité des prédictions ; précision des ajouts indéfinie. Mécanismes non départagés, pool lexical trop peu productif.

Exécutions de référence : [run e3d9885b](../results/inventaire-executions.csv) (`e3d9885b-2297-48d0-84d7-fbffe8745723`)

Code ciblé : [experiments/jev/n2_transport/n2.py](https://github.com/Fireman4740/workere/blob/4d4a27c4d3e1c971fa974d789f0febde52eda9b1/experiments/jev/n2_transport/n2.py) [experiments/jev/n2_transport/test_n2.py](https://github.com/Fireman4740/workere/blob/4d4a27c4d3e1c971fa974d789f0febde52eda9b1/experiments/jev/n2_transport/test_n2.py)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 51.


<a id="fiche-52"></a>

#### 52. N5 : ablation exhaustive 1/3/7 lectures sur caches

Expérience `a44ecab3-6c74-4895-84e0-344510820848`.

**Implémentation et intervention.** Rejouer les sept réponses Luna historiques de CONF40. Évaluer les sept singletons, les 35 triples et l'ensemble de sept ; ne choisir aucun meilleur sous-ensemble. Recalculer les votes et la fusion historique avec d1 identique et seuil 0,39.

**Protocole d’évaluation.** 40 documents, 20 topics, déjà exposés. Rapporter séparément moyenne des F1 et F1 des comptes moyens, car ces opérations ne commutent pas. Bootstrap topics des différences fondées sur les comptes attendus.

**Tests et contrôles.** Probabilités reconstruites sur sept lectures identiques aux métadonnées, zéro écart. Entrées immuables, script sans client modèle, zéro appel frais. Le bootstrap utilise 2 000 tirages seed 20261028.

**Résultat et portée.** F1 moyen : 1 lecture 38,362 ; 3 lectures 40,779 ; 7 lectures 40,928. Perte moyenne à trois −0,149 point, économie de 57,14 % des appels Luna uniquement. Non-infériorité future non établie ; N8 la teste avec un vrai préfixe frais.

Exécutions de référence : [run ac747a2d](../results/inventaire-executions.csv) (`ac747a2d-8ea7-4066-990c-f94539c3cc54`)

Code ciblé : [pipelines/jev/n5.py](https://github.com/Fireman4740/workere/blob/780c6eefff8556a1d277c8272cac010da4950839/pipelines/jev/n5.py)

Sources détaillées : [rapport.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 52.


<a id="fiche-53"></a>

#### 53. N3 : paquet de démonstrations contrastives

Expérience `c007d4b4-6142-4b2d-a3a1-7da0e3c8716b`.

**Implémentation et intervention.** Comparer trois lectures ordinaires, une démonstration positive hors topic par lecture, à trois lectures avec sélection structurelle et négatifs explicites. Même Luna high, sortie 32 768, d1 et fusion symétrique 0,39. Deux interventions restent empaquetées dans le bras contrastif.

**Protocole d’évaluation.** Douze documents de douze topics DEV exposés ; 72 requêtes, 4 656 candidats orientés. Primaire : delta F1 avec IC95 par topic ; gate ≥1 point et borne basse >0. Une seule réalisation par bras.

**Tests et contrôles.** 72 sorties JSON strictes, aucune duplication/ID inconnu ; corps comparés aux hashes de préflight, exemples hors topic. Audit distinct reconstruit votes orientés, fusion, comptes, bootstrap et coût depuis les réponses brutes, sans appeler l'agrégateur du projet.

**Résultat et portée.** F1 45,473→49,513, +4,040 [−0,950 ; 10,428]. Gate non franchi malgré les retraits de topic favorables. Signal initial à répliquer, aucune promotion.

Exécutions de référence : [run ff35cbc2](../results/inventaire-executions.csv) (`ff35cbc2-b157-4564-87c0-271081d105ff`)

Code ciblé : [pipelines/jev/n3.py](https://github.com/Fireman4740/workere/blob/f2f58ed629268e44a20458a2e80174d12942c977/pipelines/jev/n3.py)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 53.


<a id="fiche-54"></a>

#### 54. N4 : remplacer Luna par DeepSeek documentaire

Expérience `3c1651c8-a305-44a9-a1ac-32096dc3359a`.

**Implémentation et intervention.** Donner les mêmes messages/démonstrations aux deux lecteurs, trois slots par document ; Luna batch et DeepSeek direct après rejet du transport batch DeepSeek. Effort high et plafond de complétion 32 768. Primaire lecteur seul, majorité 2/3 ; fusion secondaire déclarée séparément.

**Protocole d’évaluation.** Douze documents/12 topics, 72 réponses attendues. Exiger toutes les réponses complètes avant d'ouvrir le gold. Le secondaire utilise un vote non orienté par lecture, distinct de la fusion historique orientée.

**Tests et contrôles.** Audit brut : Luna 36/36 stop ; DeepSeek 12/36 stop et 24 length, sans contenu final. Raisonnement DeepSeek 99,788 % des tokens de complétion. Sorties gelées, metrics=null, aucun sous-ensemble complet scoré pour remplacer le primaire.

**Résultat et portée.** BLOCKED_VALIDITY ; aucun F1 comparatif. Le problème observé est le contrat de génération, pas une mauvaise qualité causale mesurée de DeepSeek.

Exécutions de référence : [run c3a66705](../results/inventaire-executions.csv) (`c3a66705-3b67-4cde-869f-055cda7dab17`)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 54.


<a id="fiche-55"></a>

#### 55. N6 : question lexicale adaptée à CTB

Expérience `95286508-31f4-4318-9ea6-cb7fa3bb7392`.

**Implémentation et intervention.** Comparer avec d1 une question causale générique à une question exigeant un ancrage lexical CTB. Même contexte et une orientation canonique par paire ; seuil 0,5 dans les deux bras. Pas de transfert de la fusion ESC à 0,39.

**Protocole d’évaluation.** Sept documents CTB sélectionnés par métadonnées/hashes ; 73 paires, deux questions, 146 requêtes. Petit échantillon exposé, pas confirmation.

**Tests et contrôles.** Huit tests rapportés ; reçus durables avant POST, modèles et probabilités archivés. Premier incident avant POST, puis 60 réponses valides et 86 HTTP429. Les 86 coûts non retournés restent provisionnés. Inspection accidentelle de 12 labels après gel divulguée ; aucune sélection modifiée.

**Résultat et portée.** INCONCLUSIVE_INTEGRITY ; pas de score primaire. Les réponses manquantes ne sont ni négatives ni des erreurs de causalité.

Exécutions de référence : [run 9ad51ca1](../results/inventaire-executions.csv) (`9ad51ca1-9351-4428-bfe6-407ca4bd21be`)

Sources détaillées : [rapport-n6.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 55.


<a id="fiche-56"></a>

#### 56. N3 audit : récupération et reproduction sans génération

Expérience `126ff578-40cb-4041-8f37-6b6aad85cf20`.

**Implémentation et intervention.** Récupérer les sorties de N3 puis recalculer indépendamment son pipeline : parser les 72 réponses, reconstruire votes/fusion, comptes, bootstrap topics, retraits de topics et coûts. Ne pas modifier le nœud scientifique parent.

**Protocole d’évaluation.** Exactement les mêmes sorties que N3. Il s'agit d'un test de reproductibilité, pas d'une seconde réalisation stochastique ni d'une confirmation sur de nouveaux documents.

**Tests et contrôles.** 324 morceaux d'export, SHA de l'archive et 99 fichiers du manifeste vérifiés. Le script d'audit n'appelle ni modèle ni agrégateur original ; l'inventaire du run de récupération indique zéro génération.

**Résultat et portée.** Résultat N3 reproduit exactement ; son statut reste inconclusif. Le nombre d'exécutions ne doit pas faire compter le +4,040 deux fois.

Exécutions de référence : [run cbb76514](../results/inventaire-executions.csv) (`cbb76514-9cb5-4671-b334-ad18984bbb8a`)

Code ciblé : [experiments/jev/n3/recover_audit.py](https://github.com/Fireman4740/workere/blob/3e72fdd554e12ff162e0e393fe07f71b9e7e8d4d/experiments/jev/n3/recover_audit.py)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 56.


<a id="fiche-57"></a>

#### 57. N7 : réplication du paquet N3 sur deux calendriers

Expérience `2b118e89-feae-4028-8a5f-041b35fadc7a`.

**Implémentation et intervention.** Conserver le paquet structure+négatifs N3 et le comparer au contrôle ordinaire sur deux calendriers frais, trois lectures par bras. Même d1 et fusion 0,39. Des prompts contrastifs identiques entre calendriers doivent tout de même produire des générations fraîches distinctes.

**Protocole d’évaluation.** Seize documents de seize topics, distincts des documents N3 ; 16×2 calendriers×2 bras×3 lectures=192 requêtes. Primaire : moyenne des deux écarts F1 ; gate ≥1 point, borne IC>0 et gain dans chacun des deux calendriers.

**Tests et contrôles.** Quatre tests ciblés et revue préflight rapportés ; 192 slots, 135 corps distincts, 64 lots terminés. Gate api_call=true/cache_hit=false, IDs de réponses distincts ; gel avant gold. Audit indépendant rejoue toutes les réponses ; bootstrap 20 000 topics.

**Résultat et portée.** Écarts −3,247 puis +4,329 ; moyenne +0,541 [−1,614 ; 3,372]. Aucun gain robuste. La variation ne peut pas être attribuée uniquement au décodage : calendrier et ordre de soumission font aussi partie des conditions.

Exécutions de référence : [run 350ed2fa](../results/inventaire-executions.csv) (`350ed2fa-4dfa-4f8c-bed3-2edd259cb5ef`)

Code ciblé : [pipelines/jev/n7.py](https://github.com/Fireman4740/workere/blob/69b91eaa3b3028606086914b8f54bdaad18ee33b/pipelines/jev/n7.py) [experiments/jev/n7/test_n7.py](https://github.com/Fireman4740/workere/blob/69b91eaa3b3028606086914b8f54bdaad18ee33b/experiments/jev/n7/test_n7.py) [experiments/jev/n7/audit_n7.py](https://github.com/Fireman4740/workere/blob/69b91eaa3b3028606086914b8f54bdaad18ee33b/experiments/jev/n7/audit_n7.py)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 57.


<a id="fiche-58"></a>

#### 58. N8 : trois lectures fraîches réellement incluses dans sept

Expérience `2483cd7a-30a8-4836-8f97-06386dd31d05`.

**Implémentation et intervention.** Produire un seul calendrier frais de sept lectures ordinaires ; la variante à trois utilise exactement les trois premières réponses, pas un sous-ensemble choisi après les scores. Maintenir d1, fusion et seuil 0,39.

**Protocole d’évaluation.** Huit documents/topics ; 56 requêtes. Test de non-infériorité avec marge préfixée d'un point F1. Documents distincts de N7, mais topics pas nécessairement disjoints.

**Tests et contrôles.** Quatre tests ciblés et intégration synthétique 56 sorties rapportés ; 56/56 réponses réelles complètes. Audit brut des 21 fichiers du manifeste et vérification du partage exact du préfixe ; bootstrap par topic.

**Résultat et portée.** F1 3 lectures 35,685 contre 7 lectures 36,364 ; Δ −0,679 [−3,174 ; 0,292]. La borne basse dépasse la perte tolérée : non-infériorité non établie. Ce n'est pas une preuve d'infériorité universelle.

Exécutions de référence : [run fb701e7a](../results/inventaire-executions.csv) (`fb701e7a-0619-4fbb-9e2a-fa5775588b83`)

Code ciblé : [experiments/jev/n8/n8.py](https://github.com/Fireman4740/workere/blob/16092e10a4156f12078fc8b3d590f8bfea9524ec/experiments/jev/n8/n8.py) [experiments/jev/n8/test_n8.py](https://github.com/Fireman4740/workere/blob/16092e10a4156f12078fc8b3d590f8bfea9524ec/experiments/jev/n8/test_n8.py) [experiments/jev/n8/audit_n8.py](https://github.com/Fireman4740/workere/blob/16092e10a4156f12078fc8b3d590f8bfea9524ec/experiments/jev/n8/audit_n8.py)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 58.


<a id="fiche-59"></a>

#### 59. N9 : factoriel récupération × négatifs

Expérience `c2fc283c-e85d-40e6-b231-00aa5759a96a`.

**Implémentation et intervention.** Croiser R0/R1 (exemples aléatoires ou récupération structurelle) et N0/N1 (positifs seuls ou négatifs explicites). Les exemples sont identiques entre N0 et N1 d'un même choix R. Garder Luna high, trois lectures, d1 et fusion 0,39 ; mélanger bras et calendriers dans les lots par document.

**Protocole d’évaluation.** Dix documents/topics, deux calendriers : 10×2×4×3=240 générations. Contrastes déclarés : effets principaux, interaction et trois variantes face à R0N0, avec correction de multiplicité.

**Tests et contrôles.** Huit tests et intégration synthétique à 240 sorties rapportés ; 240 vraies réponses complètes et IDs uniques. Audit autonome des réponses, requêtes, hashes, décisions et coûts. Les interruptions de suivi SSH n'ont entraîné ni nouvelle soumission ni nouvelle génération.

**Résultat et portée.** Face au témoin, négatifs seuls +0,503 ; structure seule −1,041 ; paquet complet −1,152 point en moyenne. Tous les IC corrigés incluent zéro. Aucun composant retenu ; les facteurs peuvent enfin être distingués, sans effet robuste démontré.

Exécutions de référence : [run d9b0820c](../results/inventaire-executions.csv) (`d9b0820c-37fb-47a5-bfb9-cf18ef8c2e16`)

Code ciblé : [pipelines/jev/n9.py](https://github.com/Fireman4740/workere/blob/342709d8b29933db40c64af365f656dc4900ee17/pipelines/jev/n9.py) [experiments/jev/n9/test_n9.py](https://github.com/Fireman4740/workere/blob/342709d8b29933db40c64af365f656dc4900ee17/experiments/jev/n9/test_n9.py) [experiments/jev/n9/audit_n9.py](https://github.com/Fireman4740/workere/blob/342709d8b29933db40c64af365f656dc4900ee17/experiments/jev/n9/audit_n9.py)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 59.


<a id="fiche-60"></a>

#### 60. N10 : six lectures contre tous les triples de N7

Expérience `bfe65f5c-1775-4082-b398-94ba4b520a71`.

**Implémentation et intervention.** Réutiliser les deux calendriers de N7 pour obtenir six lectures par bras/document. Comparer les six à la moyenne de chacun des 20 sous-ensembles de trois ; refaire cette moyenne à chaque bootstrap. Comparer aussi six contrastives à six ordinaires.

**Protocole d’évaluation.** Seize documents/topics ; 192 réponses déjà payées. Trois contrastes exploratoires, correction Bonferroni, IC 98,3333 %. Ne pas choisir le meilleur triple ni requalifier l'analyse en confirmation de N7.

**Tests et contrôles.** Six tests ciblés rapportés ; audit indépendant des 42 systèmes de prédictions (21 par bras), gel avant gold. Exécution sans client réseau ; les très petits écarts d'arrondi d'un Python ancien sont distingués des changements de décisions.

**Résultat et portée.** Six−moyenne des triples : ordinaire +0,161 [−1,539 ; 2,538], contrastif +0,544 [−0,246 ; 1,460]. Aucun candidat. Zéro appel frais ne signifie pas que six lectures seraient gratuites en production.

Exécutions de référence : [run 8354656e](../results/inventaire-executions.csv) (`8354656e-4a7e-4019-8a4a-8b649fb9d257`)

Code ciblé : [pipelines/jev/n10.py](https://github.com/Fireman4740/workere/blob/1010fcd3d0195181267d021f37f21ae9d0ab25e5/pipelines/jev/n10.py) [experiments/jev/n10/test_n10.py](https://github.com/Fireman4740/workere/blob/1010fcd3d0195181267d021f37f21ae9d0ab25e5/experiments/jev/n10/test_n10.py) [experiments/jev/n10/audit_n10.py](https://github.com/Fireman4740/workere/blob/1010fcd3d0195181267d021f37f21ae9d0ab25e5/experiments/jev/n10/audit_n10.py)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 60.


<a id="fiche-61"></a>

#### 61. N11 : arrêt anticipé exact des sept lectures

Expérience `4254c7f2-ca34-437f-8f28-d61a016ca5ec`.

**Implémentation et intervention.** Après k lectures, borner le score final à sept en supposant les futurs votes tous nuls ou tous positifs. Dénominateur toujours sept ; arrêter un document seulement quand toutes ses paires sont irrévocablement classées. Aucun changement de prompt ou de seuil.

**Protocole d’évaluation.** Rejeu de deux cohortes compatibles séparées : CONF40 historique et DEV8 N8 frais. Mesures : appels économisés, tokens historiques évités et égalité exacte des décisions. Gate de poursuite : au moins 20 % d'appels économisés.

**Tests et contrôles.** Replay indépendant en arithmétique rationnelle, sans gold ; vérification des cas de frontière >=0,39 et de la reconstruction de toutes les décisions finales. Inventaire séparé de la disponibilité des caches pour un éventuel corpus complet.

**Résultat et portée.** CONF40 : 18/280 appels économisés, 6,43 % ; DEV8 : 9/56, 16,07 %. Zéro décision changée, gate de 20 % non atteint. Ce test mesure une économie exacte, pas un gain F1.

Exécutions de référence : [run 005dff89](../results/inventaire-executions.csv) (`005dff89-b55e-42d8-b81f-63e160fb83ff`)

Code ciblé : [experiments/jev/n11/main.py](https://github.com/Fireman4740/workere/blob/b73126218ae6623db94eb45c805691ab11b5cf16/experiments/jev/n11/main.py) [experiments/jev/n11/test_n11.py](https://github.com/Fireman4740/workere/blob/b73126218ae6623db94eb45c805691ab11b5cf16/experiments/jev/n11/test_n11.py)

Sources détaillées : [bilan-verifie.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 61.


<a id="fiche-62"></a>

#### 62. N12 : changer la vue des occurrences

Expérience `bdbab303-7898-4fb0-a9a9-676bda82b084`.

**Implémentation et intervention.** Trois vues déterministes : original ; local, union des phrases de toutes les mentions et de leurs voisines ±1, avec marqueurs d'omission ; table, texte intégral suivi de ID/phrase/surface pour chaque occurrence. Aucun regroupement d'événements ni renumérotation. Garder les mêmes démonstrations dans les trois bras.

**Protocole d’évaluation.** Quatre documents ESC exposés, deux calendriers, trois lectures par vue : 72 requêtes, quatre lots mixtes de 18. Même fusion Luna×d1 à 0,39 ; smoke technique/exploratoire sans promotion.

**Tests et contrôles.** Quatre tests de contrat, dont intégration synthétique à 72 sorties. Assertions : conservation de toutes les mentions/candidats, texte marqué d'origine exact, exemples hors topic et préfixes partagés. 72 réponses réelles complètes et uniques ; audit brut indépendant des votes, F1 et coûts.

**Résultat et portée.** Local : +4,566 puis +1,094 points, moyenne +2,830 ; table : moyenne −0,514. Petit signal sur quatre documents, sans généralisation. C'est une vue documentaire globale réduite, pas un prompt distinct par paire.

Exécutions de référence : [run f5f58d8c](../results/inventaire-executions.csv) (`f5f58d8c-d3ce-4c2f-b2bd-d6dcc8ca9578`)

Code ciblé : [pipelines/jev/n12.py](https://github.com/Fireman4740/workere/blob/bdb987d3acf93a1cf681d093583baee2d574e6b8/pipelines/jev/n12.py) [experiments/jev/n12/test_n12.py](https://github.com/Fireman4740/workere/blob/bdb987d3acf93a1cf681d093583baee2d574e6b8/experiments/jev/n12/test_n12.py)

Sources détaillées : [bilan-verifie-2026-10-31.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 62.


<a id="fiche-63"></a>

#### 63. N13 : représentations avec Qwen local, versions ESC puis CTB50

**Actualisation du 12 novembre.** Trois smokes supplémentaires terminés ne passent pas la validité :63/72,69/72 et68/72 sorties valides. Chaque journal écrit BLOCKED_VALIDITY sans lire le gold. Aucun score de pilote CTB50 complet annoncé. [run e163b810](../results/inventaire-executions.csv) (`e163b810-e9f7-419d-b769-168265cf4c49`) [run 2d6766a2](../results/inventaire-executions.csv) (`2d6766a2-3d10-4a32-a19f-e8009e35efbc`) [run 09873773](../results/inventaire-executions.csv) (`09873773-2908-459d-bced-dc436272fb32`)

Expérience `0594a3dc-b551-4b9a-9005-82ed671be484`.

**Implémentation et intervention.** Plan initial ESC : quatre documents smoke puis DEV40, trois vues N12, deux calendriers de trois lectures, majorité 2/3 non orientée, sans d1. Le protocole exécuté a ensuite été déplacé vers 50 documents CTB : candidats intra-phrase, démonstrations hors fold ; 900 requêtes prévues, dont 72 de contrôle sur quatre documents avant les 828 restantes. Dans la version CTB, la vue locale présente séparément chaque phrase contenant des mentions, dans son ordre et avec ses IDs originaux ; elle diffère donc de l'union ±1 phrase de N12 sur ESC. Ne pas fusionner ces plans malgré le titre historique du nœud.

**Protocole d’évaluation.** CTB50 : Qwen déclaré qwen3.8-27b, température 0. Premier smoke xhigh sortie 32 768 ; second xhigh sortie 65 536, contexte 81 920, entrée 16 384 et concurrence 4 ; puis relance high demandée, distincte. Chaque phase exige toutes ses réponses valides. Primaire : majorité par paire non orientée, une voix par lecture.

**Tests et contrôles.** Tests de contrat présents : refus de fournisseur externe, 900 corps/72+828, déduplication des sens, tokenizer exact bloquant avant génération, contexte déclaré, exclusions de folds/démonstrations, gel avant gold et IDs uniques. Audits hors réseau des sorties réelles : 63/72 puis 69/72 valides ; le document difficile reste inclus. Les 828 requêtes principales restent bloquées dans ces deux smokes xhigh.

**Résultat et portée.** Deux smokes xhigh BLOCKED_VALIDITY, aucun F1 primaire CTB50. Analyse post hoc explicitement demandée : trois documents entièrement complets, 84 paires et deux positifs ; 54 réponses toutes vides, F1 global 0 dans les trois vues/deux calendriers. Ce sous-ensemble n'est pas représentatif et ne répare pas le primaire. La relance high est active dans l'instantané ; aucun résultat final ne lui est attribué.

Exécutions de référence : [run 09873773](../results/inventaire-executions.csv) (`09873773-2908-459d-bced-dc436272fb32`) [run 2d6766a2](../results/inventaire-executions.csv) (`2d6766a2-3d10-4a32-a19f-e8009e35efbc`) [run e163b810](../results/inventaire-executions.csv) (`e163b810-e9f7-419d-b769-168265cf4c49`)

Code ciblé : [pipelines/jev/n13.py](https://github.com/Fireman4740/workere/blob/7c07872afbc1e511193d3b3dbbae61f8d4097b13/pipelines/jev/n13.py) [experiments/jev/n13/test_n13.py](https://github.com/Fireman4740/workere/blob/7c07872afbc1e511193d3b3dbbae61f8d4097b13/experiments/jev/n13/test_n13.py) [experiments/jev/n13/ctb50/protocol.json](https://github.com/Fireman4740/workere/blob/7c07872afbc1e511193d3b3dbbae61f8d4097b13/experiments/jev/n13/ctb50/protocol.json)

Sources détaillées : [bilan-lancement-2026-10-31.md](../SOURCES.md#annexes-locales) · [bilan-final.md](../SOURCES.md#annexes-locales) · [bilan.md](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 63.


<a id="fiche-64"></a>

#### 64. Qwen CTB seul : terminer la baseline avant ESC

**Actualisation du 12 novembre.** Le dernier run est enregistré en annulation, sans métrique finale complète validée. Le statut actif du 3 novembre ci-dessous est historique ; il n’est plus l’état courant. Aucun score CTB183 n’est déduit du suivi. [run b41468ee](../results/inventaire-executions.csv) (`b41468ee-d81e-4d4c-9df7-1f51309dbde9`)

Expérience `03e141a3-c891-4cbc-95f1-eba83ea3685a`.

**Implémentation et intervention.** Arrêter le contrôleur mixte et ne reprendre que r7_ctb, avec xhigh, température 0, sept lectures et paramètres de la baseline inchangés. Réutiliser uniquement checkpoints et passes dont les corps/paramètres concordent exactement. ESC reste suspendu.

**Protocole d’évaluation.** Objectif CTB183 complet. À la reprise, 173 documents disposent d'un checkpoint réussi, et les unités incomplètes ont des passes réutilisables. Ce compte provient de l'audit de reprise ; il ne vaut pas score final CTB183.

**Tests et contrôles.** 28 tests rapportés et 1 182 prompts source vérifiés identiques. Le garde de reprise autorise un sous-ensemble des datasets sans changer l'identité du corpus, du modèle ou de l'échantillonnage. Annulations liées aux redémarrages de service documentées sans les compter comme nouveaux résultats.

**Résultat et portée.** Run encore actif dans l'instantané ; pas de F1 final complet annoncé. Le journal contient la configuration effective et les reprises, mais un statut running ne démontre pas la qualité ni la complétion.

Exécutions de référence : [run b41468ee](../results/inventaire-executions.csv) (`b41468ee-d81e-4d4c-9df7-1f51309dbde9`)

Code ciblé : [experiments/jev/qwen/run.py](https://github.com/Fireman4740/workere/blob/f80a1a4678dde0105a329cb63e200ff6b1c2eed5/experiments/jev/qwen/run.py)

Sources détaillées : [ctb-only-continuation-2026-11-01.json](../SOURCES.md#annexes-locales).

[Fichiers de test et fonctions de contrôle](../SOURCES.md#annexes-locales) — entrée 64.


### H. Extensions de novembre : validité, mécanismes, votes et interventions

<a id="fiche-65"></a>

#### 65. N14 — consigne nettoyée et exécution bornée, factoriel

Expérience `91c2b8fe-5c86-4644-b1d9-ff6354054392`.

**Intervention.** Croiser ancienne/nouvelle consigne et lecture documentaire/blocs explicites, avec trois représentations et deux calendriers de trois lectures. Niveau medium vérifié dans le tokenizer avant génération.

**Panel et comparaison.** Quatre documents CTB déjà exposés ; 593 candidats non orientés ; maximum 1 332 appels.

**Résultat et portée.** Le facteur de validité empêche une conclusion primaire complète sur les quatre documents. Le diagnostic D0 ne conserve que trois documents, 84 candidats et deux positifs. Il ne permet aucun classement généralisable des quatre conditions.

**Contrôles et limites.** Ne pas transformer les sorties incomplètes en négatifs. Les conditions changent des ensembles de composants, donc ne séparent pas un effet unique. Le sous-ensemble complet est explicitement rétrospectif.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 208d3236](../results/inventaire-executions.csv) (`208d3236-fa28-438c-be67-62e6552f1495`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes1345, 1346, 1347, 1348, 2695, SHA-256 `ce1d129c09fb6d146db98912b018d03fc5601c00b0f1a8b052fc6f24db8a04a1` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [experiments/jev/n14/demo-bank.json](https://github.com/Fireman4740/workere/blob/93b2a3357cfeda1d128538284205bf2f07fd57ed/experiments/jev/n14/demo-bank.json) [experiments/jev/n14/demo-public.json](https://github.com/Fireman4740/workere/blob/93b2a3357cfeda1d128538284205bf2f07fd57ed/experiments/jev/n14/demo-public.json) [experiments/jev/n14/gold.json](https://github.com/Fireman4740/workere/blob/93b2a3357cfeda1d128538284205bf2f07fd57ed/experiments/jev/n14/gold.json) [experiments/jev/n14/prepare.py](https://github.com/Fireman4740/workere/blob/93b2a3357cfeda1d128538284205bf2f07fd57ed/experiments/jev/n14/prepare.py) [experiments/jev/n14/protocol.json](https://github.com/Fireman4740/workere/blob/93b2a3357cfeda1d128538284205bf2f07fd57ed/experiments/jev/n14/protocol.json) [experiments/jev/n14/public.json](https://github.com/Fireman4740/workere/blob/93b2a3357cfeda1d128538284205bf2f07fd57ed/experiments/jev/n14/public.json).

Documents associés : [bilan-final.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-66"></a>

#### 66. S1 — directions hint/demo et doses à la couche 32

Expérience `c7f61538-c12f-4ef7-9716-ced320116450`.

**Intervention.** Extraire des différences moyennes d’activations au dernier token du prompt, puis ajouter la direction à la couche 32 et pendant la génération ; comparer au contrôle sans intervention.

**Panel et comparaison.** Dix documents CTB ; une lecture greedy par condition ; doses 0,05, 0,10 et 0,20.

**Résultat et portée.** Les textes de réflexion changent, mais les relations finales des documents communément scorés restent généralement identiques. L’exception hint 0,20 est défavorable. Aucun bénéfice spécifique démontré.

**Contrôles et limites.** Les troncatures font varier les sous-ensembles scorés ; les F1 historiques sur documents complets seulement ne doivent pas servir de classement. Les directions sont des contrastes de prompts, pas des contrastes de décisions correctes et incorrectes.

**État des exécutions à l’instantané.** done :1, failed :2. Référence de lecture : [run c3479053](../results/inventaire-executions.csv) (`c3479053-51cd-4ff1-b50e-6334eee13e9c`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/a29233e9b7d5a7ea87e79ec0d769af4abd3983f6/steering/run_pilot.sh).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-67"></a>

#### 67. N15 — contrat de génération court : medium ou sans réflexion

Expérience `cb948db6-5d40-4566-a69b-a2fa72b449a9`.

**Intervention.** Même Qwen, trois lectures dans chacune des deux conditions ; plafond et parseur explicites, requêtes séquentielles, sans relance silencieuse.

**Panel et comparaison.** Six documents CTB délibérément déjà exposés ; 36 réponses attendues.

**Résultat et portée.** 36/36 réponses valides. Medium : TP6/FP5/FN6, F1 52,17 ; sans réflexion : TP5/FP3/FN7, F1 50,00. Temps cumulé des requêtes 859,26 contre 415,84 secondes ; temps du contrôleur 1 275,48 secondes.

**Contrôles et limites.** Recomptage indépendant des matrices conforme au journal. Petit panel : ni supériorité ni équivalence établies. Les temps cumulés de requêtes ne sont pas une vitesse de déploiement garantie.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 1d1a1daf](../results/inventaire-executions.csv) (`1d1a1daf-1388-45de-87c2-4fdd756e10b2`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes6, 43, 44, 87, SHA-256 `5fa36c6e0a9e1d06899767c7032b997d48da4660c11f801bc2153832246ce87a` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [experiments/jev/n15/demo-bank.json](https://github.com/Fireman4740/workere/blob/a3242f7ad3015333dc8f445857c4f274a9b1f4a8/experiments/jev/n15/demo-bank.json) [experiments/jev/n15/gold.json](https://github.com/Fireman4740/workere/blob/a3242f7ad3015333dc8f445857c4f274a9b1f4a8/experiments/jev/n15/gold.json) [experiments/jev/n15/historical-context.json](https://github.com/Fireman4740/workere/blob/a3242f7ad3015333dc8f445857c4f274a9b1f4a8/experiments/jev/n15/historical-context.json) [experiments/jev/n15/protocol.json](https://github.com/Fireman4740/workere/blob/a3242f7ad3015333dc8f445857c4f274a9b1f4a8/experiments/jev/n15/protocol.json) [experiments/jev/n15/public.json](https://github.com/Fireman4740/workere/blob/a3242f7ad3015333dc8f445857c4f274a9b1f4a8/experiments/jev/n15/public.json) [experiments/jev/n15/test_n15.py](https://github.com/Fireman4740/workere/blob/a3242f7ad3015333dc8f445857c4f274a9b1f4a8/experiments/jev/n15/test_n15.py).

Documents associés : [bilan-implementation-et-pilotes.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-68"></a>

#### 68. D0/E6 — MBR exact sur les caches N14

Expérience `a2b83fcb-c3f9-4b1a-b8f7-91d0cfe95d21`.

**Intervention.** Rejouer les hypothèses sauvegardées et comparer majorité, MBR par lot, MBR documentaire et variante à cardinalité neutralisée, sans génération.

**Panel et comparaison.** Intersection de trois documents CTB, 84 candidats et deux positifs ; quatrième document exclu pour couverture invalide.

**Résultat et portée.** Les systèmes sur cette intersection obtiennent F1 0. Le résultat est un diagnostic sur un sous-ensemble exposé très pauvre en positifs, pas un score de CTB.

**Contrôles et limites.** 72 recomptages documentaires des contrôles concordent avec les sorties sources. Aucun seuil choisi sur le gold, aucune inférence nouvelle.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run ed0965f8](../results/inventaire-executions.csv) (`ed0965f8-3e13-4bc0-b374-4ad519d65638`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes6, SHA-256 `9398773133fab3ff709d484d0168308f0e66a7f63854bcf863863142040407ac` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [experiments/jev/research/cached-controls.json](https://github.com/Fireman4740/workere/blob/7113ba2d5fc969abc8ad6dbb88f64f2314d7b5bf/experiments/jev/research/cached-controls.json) [experiments/jev/research/cached-gold.json](https://github.com/Fireman4740/workere/blob/7113ba2d5fc969abc8ad6dbb88f64f2314d7b5bf/experiments/jev/research/cached-gold.json) [experiments/jev/research/cached-protocol.json](https://github.com/Fireman4740/workere/blob/7113ba2d5fc969abc8ad6dbb88f64f2314d7b5bf/experiments/jev/research/cached-protocol.json) [experiments/jev/research/cached-readings.json](https://github.com/Fireman4740/workere/blob/7113ba2d5fc969abc8ad6dbb88f64f2314d7b5bf/experiments/jev/research/cached-readings.json) [experiments/jev/run.sh](https://github.com/Fireman4740/workere/blob/7113ba2d5fc969abc8ad6dbb88f64f2314d7b5bf/experiments/jev/run.sh) [pipelines/jev/research/__init__.py](https://github.com/Fireman4740/workere/blob/7113ba2d5fc969abc8ad6dbb88f64f2314d7b5bf/pipelines/jev/research/__init__.py).

Documents associés : [bilan-implementation-et-pilotes.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-69"></a>

#### 69. D0 — union des lectures brutes N14

Expérience `95cb5375-0798-43bb-a8f9-0da52fc85681`.

**Intervention.** Élargir le diagnostic précédent à l’union de toutes les hypothèses brutes plutôt qu’aux seuls votes agrégés.

**Panel et comparaison.** Même intersection de trois documents, 84 candidats et deux liens gold.

**Résultat et portée.** Les 72 lectures ne proposent aucun des deux liens gold : rappel de cette union égal à zéro. La limite concerne uniquement ce cache et ce panel.

**Contrôles et limites.** Reprise des 72 contrôles exacts du parent ; aucune génération. Aucun plafond de capacité générale du modèle n’est déduit de cette observation.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run e1b0a948](../results/inventaire-executions.csv) (`e1b0a948-8197-4a71-a69a-e0c69f8d8ef7`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes6, SHA-256 `956462a7a00552ffc741d899eb6c815ebd459c163580a26fb88ba2d202edffb8` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [pipelines/jev/research/cached_study.py](https://github.com/Fireman4740/workere/blob/8d95f6930dfad370479f54f37071cc15d0fe5d01/pipelines/jev/research/cached_study.py).

Documents associés : [bilan-implementation-et-pilotes.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-70"></a>

#### 70. S2-A — augmenter la dose demo

Expérience `d8d2016a-3035-41d4-b088-f68c09f510ad`.

**Intervention.** À la couche 32, comparer contrôle et doses demo 0,3/0,6/1,0, correspondant à environ une à trois fois le déplacement naturel mesuré.

**Panel et comparaison.** Douze documents CTB riches en positifs, déjà exposés.

**Résultat et portée.** Les deltas historiques sur documents communs sont −5,1 / +5,4 / −4,8 points. Le gain central à 0,6 dépend d’un seul document ; aucune relation dose–effet convaincante.

**Contrôles et limites.** Trois ou quatre documents tronqués par condition ; les valeurs sur intersections sont exploratoires et ne sont pas des résultats sur les douze documents entiers.

**État des exécutions à l’instantané.** done :1, failed :1. Référence de lecture : [run 8e62c273](../results/inventaire-executions.csv) (`8e62c273-20b8-406e-acf3-0bc607d811ed`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/77b8950ec4d78940fb68bcb9a7c84fc798c87d4b/steering/run_pilot.sh).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-71"></a>

#### 71. S2-B — direction opposée et contrôle aléatoire

Expérience `dd20aa01-9cd5-40da-8f3d-0a92eadc6eab`.

**Intervention.** Comparer hint 0,2, son opposé et une direction aléatoire de même norme au contrôle de couche 32.

**Panel et comparaison.** Même panel CTB de douze documents.

**Résultat et portée.** Deltas sur les documents communs : −3,7 / −6,3 / −2,1 points. Le changement de signe ne renverse pas l’effet et le contrôle aléatoire perturbe aussi les réponses ; pas de spécificité démontrée.

**Contrôles et limites.** Intervalles comprenant zéro ; troncatures et faible nombre de liens. Ces témoins ne prouvent pas que toute perturbation équivalente produit toujours le même effet.

**État des exécutions à l’instantané.** done :1, failed :1. Référence de lecture : [run 8f68fd57](../results/inventaire-executions.csv) (`8f68fd57-5f0e-43e1-9e8d-0715813dd14a`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/9843a6ca2d163bd9f34bb5de4d23b7b145a8bb6a/steering/run_pilot.sh).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-72"></a>

#### 72. S2-C — transfert de l’intervention à la couche 40

Expérience `ed88134b-104e-4742-9387-50d9aaab6771`.

**Intervention.** Tester hint 0,13 et demo 0,4/0,8 avec le même type d’intervention, mais à la couche 40.

**Panel et comparaison.** Douze documents CTB, comparaisons avec le contrôle propre au run.

**Résultat et portée.** Les deux doses demo gardent les mêmes décisions que le contrôle sur l’intersection valide ; hint ne fournit pas de gain. Aucun nouveau système retenu.

**Contrôles et limites.** Les sous-ensembles valides dépendent des troncatures ; identité locale des sorties ne démontre pas une équivalence sur CTB complet.

**État des exécutions à l’instantané.** done :1, failed :1. Référence de lecture : [run 506226ed](../results/inventaire-executions.csv) (`506226ed-8856-4530-b8c0-752110893f87`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/0c69b299f62cc7ef65423692fccd57d18d52b59a/steering/run_pilot.sh).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-73"></a>

#### 73. D0/E6 — MBR exact sur N15 complet

Expérience `1fb33142-09a7-4880-9439-3a3fe8e64816`.

**Intervention.** Reconstituer les 36 lectures N15, figer les règles de majorité/MBR avant évaluation et mesurer le rappel de l’union brute.

**Panel et comparaison.** Six documents, 189 candidats, douze positifs.

**Résultat et portée.** MBR par lot et MBR à cardinalité neutralisée reproduisent la majorité. MBR documentaire : medium 52,17, sans réflexion 48,00. Union brute : 7/12 positifs, rappel 58,33 ; cinq liens ne sont jamais proposés.

**Contrôles et limites.** Douze matrices documentaires reproduites exactement. Le plafond oracle 73,68 utilise le gold et n’est pas un score déployable. Aucun gain démontré du MBR.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 309d905d](../results/inventaire-executions.csv) (`309d905d-f4c5-417d-b226-deb00f8279fc`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes6, SHA-256 `901cb4ba572e6d400c9ea19e9eb71406f10bd4a572f4bbf9bb17fdb7fe274b58` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : `experiments/jev/research/cached-controls.json` (expérience `1fb33142-09a7-4880-9439-3a3fe8e64816`) `experiments/jev/research/cached-gold.json` (expérience `1fb33142-09a7-4880-9439-3a3fe8e64816`) `experiments/jev/research/cached-protocol.json` (expérience `1fb33142-09a7-4880-9439-3a3fe8e64816`) `experiments/jev/research/cached-readings.json` (expérience `1fb33142-09a7-4880-9439-3a3fe8e64816`) `experiments/jev/run.sh` (expérience `1fb33142-09a7-4880-9439-3a3fe8e64816`) `pipelines/jev/research/__init__.py` (expérience `1fb33142-09a7-4880-9439-3a3fe8e64816`).

Documents associés : [bilan-implementation-et-pilotes.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-74"></a>

#### 74. E1 — décodage, boucles et fermeture à 8k

Expérience `9350744f-920c-4f1d-9de8-32a9fc5f5fca`.

**Intervention.** Comparer greedy, température 0,6 et réflexion bornée à 8 192 tokens suivie d’une réponse libre.

**Panel et comparaison.** 47 documents CTB ; une lecture par condition.

**Résultat et portée.** Le replay strict sur les 47 documents donne contrôle F1 40,38, température 44,04 et budget 8k 50,00. Les comptes publiés dans l’audit sont respectivement42/26/98,48/30/92 et68/64/72. Le budget fournit des réponses finales pour tous les documents, sans supériorité confirmée.

**Contrôles et limites.** Les anciens nombres lisaient parfois des fragments JSON de la réflexion tronquée. Seuls les résultats après le dernier </think> et le recomptage corrigé sont utilisés. Gain de qualité non confirmé ; fermeture forcée distincte d’une réponse finale absente.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run a5f034d1](../results/inventaire-executions.csv) (`a5f034d1-5e13-4d2f-9b7a-3daf00cafe79`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/41f2bf4a2a12a6cbd904294430718222514f4424/steering/config.env).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales) ; [audit-scores-2026-11-06.json](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-75"></a>

#### 75. E2 — arrêt anticipé par continuations en accord

Expérience `3a33f02f-2843-44e4-a737-8b8bbeca2755`.

**Intervention.** Aux points de contrôle de réflexion, générer trois continuations et arrêter si un accord stable satisfait la règle fixée.

**Panel et comparaison.** 47 documents CTB, avec contrôle dans le même run.

**Résultat et portée.** Le replay strict sur 47 documents donne contrôle33,33 (TP34/FP30/FN106) et arrêt anticipé32,69 (34/34/106). L’identité des scores rapportée sur une ancienne intersection ne vaut donc pas pour le panel entier. Aucun gain de tâche établi.

**Contrôles et limites.** Les anciennes métriques sur sous-ensembles complets ne remplacent pas le recomptage strict à 47 documents. Le surcoût des continuations doit entrer dans une mesure d’économie.

**État des exécutions à l’instantané.** done :1, failed :1. Référence de lecture : [run 5ce8a7ea](../results/inventaire-executions.csv) (`5ce8a7ea-342a-40ca-814b-01de968890a9`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/early_exit.py](https://github.com/Fireman4740/workere/blob/4b964379b49db7b3bef35629ea0ec29b052d1625/steering/early_exit.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales) ; [audit-scores-2026-11-06.json](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-76"></a>

#### 76. E3 — retirer une composante d’hésitation

Expérience `c1a25575-2ae4-4a6d-b9cb-39bef8832f7b`.

**Intervention.** Extraire un contraste entre débuts de phrases hésitantes et décisives ; retirer sa projection positive à deux intensités, avec direction aléatoire témoin.

**Panel et comparaison.** 47 documents CTB ; extraction hors panel, couche choisie sur séparation tenue à part.

**Résultat et portée.** Le replay strict donne contrôle35,29 ; intervention0,5 :33,66 ; intervention1,0 :25,26 ; contrôle aléatoire 32,99. Aucun bénéfice spécifique démontré ; les traces ne sont pas systématiquement raccourcies.

**Contrôles et limites.** La décodabilité d’un marqueur de style n’identifie pas une variable causale de décision. Les scores bruts du premier parseur sont remplacés par l’audit strict.

**État des exécutions à l’instantané.** done :1, failed :1. Référence de lecture : [run 865d312c](../results/inventaire-executions.csv) (`865d312c-0ad3-4d0a-a46e-d3964470826f`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/bf0c427adbb42b7146f4226ddd6432a6d3647b95/steering/config.env).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales) ; [audit-scores-2026-11-06.json](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-77"></a>

#### 77. E4 — sonde avant et après réflexion

Expérience `0d026a92-a11d-4f27-9fb1-6f4f5ff64286`.

**Intervention.** Entraîner une régression logistique sur les états résiduels de paires ; sélectionner couche, régularisation et seuil par validation groupée par document.

**Panel et comparaison.** 133 documents d’entraînement prévus ; avant réflexion : 47 évalués. Après réflexion : 106 train et 38 évalués avec traces complètes compatibles.

**Résultat et portée.** Avant réflexion : F1 34,88 contre Qwen7 58,91. Après : sonde 51,61 contre Qwen7 44,44 sur les mêmes 38 documents, delta +7,17 [−4,29 ; +18,97].

**Contrôles et limites.** Cette sonde apprend sur annotations : elle ne relève pas du strict sans entraînement. Neuf documents sont exclus du bras après réflexion ; comparaison avant/après confondue par le changement de panel. Pas de supériorité confirmée.

**État des exécutions à l’instantané.** done :1, failed :1. Référence de lecture : [run 3aef44d2](../results/inventaire-executions.csv) (`3aef44d2-eb82-41a1-bef0-f7b14a99482f`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/exp/run_mech.sh](https://github.com/Fireman4740/workere/blob/88a915e3dea588f2adae61c6899746ff9ac44ea9/steering/exp/run_mech.sh).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-78"></a>

#### 78. E5 — éditer des têtes associées à l’hésitation

Expérience `7db6a2ad-d29f-453b-8743-b5739aac2d0b`.

**Intervention.** Retirer la composante d’hésitation dans quinze têtes sélectionnées, avec le même nombre de têtes aléatoires comme témoin.

**Panel et comparaison.** 47 documents CTB ; réponses finales seules dans l’évaluation corrigée.

**Résultat et portée.** Replay strict : contrôle25,53, édition 33,33, contrôle aléatoire 35,64. L’intervention ne dépasse pas le témoin aléatoire en valeur centrale et aucun avantage spécifique n’est établi.

**Contrôles et limites.** La sélection concerne des marqueurs de sortie et ne démontre pas un mécanisme de causalité. Les anciens scores qui exploitaient la réflexion tronquée ne sont pas retenus.

**État des exécutions à l’instantané.** done :1, failed :2. Référence de lecture : [run 25b64a47](../results/inventaire-executions.csv) (`25b64a47-33b5-44b8-bdb4-1b189df73bfb`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/reflection.py](https://github.com/Fireman4740/workere/blob/d1467884767bd975002be406159c43a85d9fe18b/steering/reflection.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales) ; [audit-scores-2026-11-06.json](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-79"></a>

#### 79. R1-Qwen — cinq lectures échantillonnées et sonde

Expérience `5b3cd16c-e39d-4875-ba63-c924445af910`.

**Intervention.** Cinq passes à température 0,6, même démonstration, budget 8k+4k ; voter et appliquer une sonde après réflexion.

**Panel et comparaison.** CTB47 ; témoin Qwen7 58,91.

**Résultat et portée.** Vote cinq passes : 55,64 ; sonde moyennée : 40,00 malgré AUROC 0,890. La tête du vérificateur prévue n’est pas trouvée : vérificateur et fusion ne sont pas évalués.

**Contrôles et limites.** Zéro réponse finale manquante. Une tête absente est un problème technique, pas un résultat scientifique négatif du vérificateur. L’AUROC ne garantit pas le F1 au seuil fixé.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run dd5cebc4](../results/inventaire-executions.csv) (`dd5cebc4-f7de-400d-b1d4-75b2372ad530`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/782d2574a25c7d457a5f9fe7fe8375ebec92c192/steering/config.env).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-80"></a>

#### 80. R2-Qwen — vote documentaire sur ESC40

Expérience `99bd998e-0196-4167-bf0f-4fab17154dcd`.

**Intervention.** Trois lectures échantillonnées, démonstrations d’autres sujets, budget 10k de réflexion et 5k de réponse.

**Panel et comparaison.** TEST-S40 ; comparaisons appariées à Luna7 et à la fusion historique.

**Résultat et portée.** F1 brut 41,81, contre Luna7 46,02 et fusion 48,64. La propagation ECB+ oracle donne 49,08 ; elle reste hors résultat autonome.

**Contrôles et limites.** 92/120 réflexions atteignent le budget ; une réponse finale manque. Le budget est une hypothèse explicative à tester, pas une cause prouvée de l’écart.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 6ef55acc](../results/inventaire-executions.csv) (`6ef55acc-155e-4b5a-873b-48c878858e15`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes719, 727, 728, 729, 730, SHA-256 `a4426d5676c747ccdf78693a3194ffa192197d4cbedc48c2cfd3717bbebf0aac` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/43cff9d6500a7aef86d6e4a4ad014bcff9dc9638/steering/config.env).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-81"></a>

#### 81. R3-Qwen — élargir la consigne aux buts et à l’habilitation

Expérience `7074adf9-5749-4e48-b08e-682d400d6c3c`.

**Intervention.** Changer deux règles de la consigne pour autoriser davantage de liens de but/habilitation, sans changer les démonstrations.

**Panel et comparaison.** CTB47 ; premier passage de trois prévus.

**Résultat et portée.** Premier passage F1 29,46, TP66/FP242/FN74. Essai arrêté après ce passage ; pas de résultat de vote à trois passes.

**Contrôles et limites.** Le run est annulé mais apporte un signal négatif réel sur cette formulation. Ne pas transformer cet arrêt en preuve que toute adaptation de consigne échoue.

**État des exécutions à l’instantané.** Cancelled :1. Référence de lecture : [run fdfdbcc2](../results/inventaire-executions.csv) (`fdfdbcc2-5a14-4ba0-9e1c-17c9ccfa8593`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/face2a940e7be5c78a5a3c46170ace1b51a4923b/steering/config.env) [steering/exp/run_mech.sh](https://github.com/Fireman4740/workere/blob/face2a940e7be5c78a5a3c46170ace1b51a4923b/steering/exp/run_mech.sh) [steering/guideline.py](https://github.com/Fireman4740/workere/blob/face2a940e7be5c78a5a3c46170ace1b51a4923b/steering/guideline.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-82"></a>

#### 82. N1-Qwen — diversité des démonstrations

Expérience `a21a2e95-cf0a-4046-b10e-d935a7588c29`.

**Intervention.** Cinq lectures greedy avec cinq démonstrations différentes, budget 8k+4k ; comparaison aux cinq passes échantillonnées R1.

**Panel et comparaison.** CTB47 ; mêmes candidats et témoins.

**Résultat et portée.** Vote F1 59,70 ; contre R1 55,64 : +4,06 [−1,94 ; +10,89]. Contre Qwen7 : +0,79 [−4,04 ; +6,77]. Le vote R1+N1 à dix lectures donne 57,14.

**Contrôles et limites.** Zéro réponse finale manquante. La diversité des démonstrations est un candidat favorable en valeur centrale, sans supériorité statistique démontrée sur ces témoins.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 9876c9f5](../results/inventaire-executions.csv) (`9876c9f5-35ae-4b15-b691-3d377cac16b2`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/d5060c5dff21e83987b523fbaa124773a58269da/steering/config.env) [steering/exp/run_mech.sh](https://github.com/Fireman4740/workere/blob/d5060c5dff21e83987b523fbaa124773a58269da/steering/exp/run_mech.sh) [steering/usc.py](https://github.com/Fireman4740/workere/blob/d5060c5dff21e83987b523fbaa124773a58269da/steering/usc.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-83"></a>

#### 83. N2-Qwen — relecture des propositions R1

Expérience `72c9b075-22e3-4ed2-9330-156ca31e531a`.

**Intervention.** Montrer au même modèle les propositions et comptes de vote des cinq lectures R1 et lui demander une vérification documentaire.

**Panel et comparaison.** CTB47 ; six générations locales au total avec les cinq lectures sources.

**Résultat et portée.** F1 60,15 ; delta +4,51 [−1,28 ; +12,80] contre ses entrées R1. Signal central favorable mais incertain, suivi par P1 et P4.

**Contrôles et limites.** La lecture du vote peut ancrer le vérificateur. Les expériences ultérieures ne reproduisent pas un bénéfice stable : cette valeur n’est pas une amélioration acquise.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run e1f30751](../results/inventaire-executions.csv) (`e1f30751-ad2b-4aeb-acde-37cc7d690781`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/80649a7db8a3c38ea459d3ef73781d8d09601a8c/steering/config.env) [steering/exp/run_mech.sh](https://github.com/Fireman4740/workere/blob/80649a7db8a3c38ea459d3ef73781d8d09601a8c/steering/exp/run_mech.sh) [steering/usc.py](https://github.com/Fireman4740/workere/blob/80649a7db8a3c38ea459d3ef73781d8d09601a8c/steering/usc.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-84"></a>

#### 84. P1-Qwen — même relecture sur N1

Expérience `f250f017-471a-4e0b-a9ea-cd6f9ddbdca4`.

**Intervention.** Reprendre la vérification documentaire N2, cette fois sur les cinq lectures N1 à démonstrations diverses.

**Panel et comparaison.** CTB47 ; comparaison au vote N1 59,70.

**Résultat et portée.** Le replay strict structuré donne F1 56,49, TP74/FP48/FN66, soit −3,21 points contre N1 59,70. Le score historique 56,06 (TP74/FP50/FN66) lisait le texte complet. Son intervalle −3,64 [−12,36 ; +2,36] contre N1 n’est pas transposé au recomptage strict. Le gain ponctuel de N2 ne se reproduit pas.

**Contrôles et limites.** 14/47 réflexions fermées au budget. La différence entre sources et vérification ne suffit pas à attribuer une cause unique au changement.

**État des exécutions à l’instantané.** done :1, failed :2. Référence de lecture : [run f5d5f69e](../results/inventaire-executions.csv) (`f5d5f69e-1c35-455f-80bd-9cf89f600071`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/offline.py](https://github.com/Fireman4740/workere/blob/febf932b371265271311ca727676e62b78eaac7e/steering/offline.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-85"></a>

#### 85. P3 — augmenter la réflexion CTB à 20k

Expérience `cd47961f-f03b-42a4-a7ad-cba131dc7061`.

**Intervention.** Même groupe de cinq démonstrations que N1, budget 20 480+4 096, contexte 32k et préremplissage découpé ; voter les lectures et leur réunion avec N1.

**Panel et comparaison.** CTB47 ; référence Qwen7 58,91.

**Résultat et portée.** P3 seul 57,14. Réunion N1+P3, dix lectures : 61,19, TP82/FP46/FN58 ; delta contre Qwen7 +2,28 [−2,09 ; +8,44]. Les fermetures passent de 82/235 à 33/235.

**Contrôles et limites.** Meilleure référence Qwen conservée sur ce panel, sans supériorité confirmée contre Qwen7. Budget et configuration du moteur changent ensemble ; le gain ne peut être attribué au seul raisonnement plus long.

**État des exécutions à l’instantané.** done :1, failed :2. Référence de lecture : [run 3d8a86a3](../results/inventaire-executions.csv) (`3d8a86a3-8b79-4aab-acaa-96b5ddbb6f82`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/offline.py](https://github.com/Fireman4740/workere/blob/c69d2c1dc00988eed78436b7772d1ef3a4880628/steering/offline.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-86"></a>

#### 86. P2 — réflexion ESC à 24k

Expérience `63c47625-338f-4fab-8903-ebb9b26a6229`.

**Intervention.** Même recette R2, mais 24 576 tokens de réflexion, 8 192 de réponse et contexte 40k.

**Panel et comparaison.** ESC TEST-S40 entier ; 120 lectures.

**Résultat et portée.** F1 brut 42,38 ; delta contre R2 +0,58 [−3,64 ; +5,29]. Quatre réponses manquantes ; 44 fermetures au lieu de 92. Le score reste sous la fusion 48,64.

**Contrôles et limites.** Le résultat oracle +coref 49,78 est séparé du système autonome. L’augmentation de budget testée n’apporte pas de gain F1 démontré ; cela n’exclut pas d’autres limites de calcul.

**État des exécutions à l’instantané.** done :1, failed :2. Référence de lecture : [run 42e7e53b](../results/inventaire-executions.csv) (`42e7e53b-3529-4fda-a1e9-506fe34b732c`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes801, 809, 810, 811, 812, SHA-256 `027b14d1119184187f892dc32fc95c60bbb562914a120f32eef926460b198441` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/offline.py](https://github.com/Fireman4740/workere/blob/de67539361377256a24f56d6b055221ab13524a9/steering/offline.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-87"></a>

#### 87. P4 — vérifier les longues lectures P3

Expérience `42192e53-5368-4572-a18e-a7be80b7afe3`.

**Intervention.** Une relecture documentaire à budget 20k sur les cinq propositions P3.

**Panel et comparaison.** CTB47 ; même univers que P3 et N1+P3.

**Résultat et portée.** F1 57,14, mêmes comptes TP76/FP50/FN64 que le vote P3. Trois réflexions fermées et un document sans réponse.

**Contrôles et limites.** L’égalité des comptes agrégés ne prouve pas l’identité de toutes les paires. Aucun bénéfice F1 de cette relecture démontré.

**État des exécutions à l’instantané.** done :1, Cancelled :1, failed :1. Référence de lecture : [run b2dc2418](../results/inventaire-executions.csv) (`b2dc2418-a64f-498f-b116-4a095360b975`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/offline.py](https://github.com/Fireman4740/workere/blob/42a273d61d28a1ebf942e4ec4753df9eba4062fc/steering/offline.py).

Documents associés : [revue-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-88"></a>

#### 88. V1 — vérificateur dense par probabilités natives

Expérience `9676b3f8-704c-4791-a07f-ed0acad37bd3`.

**Intervention.** Scorer toutes les paires avec labels permutés, score brut/centré et fusions fixes avec N1/P3 ; seuil scalaire choisi hors évaluation.

**Panel et comparaison.** 113 documents de calibration ; CTB47 évalué, toutes les paires.

**Résultat et portée.** Score brut : F1 29,37 ; hybride N1+P3 : 44,97 contre vote 61,19. Le score retrouve trois vrais liens non orientés à zéro vote, pour 31 faux.

**Contrôles et limites.** Le run final est un replay court des scores déjà calculés ; sa durée n’est pas le coût d’inférence complet. Aucun apprentissage de réseau, mais sélection d’un seuil sur annotations.

**État des exécutions à l’instantané.** done :1, failed :2. Référence de lecture : [run c2a7c92c](../results/inventaire-executions.csv) (`c2a7c92c-7bf3-496d-b239-d343bc6140e2`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes7, 8, 9, 10, 11, SHA-256 `442e78a702099bb8ce6b93b43705b7975667f36b666e75e5445311e4858b91e5` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/pair_verifier.py](https://github.com/Fireman4740/workere/blob/06f5083493c36bea595bfc160e51b21792c9fdaf/steering/pair_verifier.py) [steering/test_pair_verifier.py](https://github.com/Fireman4740/workere/blob/06f5083493c36bea595bfc160e51b21792c9fdaf/steering/test_pair_verifier.py).

Documents associés : [suite-verificateurs-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-89"></a>

#### 89. V2 — vérificateur avec exemples positifs et négatifs récupérés

Expérience `6a372e0e-6eb6-4189-830a-3e19f677b30f`.

**Intervention.** Même vérificateur dense, enrichi de deux exemples positifs et deux négatifs par récupération lexicale et structurelle.

**Panel et comparaison.** 23 958 requêtes ; CTB47 entier, banque de démonstrations disjointe de la calibration et de l’évaluation.

**Résultat et portée.** F1 brut 29,58 ; centré 30,43. Meilleur hybride centré 47,27 contre N1+P3 61,19 : −13,92 [−20,49 ; −5,31].

**Contrôles et limites.** Tous les documents évalués. Le classement AUROC élevé ne fournit pas une précision suffisante sur les liens rares ; la récupération de liens à zéro vote reste accompagnée de nombreux faux positifs.

**État des exécutions à l’instantané.** done :1, Cancelled :1. Référence de lecture : [run 682c9c50](../results/inventaire-executions.csv) (`682c9c50-aa02-4611-a125-39ccd5893626`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes24554, 24555, 24556, 24557, 24558, SHA-256 `07bfc80d8ff16d170ec6a59f6607c8ae462b5843e2ced90a3272a4272ac0e222` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/pair_verifier.py](https://github.com/Fireman4740/workere/blob/c4568ae80322b189a0b5db08dc87dddcb45541c3/steering/pair_verifier.py) [steering/test_pair_verifier.py](https://github.com/Fireman4740/workere/blob/c4568ae80322b189a0b5db08dc87dddcb45541c3/steering/test_pair_verifier.py).

Documents associés : [suite-verificateurs-2026-11-06.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-90"></a>

#### 90. C1 — contexte et réflexion par blocs de paires

Expérience `d1f3eb8b-1a38-4b49-b58a-1a171f7589d1`.

**Intervention.** Factoriel contexte local/document entier × sortie directe/réflexion 2 048 tokens, deux ordres, blocs exhaustifs de douze paires.

**Panel et comparaison.** CTB20 et ESC12 déjà exposés ; 2 856 requêtes.

**Résultat et portée.** CTB local : 10,42/15,38 sans réflexion, 40,00/50,85 avec ; référence documentaire 65,00. ESC local avec réflexion : 37,64/35,85, référence P2 35,50.

**Contrôles et limites.** Le contexte complet n’apporte pas de gain stable. Fort effet de l’ordre et nombreux plafonds atteints ; combinaisons examinées après résultat exploratoires.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 8140715c](../results/inventaire-executions.csv) (`8140715c-d969-4b0a-af63-1d7abcce0733`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes4555, 4556, 4557, 4558, 4568, SHA-256 `a4bee7fdd49ec56a14eae116e19f84de52a76a0ca0dab91c2926366f5ab318a9` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/campaign_blocks.py](https://github.com/Fireman4740/workere/blob/bcba516382fc94d872200753d3019075a2bae73e/steering/campaign_blocks.py) [steering/config.env](https://github.com/Fireman4740/workere/blob/bcba516382fc94d872200753d3019075a2bae73e/steering/config.env) [steering/data/esc/baseline.json](https://github.com/Fireman4740/workere/blob/bcba516382fc94d872200753d3019075a2bae73e/steering/data/esc/baseline.json) [steering/data/esc/prompt.json](https://github.com/Fireman4740/workere/blob/bcba516382fc94d872200753d3019075a2bae73e/steering/data/esc/prompt.json) [steering/data/esc/test-s40.json](https://github.com/Fireman4740/workere/blob/bcba516382fc94d872200753d3019075a2bae73e/steering/data/esc/test-s40.json) [steering/exp/run_campaign.sh](https://github.com/Fireman4740/workere/blob/bcba516382fc94d872200753d3019075a2bae73e/steering/exp/run_campaign.sh).

Documents associés : [c1-resultats-2026-11-07.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-91"></a>

#### 91. C2 — fiches d’événements et réflexion 4k

Expérience `bf3f19ed-935e-4947-ba68-bcf1e59c6cd4`.

**Intervention.** Ajouter fiches réelles, fiches permutées, budget doublé sans fiche et budget doublé avec fiches, en conservant C1 et ses deux ordres.

**Panel et comparaison.** CTB20/ESC12 ; 81 extractions de fiches valides, 2 855/2 856 classifications valides.

**Résultat et portée.** CTB sans fiche à 4k : 53,85/58,82, sous le vote 65,00. Les fiches ne battent pas de façon établie les fiches permutées. Filtrage conservateur : 66,67 contre 65,00, dû à un seul faux lien retiré.

**Contrôles et limites.** Intervalles nominaux, 32 combinaisons examinées, développement réutilisé. La validité d’une citation textuelle ne garantit pas l’attribution sémantique de la fiche. Le filtre reçoit sa validation élargie VC1.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run f275b681](../results/inventaire-executions.csv) (`f275b681-b3f3-402d-b9cd-e860d4d629d9`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes5781, 5782, 5783, 5784, 5807, SHA-256 `60b142366fbb56df670cc30b1489735ac06f9e7ba4cff791730762d0d7970db6` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/campaign_facts.py](https://github.com/Fireman4740/workere/blob/792310cc11bb5bb8c8eaa47001de5d0e690af7bb/steering/campaign_facts.py) [steering/config.env](https://github.com/Fireman4740/workere/blob/792310cc11bb5bb8c8eaa47001de5d0e690af7bb/steering/config.env) [steering/exp/run_campaign2.sh](https://github.com/Fireman4740/workere/blob/792310cc11bb5bb8c8eaa47001de5d0e690af7bb/steering/exp/run_campaign2.sh) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/792310cc11bb5bb8c8eaa47001de5d0e690af7bb/steering/run_pilot.sh) [steering/test_campaign_facts.py](https://github.com/Fireman4740/workere/blob/792310cc11bb5bb8c8eaa47001de5d0e690af7bb/steering/test_campaign_facts.py).

Documents associés : [c2-resultats-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-92"></a>

#### 92. C3 natif — arbitrage direct ou avec preuve

Expérience `966d8898-5b02-4ac3-9cc0-de9177c9b9aa`.

**Intervention.** Sur désaccords C2/vote, juger une paire à la fois, directement ou avec justification ancrée, dans deux présentations.

**Panel et comparaison.** CTB20/ESC12 ; 1 584 décisions prévues.

**Résultat et portée.** Annulé lors du passage au service API persistant ; 912 sorties complètes conservées. Aucun rapport final comparable sur tout le panel.

**Contrôles et limites.** Les sorties natives ne sont pas mélangées au frère API : précision du cache KV et configuration de service diffèrent. Une annulation partielle n’est pas un F1 nul.

**État des exécutions à l’instantané.** Cancelled :1. Référence de lecture : [run 6fc559ee](../results/inventaire-executions.csv) (`6fc559ee-808d-4b69-8bd3-fe02dd30e790`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/campaign_adjudicate.py](https://github.com/Fireman4740/workere/blob/dd84c64fa83b6d81864087c2401cfc3d742c6141/steering/campaign_adjudicate.py) [steering/config.env](https://github.com/Fireman4740/workere/blob/dd84c64fa83b6d81864087c2401cfc3d742c6141/steering/config.env) [steering/exp/run_campaign3.sh](https://github.com/Fireman4740/workere/blob/dd84c64fa83b6d81864087c2401cfc3d742c6141/steering/exp/run_campaign3.sh) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/dd84c64fa83b6d81864087c2401cfc3d742c6141/steering/run_pilot.sh) [steering/test_campaign_adjudicate.py](https://github.com/Fireman4740/workere/blob/dd84c64fa83b6d81864087c2401cfc3d742c6141/steering/test_campaign_adjudicate.py).

Documents associés : [plan-execution-qwen-api-terra-low-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-93"></a>

#### 93. C3 API — direct contre preuve textuelle

Expérience `be5c00d5-1a90-42c9-980e-0ec84231d890`.

**Intervention.** Même question scientifique via l’API batch Qwen locale, deux orientations, modification seulement si elles concordent ; sinon repli au vote.

**Panel et comparaison.** CTB20/ESC12 ; 1 584 jugements, prédictions de référence rejouées.

**Résultat et portée.** CTB : direct 54,17, preuve 59,57, référence 65,00. ESC : direct 41,05, preuve 39,69, référence 35,50 ; direct−référence +5,55 [−3,09 ; +13,68].

**Contrôles et limites.** Le primaire preuve−direct ne démontre pas de bénéfice stable. Données de développement ; service différent du natif explicitement déclaré.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 00a20ccc](../results/inventaire-executions.csv) (`00a20ccc-434d-4d4c-a11f-66fbfb86f72c`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes130, 131, 132, 133, 135, SHA-256 `ea136371722f74ed944ba821351995bda097a8ed5f5f95c972bf5d1489d4a14b` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/campaign_api.py](https://github.com/Fireman4740/workere/blob/9c0fe4789643569e7c9b2ae752195c733eb1f3aa/steering/campaign_api.py) [steering/config.env](https://github.com/Fireman4740/workere/blob/9c0fe4789643569e7c9b2ae752195c733eb1f3aa/steering/config.env) [steering/exp/run_campaign_api.sh](https://github.com/Fireman4740/workere/blob/9c0fe4789643569e7c9b2ae752195c733eb1f3aa/steering/exp/run_campaign_api.sh) [steering/inspect_api_server.py](https://github.com/Fireman4740/workere/blob/9c0fe4789643569e7c9b2ae752195c733eb1f3aa/steering/inspect_api_server.py) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/9c0fe4789643569e7c9b2ae752195c733eb1f3aa/steering/run_pilot.sh) [steering/test_campaign_api.py](https://github.com/Fireman4740/workere/blob/9c0fe4789643569e7c9b2ae752195c733eb1f3aa/steering/test_campaign_api.py).

Documents associés : [c3-api-resultats-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-94"></a>

#### 94. AP0 — replay C3 et gel de campagne

Expérience `75f3d8a3-e800-45ca-bec8-47ca9fcb77bb`.

**Intervention.** Recompter les sorties C3 et vérifier les matrices de chaque document avant de figer les règles AP1–AP4.

**Panel et comparaison.** Mêmes CTB20/ESC12 ; CPU seulement.

**Résultat et portée.** Replay exact de C3. Le jugement direct ESC est retenu comme candidat exploratoire ; CTB conserve son vote.

**Contrôles et limites.** Un replay valide la cohérence du calcul, pas la généralisation. Aucune nouvelle inférence ; conserver les coûts historiques.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 032602d3](../results/inventaire-executions.csv) (`032602d3-4313-4a89-bd6c-2532f151c1c8`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes61, 62, 63, 65, 66, SHA-256 `5a108222d9682c3b735d63f08519f704d48799897618146657aeef45cd20e6b5` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/ap_campaign.py](https://github.com/Fireman4740/workere/blob/53307fa4ae8008ab2560313763df9cb95ccabbb6/steering/ap_campaign.py) [steering/ap_protocol.py](https://github.com/Fireman4740/workere/blob/53307fa4ae8008ab2560313763df9cb95ccabbb6/steering/ap_protocol.py) [steering/config.env](https://github.com/Fireman4740/workere/blob/53307fa4ae8008ab2560313763df9cb95ccabbb6/steering/config.env) [steering/exp/run_ap.sh](https://github.com/Fireman4740/workere/blob/53307fa4ae8008ab2560313763df9cb95ccabbb6/steering/exp/run_ap.sh) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/53307fa4ae8008ab2560313763df9cb95ccabbb6/steering/run_pilot.sh) [steering/test_ap_campaign.py](https://github.com/Fireman4740/workere/blob/53307fa4ae8008ab2560313763df9cb95ccabbb6/steering/test_ap_campaign.py).

Documents associés : [ap4-bilan-campagne-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-95"></a>

#### 95. AP1 direct — témoin à budget de sortie égal

Expérience `427b7f29-9918-4521-9198-205b2965e521`.

**Intervention.** Un jugement causal direct à budget 3 072 tokens sur les mêmes paires et deux orientations que le bras décomposé.

**Panel et comparaison.** CTB20/ESC12 ; 792 réponses finales valides.

**Résultat et portée.** CTB 56,52 contre 65,00 ; ESC 40,44 contre 35,50. Sur ESC, delta +4,94 [−1,05 ; +12,55], encore incertain.

**Contrôles et limites.** Collision de fichiers temporaires réparée avant le run final, sans changer le protocole scientifique ; caches et budgets persistants. La sélection d’AP1 direct reste exploratoire.

**État des exécutions à l’instantané.** done :1, failed :1. Référence de lecture : [run f5b579ef](../results/inventaire-executions.csv) (`f5b579ef-ff00-4587-9610-42aa6f2abf24`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes292, SHA-256 `9594a744e38b361efe005f7b10207f4ff7fa299a8c89a15bb57903e2c148efdb` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/campaign_api.py](https://github.com/Fireman4740/workere/blob/4c14675614267c7d1459c5bdb2b909f86a47aafe/steering/campaign_api.py) [steering/test_campaign_api.py](https://github.com/Fireman4740/workere/blob/4c14675614267c7d1459c5bdb2b909f86a47aafe/steering/test_campaign_api.py).

Documents associés : [ap1-comparaison-resultats-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-96"></a>

#### 96. AP1 décomposé — faits, mécanisme et explication concurrente

Expérience `2c5d4351-e315-4ca1-abf7-1f84b23a642e`.

**Intervention.** Trois sous-analyses puis synthèse, avec plafond de sortie total contrôlé contre AP1 direct.

**Panel et comparaison.** Même panel et mêmes paires que le frère direct.

**Résultat et portée.** CTB 53,06 ; ESC 39,20. Décomposé−direct : CTB −3,46 [−8,60 ; 0], ESC −1,24 [−5,28 ; +1,15]. Coût d’entrée total 2,32 fois celui du direct.

**Contrôles et limites.** 70 sous-analyses invalides déclenchent l’abstention prévue, sans retirer de documents. Aucun gain démontré de la décomposition ; absence de significativité ne signifie pas équivalence.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run eca9ab9a](../results/inventaire-executions.csv) (`eca9ab9a-3941-45c4-b529-61a5cff1f501`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes253, SHA-256 `e13995084f35edc7a830f260491df7327a8189e6dd405e2127c42671c0618d54` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/e13b39f0e08c2f8248a528adf5da4579eae568f4/steering/config.env).

Documents associés : [ap1-comparaison-resultats-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-97"></a>

#### 97. AP2 — récupérer les liens rejetés unanimement

Expérience `3cc9354b-c8c5-45cd-9815-fafed98bf23c`.

**Intervention.** Réinterroger une sélection de paires négatives du vote et des deux présentations C2, avec politique fixée sans lire leurs labels.

**Panel et comparaison.** CTB20/ESC12, panel déjà exposé.

**Résultat et portée.** Aucun vrai lien récupéré, un faux lien ESC ajouté. L’audit ultérieur trouve pourtant un vrai lien CTB et treize ESC dans la sélection ; ils restent tous rejetés.

**Contrôles et limites.** Couverture de candidats et discrimination sont deux problèmes distincts. Les ajouts AP2 sont désactivés dans la politique suivante.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 890c6bda](../results/inventaire-executions.csv) (`890c6bda-68a2-44be-8e36-32c37b3f52a7`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes52, SHA-256 `150c2c008dffcca7e17423874cb3bd7f17291f03c0c1c3da762729c7f9c6b4e3` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/ap_campaign.py](https://github.com/Fireman4740/workere/blob/0ab3363e3e883e4966b214fbbc5b19b61dd3bc52/steering/ap_campaign.py) [steering/config.env](https://github.com/Fireman4740/workere/blob/0ab3363e3e883e4966b214fbbc5b19b61dd3bc52/steering/config.env).

Documents associés : [ap2-resultats-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-98"></a>

#### 98. AP3 — stabilité et gel de la réparation

Expérience `deba7259-62b1-432e-b49b-b73ef0948283`.

**Intervention.** Répéter des prompts identiques avec réorganisation des lots, puis figer la règle avant l’extension AP4.

**Panel et comparaison.** Audit de 32 décisions binaires ESC comparables.

**Résultat et portée.** Sept décisions changent : deux deviennent correctes et cinq incorrectes. La politique garde CTB inchangé et la réparation directe seule sur ESC.

**Contrôles et limites.** Le test ne sépare pas la cause numérique de la variabilité et ne mesure pas un nouveau F1 du corpus entier. Prompts identiques et température nulle ne suffisent pas à garantir l’identité des sorties.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 3a78adf3](../results/inventaire-executions.csv) (`3a78adf3-e255-4a16-82f9-56eb4718b492`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes11, SHA-256 `9f24ee0c54d716f6e869a60026b082fceb1c5b16a97d923ef734e282a33ff6b1` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/ap_campaign.py](https://github.com/Fireman4740/workere/blob/f81f258c8214f3d925bd41e6f92c78ddc600a925/steering/ap_campaign.py) [steering/config.env](https://github.com/Fireman4740/workere/blob/f81f258c8214f3d925bd41e6f92c78ddc600a925/steering/config.env).

Documents associés : [ap3-stabilite-ap4-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-99"></a>

#### 99. AP4 — extension de la politique figée

Expérience `ea8f91d6-d566-4d54-be3d-63e9befcb372`.

**Intervention.** Réutiliser les douze documents initiaux et n’inférer que sur huit ESC supplémentaires sélectionnés par métadonnées ; CTB reste en cache.

**Panel et comparaison.** ESC20, dont huit ajoutés ; CTB30 sans nouvelle décision.

**Résultat et portée.** ESC20 : réparation 47,45 contre P2 43,70, +3,76 [−0,48 ; +8,04]. Extension8 : 58,91 contre 56,98, +1,93 [−4,81 ; +5,66]. Luna7 48,55 et fusion 50,98 sur ESC20.

**Contrôles et limites.** Les douze documents de sélection contribuent au total. Les huit ajoutés sont eux aussi issus de données déjà utilisées dans le projet. Aucun gain indépendant confirmé.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run ba9532fd](../results/inventaire-executions.csv) (`ba9532fd-373f-48b2-8940-e8d078935ff5`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes45, SHA-256 `2459ca138b74f9c47681d9e5a011f61945df9ee55d69e3e0c430145c06fe17f7` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/ap_campaign.py](https://github.com/Fireman4740/workere/blob/24d0aba9db23b77313873a3f49d28d539ae885e7/steering/ap_campaign.py) [steering/config.env](https://github.com/Fireman4740/workere/blob/24d0aba9db23b77313873a3f49d28d539ae885e7/steering/config.env) [steering/test_ap_campaign.py](https://github.com/Fireman4740/workere/blob/24d0aba9db23b77313873a3f49d28d539ae885e7/steering/test_ap_campaign.py).

Documents associés : [ap4-bilan-campagne-2026-11-08.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-100"></a>

#### 100. QG1 — sélection d’un graphe anonyme

Expérience `da90718c-25de-4328-bf82-fee5bd7d0d43`.

**Intervention.** Présenter de petits blocs de graphes candidats anonymisés ; demander de choisir, avec deux présentations et repli à la référence en cas de désaccord.

**Panel et comparaison.** CTB20/ESC20 ; référence CTB : 65,00 et ESC AP4 47,45.

**Résultat et portée.** CTB : 52,17 ; ESC : 46,61. Deltas −12,83 [−26,46 ; −2,07] et −0,84 [−2,01 ; +0,14]. Aucun système promu.

**Contrôles et limites.** 455 réponses logiques valides, audit de répétition inclus. Le gain hérité contre l’ancien P2 ne doit pas être attribué à QG1.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run f552cb19](../results/inventaire-executions.csv) (`f552cb19-4f28-4648-9432-723c126029c2`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes39, SHA-256 `8ccb5adfc2b08633a59cc1be2f2af59fdbd1be8bdb0104041cad8983a6bb283a` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/5d4b45be62def4b4d879524d0a9c275ccd277aff/steering/config.env) [steering/exp/run_qg.sh](https://github.com/Fireman4740/workere/blob/5d4b45be62def4b4d879524d0a9c275ccd277aff/steering/exp/run_qg.sh) [steering/qg_campaign.py](https://github.com/Fireman4740/workere/blob/5d4b45be62def4b4d879524d0a9c275ccd277aff/steering/qg_campaign.py) [steering/qg_protocol.py](https://github.com/Fireman4740/workere/blob/5d4b45be62def4b4d879524d0a9c275ccd277aff/steering/qg_protocol.py) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/5d4b45be62def4b4d879524d0a9c275ccd277aff/steering/run_pilot.sh) [steering/test_qg_campaign.py](https://github.com/Fireman4740/workere/blob/5d4b45be62def4b4d879524d0a9c275ccd277aff/steering/test_qg_campaign.py).

Documents associés : [qg1-resultats-2026-11-09.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-101"></a>

#### 101. QG2 — édition de graphes lien par lien

Expérience `3362b071-d832-4651-bf12-5a7cd15d4f12`.

**Intervention.** Garder le pool et les blocs QG1, mais autoriser des corrections de liens plutôt qu’un choix de graphe entier.

**Panel et comparaison.** Même CTB20/ESC20, mêmes références et manifestes que QG1.

**Résultat et portée.** CTB : 54,55 ; ESC : 47,12, contre 65,00/47,45. Différence QG2−QG1 : +2,37 [−5,48 ; +12,65] et +0,51 [−0,58 ; +2,75].

**Contrôles et limites.** Nettoyage de faux positifs compensé par une perte de rappel. Audit de stabilité petit et encore divergent. Série close sans promotion.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 5cc51d20](../results/inventaire-executions.csv) (`5cc51d20-07f6-4d25-a61a-35535693d06b`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes137, SHA-256 `bc187310d9426a3d2a0b780ddbb8e02a483d3abba451afb7fe00fe965b531ec9` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/e026a616ec429444b293433d44c95cbf018b2eef/steering/config.env).

Documents associés : [qg2-bilan-serie-2026-11-09.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-102"></a>

#### 102. VC1 — valider filtrage et réparation sur CTB47

Expérience `76150bd5-2bf5-4086-bcd5-13c8caa7eb92`.

**Intervention.** Étendre les quatre règles figées : vote, filtre si double rejet, réparation directe et filtre+ajouts, en rejouant les caches compatibles.

**Panel et comparaison.** CTB47 ; distinguer les vingt initiaux et les 27 ajoutés.

**Résultat et portée.** Vote61,19 ; filtre60,61 ; réparation58,28 ; filtre+ajouts57,33. Le gain du filtre CTB20 ne se maintient pas : sur extension27 il retire aussi un vrai lien.

**Contrôles et limites.** Audit : zéro changement sur 96 décisions comparables. Développement réutilisé ; le temps actif 4 196 s diffère du temps de suivi resté périmé.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 89900552](../results/inventaire-executions.csv) (`89900552-7067-4e64-926f-625bcf9201fb`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes45, SHA-256 `cd62d7c8706b0b87ddb2f2d18b264719167e673e2f66ba6cb9e059045a9d416c` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/config.env) [steering/exp/run_validation.sh](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/exp/run_validation.sh) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/run_pilot.sh) [steering/test_validation_campaign.py](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/test_validation_campaign.py) [steering/validation_campaign.py](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/validation_campaign.py) [steering/validation_protocol.py](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/validation_protocol.py).

Documents associés : [vc1-resultats-et-reprises-2026-11-09.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-103"></a>

#### 103. VC2 — étendre la réparation sur ESC40

Expérience `93a28337-c98b-43bb-9bb8-7d6f4a2654b2`.

**Intervention.** Conserver les quatre politiques et les prédictions AP4 des anciens documents ; produire seulement les jugements manquants.

**Panel et comparaison.** TEST-S40 complet, bootstrap apparié par sujet.

**Résultat et portée.** Vote42,38 ; filtre41,25 ; réparation46,07 ; filtre+ajouts45,83. Réparation−vote +3,68 [−0,10 ; +7,48].

**Contrôles et limites.** Erreur du découpeur de phrases sur balise décimale réparée sans exclure de document. Le gain reste incertain et TEST-S40 est déjà exposé. Réparation conservée comme candidat, pas comme résultat confirmé.

**État des exécutions à l’instantané.** done :1, failed :1. Référence de lecture : [run 52629af7](../results/inventaire-executions.csv) (`52629af7-60c7-40f1-ae61-51dbc4d538a2`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes139, SHA-256 `09e3867a0d8d351945f92edf33394700785dd9be1bdf4780f25ced72ae26d492` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/test_validation_campaign.py](https://github.com/Fireman4740/workere/blob/e130b6d88c02b2468103fa6b9bd31b41b136a4e4/steering/test_validation_campaign.py) [steering/validation_campaign.py](https://github.com/Fireman4740/workere/blob/e130b6d88c02b2468103fa6b9bd31b41b136a4e4/steering/validation_campaign.py) [steering/validation_protocol.py](https://github.com/Fireman4740/workere/blob/e130b6d88c02b2468103fa6b9bd31b41b136a4e4/steering/validation_protocol.py).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-104"></a>

#### 104. VC3 — transfert des politiques sur MAVEN12

Expérience `e3fc6f2a-4a11-444b-a667-6bbb4fef334d`.

**Intervention.** Cinq lectures documentaires puis les mêmes politiques de filtre/réparation que CTB et ESC ; conserver tous les candidats.

**Panel et comparaison.** Douze documents MAVEN-ERE test ; 60 lectures attendues.

**Résultat et portée.** Vote29,47 ; filtre29,79 ; réparation26,54 ; filtre+ajouts26,92. Le filtre retire deux faux liens non ordonnés ; une lecture invalide sur soixante.

**Contrôles et limites.** Douze documents ne valent pas une évaluation des 710. Choix du vote simple pour la suite, sans comparaison appariée au score historique DeepSeek710.

**État des exécutions à l’instantané.** done :1, lost :1, failed :2. Référence de lecture : [run 71e9dace](../results/inventaire-executions.csv) (`71e9dace-b52a-4774-8332-a429a640a587`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes69, SHA-256 `726d7c696f8e84fd1eb6e2acece9dcd05d168bdd4658f6397d85d3dae9d3b9ec` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/test_validation_campaign.py](https://github.com/Fireman4740/workere/blob/c784e8fc9ffcbaa1070e1a9c7251c12260141662/steering/test_validation_campaign.py) [steering/validation_campaign.py](https://github.com/Fireman4740/workere/blob/c784e8fc9ffcbaa1070e1a9c7251c12260141662/steering/validation_campaign.py).

Documents associés : [campagne-complete-deepseek-plan-2026-11-10.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-105"></a>

#### 105. MV1 — votes de familles, première tentative

Expérience `a1392290-6b52-4b40-889e-c4765c6597cf`.

**Intervention.** Recombiner hors ligne vote documentaire, deux vues C2 et deux jugements directs ; repli au vote si panel incomplet.

**Panel et comparaison.** CTB47 et ESC40 calculés ; attente MAVEN avant perte du suivi.

**Résultat et portée.** Les résultats CTB et ESC sont présents malgré le statut lost. Leur replay complet est repris dans MV2 ; aucune nouvelle génération n’est nécessaire.

**Contrôles et limites.** Statut de suivi et disponibilité d’un résultat sont distincts. Ne pas compter MV1 et MV2 comme deux réplications scientifiques indépendantes.

**État des exécutions à l’instantané.** lost :1. Référence de lecture : [run cf0a2dbe](../results/inventaire-executions.csv) (`cf0a2dbe-5897-450e-9899-75975cfd5f15`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes2, 3, SHA-256 `cd48d8c4368f5accea1db96207167e02ea58457a640a61b1865bf6ae821ce499` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/config.env) [steering/exp/run_method_vote.sh](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/exp/run_method_vote.sh) [steering/method_vote.py](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/method_vote.py) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/run_pilot.sh) [steering/test_method_vote.py](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/test_method_vote.py).

Documents associés : [vote-combinaisons-plan-2026-11-10.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-106"></a>

#### 106. FULL CTB183 — plan de corpus complet

Expérience `13e2df00-781e-413d-8fb6-ef35392f290f`.

**Intervention.** Recette prévue : cinq lectures 8k et cinq 20k, avec contrôle documentaire zéro-shot distinct et données épinglées.

**Panel et comparaison.** 183 documents CTB ; aucune exécution enregistrée à l’instantané.

**Résultat et portée.** Plan seulement. Aucun F1 Qwen complet attribuable à cette expérience.

**Contrôles et limites.** Le pilote 61,19 sur CTB47 ne devient pas un score CTB183. Comparaison DeepSeek63,3 descriptive seulement, prédictions historiques absentes.

**État.** Aucun run enregistré ; préparation seulement.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/config.env) [steering/exp/run_validation.sh](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/exp/run_validation.sh) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/run_pilot.sh) [steering/test_validation_campaign.py](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/test_validation_campaign.py) [steering/validation_campaign.py](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/validation_campaign.py) [steering/validation_protocol.py](https://github.com/Fireman4740/workere/blob/f6fca6c31dd520c01ac3ee24e314a49ce70ee0bf/steering/validation_protocol.py).

Documents associés : [campagne-complete-deepseek-plan-2026-11-10.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-107"></a>

#### 107. FULL ESC233 — plan de combinaison complète

Expérience `9626b007-5433-469e-8ac2-b014200e85e2`.

**Intervention.** Majorité prévue de cinq vues avec repli explicite ; vote et réparation en ablations ; contrôle zéro-shot distinct.

**Panel et comparaison.** 233 documents ESC ; aucune exécution enregistrée à l’instantané.

**Résultat et portée.** Plan seulement ; aucun résultat de corpus complet produit par ce nœud.

**Contrôles et limites.** Implémentation et précontrôle du transport long contexte requis. Pas de propagation gold dans le score principal ; comparaison historique DeepSeek46,6 non appariée.

**État.** Aucun run enregistré ; préparation seulement.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/config.env) [steering/exp/run_method_vote.sh](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/exp/run_method_vote.sh) [steering/method_vote.py](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/method_vote.py) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/run_pilot.sh) [steering/test_method_vote.py](https://github.com/Fireman4740/workere/blob/80a3beeeb14e9a277b52f2af63f25163bea1f51b/steering/test_method_vote.py).

Documents associés : [campagne-complete-deepseek-plan-2026-11-10.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-108"></a>

#### 108. FULL MAVEN710 — plan de transfert complet

Expérience `d17d4185-6be0-43eb-97f8-5645f601df8f`.

**Intervention.** Figer une recette issue du pilote, démonstrations issues du train, contrôle zéro-shot et suivi de tous les échecs.

**Panel et comparaison.** 710 documents MAVEN test ; aucune exécution enregistrée à l’instantané.

**Résultat et portée.** Plan seulement. Séparer à terme les douze documents du pilote des 698 autres, sans revendiquer une absence globale d’exposition non auditée.

**Contrôles et limites.** Pas de score inféré du titre du nœud ni de lancement par cette mise à jour documentaire. DeepSeek31,1 reste une référence historique descriptive.

**État.** Aucun run enregistré ; préparation seulement.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/test_validation_campaign.py](https://github.com/Fireman4740/workere/blob/3d4a318e12388ed4ba380ae6042f1872463761a7/steering/test_validation_campaign.py) [steering/validation_campaign.py](https://github.com/Fireman4740/workere/blob/3d4a318e12388ed4ba380ae6042f1872463761a7/steering/validation_campaign.py).

Documents associés : [campagne-complete-deepseek-plan-2026-11-10.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-109"></a>

#### 109. MV2 — replay complet des votes de familles

Expérience `e8ee7f69-9f5a-4c5e-8d56-e1296e545e1b`.

**Intervention.** Reprendre les règles figées de MV1 après redémarrage de l’hôte, sans nouvel appel au modèle.

**Panel et comparaison.** CTB47 / ESC40 / MAVEN12 ; panels complets à cinq vues sur 81/2 375, 1 142/8 979 et 339/2 530 paires respectivement.

**Résultat et portée.** Majorité cinq vues : CTB : 56,25 contre 61,19 ; ESC : 46,74 contre 42,38, +4,35 [−0,88 ; +9,00] ; MAVEN : 24,78 contre 29,47. Consensus strict et règle asymétrique ne démontrent pas de progrès.

**Contrôles et limites.** Vues du même modèle corrélées, non réplications indépendantes. Le faible routage complet impose de rapporter le repli au vote et le coût amont. ESC : 46,74 ne bat pas la fusion historique48,64 en valeur centrale.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 35ecb7c3](../results/inventaire-executions.csv) (`35ecb7c3-ac28-45fe-a470-122bb84e3913`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes2, 3, 237, 238, SHA-256 `21bce7bc847c9634b65e48eba16b79a112f62d07dfe6ebd0bad85ccf22e9d7f4` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/c28119b6c4038d707698b675daefec4d22358cd5/steering/config.env) [steering/method_vote.py](https://github.com/Fireman4740/workere/blob/c28119b6c4038d707698b675daefec4d22358cd5/steering/method_vote.py).

Documents associés : [vote-combinaisons-plan-2026-11-10.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-110"></a>

#### 110. W1 — modification de poids : conclusion et rappel

Expérience `80aaa57e-9c8d-4546-96a6-d30a2034dc84`.

**Intervention.** Construire des corrections réversibles de rang 1 depuis les traces ; deux directions, deux doses et quatre témoins aléatoires ; aucune optimisation par gradient.

**Panel et comparaison.** CTB20 de sélection ; 180 réponses pour neuf conditions ; extraction sur traces hors panel.

**Résultat et portée.** Base70,00 ; conclusion 0,10 :73,68, delta+3,68 [−2,62 ; +14,70], réflexion−37,69 %. Contrôle aléatoire recall0 :76,54. Direction recall0,10 :30,46.

**Contrôles et limites.** La construction de rappel exploite les annotations existantes et l’intervention modifie les poids effectifs ; ne pas la qualifier de Qwen inchangé ni de zéro supervision. Seule conclusion 0,10 passe le gate de sélection, sans preuve de spécificité.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 7540caf0](../results/inventaire-executions.csv) (`7540caf0-4116-4cb7-bf1e-bf1c2c0ef625`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes1363, 1365, SHA-256 `2f36ae1a3831bdccc0a86dd3de78bfc3585a9724c9d0d09d01e7f573aa3480c5` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/3fcf18fbcda68c2dc06c4b4593760cef595e3784/steering/config.env) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/3fcf18fbcda68c2dc06c4b4593760cef595e3784/steering/run_pilot.sh) [steering/run_vllm.py](https://github.com/Fireman4740/workere/blob/3fcf18fbcda68c2dc06c4b4593760cef595e3784/steering/run_vllm.py) [steering/run_wsp.sh](https://github.com/Fireman4740/workere/blob/3fcf18fbcda68c2dc06c4b4593760cef595e3784/steering/run_wsp.sh) [steering/weight_space.py](https://github.com/Fireman4740/workere/blob/3fcf18fbcda68c2dc06c4b4593760cef595e3784/steering/weight_space.py) [steering/wsp_pilot.py](https://github.com/Fireman4740/workere/blob/3fcf18fbcda68c2dc06c4b4593760cef595e3784/steering/wsp_pilot.py).

Documents associés : [resultats-w1-et-confirmation-w2.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-111"></a>

#### 111. W2 — confirmation réservée de conclusion 0,10

Expérience `4abfa954-6b32-4464-861c-e4840e8ae1bf`.

**Intervention.** Garder les vecteurs, dose, couches et prompts figés ; comparer base, candidate et quatre témoins sur le panel réservé avant W1.

**Panel et comparaison.** Vingt autres documents CTB ; 120 réponses complètes ; 19 liens non orientés gold.

**Résultat et portée.** Base63,41 ; conclusion 72,73 : +9,31 [−1,72 ; +22,62]. Meilleur contrôle aléatoire 76,92. Réflexion−48,90 %, fermetures forcées9→0.

**Contrôles et limites.** Critère primaire et critère de spécificité échouent ; série close sans promotion. Panel réservé pour cette série, dans un corpus déjà étudié. Les économies de tokens sont secondaires, sans non-infériorité de F1 démontrée.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 9b4089c0](../results/inventaire-executions.csv) (`9b4089c0-2a09-4e70-83f3-95164fb121dc`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes610, 612, SHA-256 `17bf1be95d23697efa66fd569969334eaabe3bda174649d0afb21397df78ce02` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/f675d3a01e06ad9f2222d5fa3e2961573cd35e26/steering/config.env) [steering/run_wsp.sh](https://github.com/Fireman4740/workere/blob/f675d3a01e06ad9f2222d5fa3e2961573cd35e26/steering/run_wsp.sh) [steering/wsp_confirmation.py](https://github.com/Fireman4740/workere/blob/f675d3a01e06ad9f2222d5fa3e2961573cd35e26/steering/wsp_confirmation.py) [steering/wsp_confirmation_plan.json](https://github.com/Fireman4740/workere/blob/f675d3a01e06ad9f2222d5fa3e2961573cd35e26/steering/wsp_confirmation_plan.json) [tests/test_wsp_confirmation.py](https://github.com/Fireman4740/workere/blob/f675d3a01e06ad9f2222d5fa3e2961573cd35e26/tests/test_wsp_confirmation.py).

Documents associés : [resultats-w1-et-confirmation-w2.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-112"></a>

#### 112. W3 — diversité et majorité stricte sur caches W1/W2

Expérience `27a2a07a-960e-49e4-9004-a70c8e3a50fb`.

**Intervention.** Auditer la complémentarité des six lectures et des contrôles, puis appliquer des votes préspécifiés sans seuil choisi sur les résultats.

**Panel et comparaison.** Deux panels W1/W2 ; CPU uniquement, zéro nouvelle génération.

**Résultat et portée.** Majorité de six lectures : W1 72,97, W2 63,41, pour environ5,73×/5,38× les tokens de la base. La diversité ne se convertit pas en gain fiable.

**Contrôles et limites.** Pas de témoin à six lectures indépendantes sans intervention dans ces caches. Ce diagnostic ne revalide pas W2 ni ne prouve un effet général des votes.

**État des exécutions à l’instantané.** done :1. Référence de lecture : [run 7cc3225b](../results/inventaire-executions.csv) (`7cc3225b-7ea8-4517-88d6-2121bbf5bba8`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes2, 3, 4, SHA-256 `17fcd98e99b60f218b35b80cfb2c76bfba51abfd8559bfcf507d953e8081854b` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/445063a1d0ba658ad83eedbedbb3cc5d72a56320/steering/config.env) [steering/run_wsp.sh](https://github.com/Fireman4740/workere/blob/445063a1d0ba658ad83eedbedbb3cc5d72a56320/steering/run_wsp.sh) [steering/wsp_diversity.py](https://github.com/Fireman4740/workere/blob/445063a1d0ba658ad83eedbedbb3cc5d72a56320/steering/wsp_diversity.py) [steering/wsp_diversity_plan.json](https://github.com/Fireman4740/workere/blob/445063a1d0ba658ad83eedbedbb3cc5d72a56320/steering/wsp_diversity_plan.json) [tests/test_wsp_diversity.py](https://github.com/Fireman4740/workere/blob/445063a1d0ba658ad83eedbedbb3cc5d72a56320/tests/test_wsp_diversity.py).

Documents associés : [audit-w3-diversite.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-113"></a>

#### 113. W4 — steering, votes et vérification sur CTB47

Expérience `cac94efe-b61a-4c78-8b92-f92503a10bfc`.

**Intervention.** Cinq nouvelles lectures sans intervention et cinq avec conclusion 0,10, mêmes démonstrations ; douze systèmes de vote et sept politiques de vérification : 96 variantes avec référence.

**Panel et comparaison.** CTB47 ; référence historique N1+P3 rejouée à61,19.

**Résultat et portée.** Sans steering55,64 ; avec 54,14 ; vote vingt lectures 59,09 ; meilleure variante ajoutée59,54. Aucune ne dépasse61,19. Steering−sans :−1,50 [−7,31 ; +4,02].

**Contrôles et limites.** 470 lectures et128 vérifications complètes ; pool vérifié100 paires dont44/70 vrais liens,26 absents. Réduction de31,84 % des tokens des lectures, sans preuve de gain de temps ni de non-infériorité. Exploration à comparaisons multiples.

**État des exécutions à l’instantané.** done :1, lost :2. Référence de lecture : [run e86ed345](../results/inventaire-executions.csv) (`e86ed345-3d30-49da-9f23-b30b6520f7b0`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes2, 333, 548, 549, SHA-256 `d6959089d610d03563ccf06072ee0c152441ca27b9bd5355a9003e5aa25b2741` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/8c0d8261320e03cd2e1a5891deaa1df3c0373ea4/steering/config.env) [steering/data/esc/baseline.json](https://github.com/Fireman4740/workere/blob/8c0d8261320e03cd2e1a5891deaa1df3c0373ea4/steering/data/esc/baseline.json) [steering/data/esc/prompt.json](https://github.com/Fireman4740/workere/blob/8c0d8261320e03cd2e1a5891deaa1df3c0373ea4/steering/data/esc/prompt.json) [steering/data/esc/test-s40.json](https://github.com/Fireman4740/workere/blob/8c0d8261320e03cd2e1a5891deaa1df3c0373ea4/steering/data/esc/test-s40.json) [steering/run_pilot.sh](https://github.com/Fireman4740/workere/blob/8c0d8261320e03cd2e1a5891deaa1df3c0373ea4/steering/run_pilot.sh) [steering/run_wsp_combo.sh](https://github.com/Fireman4740/workere/blob/8c0d8261320e03cd2e1a5891deaa1df3c0373ea4/steering/run_wsp_combo.sh).

Documents associés : [bilan-w4-ctb47-combinaisons.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-114"></a>

#### 114. W5 — pilote ESC réduit à quatre documents

Expérience `71431ef8-2888-4329-9e6d-084406b938f3`.

**Intervention.** Appliquer le protocole de votes sans/avec steering et de vérification, mais réduire le panel avant le run final.

**Panel et comparaison.** Quatre documents ESC de quatre sujets. Le titre historique annonce encore ESC40 ; ce titre n’est pas le périmètre effectivement mesuré.

**Résultat et portée.** Référence P2 sur ces quatre33,33 ; nouvelles lectures sans 44,00, avec 37,21 ; vote mixte47,83 ; meilleure combinaison observée54,17.

**Contrôles et limites.** Meilleur résultat parmi96 variantes corrélées, quatre documents déjà exposés. Ce54,17 n’est ni un résultat ESC40/ESC233 ni une confirmation. Il ne démontre pas l’effet propre du steering.

**État des exécutions à l’instantané.** done :1, Cancelled · couldn't confirm :1, lost :3. Référence de lecture : [run aa041b15](../results/inventaire-executions.csv) (`aa041b15-ebb2-4cd1-8bbf-079aa72ff648`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes3, 270, 325, 326, SHA-256 `08507fec6ceeeffc96b21540da890326c2de48a3789be2b051fec9ad930f4d37` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/9c73372d8c6f13a5f894e53c026681b911dce10b/steering/config.env) [steering/run_wsp_combo.sh](https://github.com/Fireman4740/workere/blob/9c73372d8c6f13a5f894e53c026681b911dce10b/steering/run_wsp_combo.sh) [steering/wsp_short.py](https://github.com/Fireman4740/workere/blob/9c73372d8c6f13a5f894e53c026681b911dce10b/steering/wsp_short.py) [tests/test_wsp_short.py](https://github.com/Fireman4740/workere/blob/9c73372d8c6f13a5f894e53c026681b911dce10b/tests/test_wsp_short.py).

Documents associés : [pilotes-courts-esc4-maven4.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-115"></a>

#### 115. W6 — pilote MAVEN réduit et format invalide

Expérience `40d27790-2277-4230-8041-966ec13e88de`.

**Intervention.** Transférer le protocole de votes/vérification avec le même edit figé.

**Panel et comparaison.** Quatre documents MAVEN, dont deux du panel initial et deux de l’extension ; titre historique MAVEN24 conservé.

**Résultat et portée.** Sans intervention : vote 0,00 ; avec :36,73 ; meilleure réparation jointe41,94. Le résumé final du journal compte16/20 lectures sans intervention et9/20 avec invalides ;48/48 réponses de vérification sont valides. Une note antérieure donne18/7 : ces deux décomptes ne sont pas interchangeables.

**Contrôles et limites.** Échec de format fortement asymétrique : le contraste n’isole pas une efficacité du steering. Référence historique disponible seulement sur les deux anciens documents. Les scores effectifs sont consignés, sans conclure à une supériorité.

**État des exécutions à l’instantané.** done :1, Cancelled · couldn't confirm :1, lost :3. Référence de lecture : [run 95133280](../results/inventaire-executions.csv) (`95133280-621a-4bb2-87cf-29aef6dca1ae`). Le statut ne remplace pas le résultat décrit ci-dessus.

Repères du journal local : lignes3, 410, 505, 506, SHA-256 `db4b0d3ddb2843c07616438ddf4498541488bc8a975096b9c807ba6961f9997d` ; voir [l’index des journaux](../SOURCES.md#annexes-locales).

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/356eabec785cc2ca849114bdddee81ae624a27e8/steering/config.env) [steering/run_wsp_combo.sh](https://github.com/Fireman4740/workere/blob/356eabec785cc2ca849114bdddee81ae624a27e8/steering/run_wsp_combo.sh) [steering/wsp_short.py](https://github.com/Fireman4740/workere/blob/356eabec785cc2ca849114bdddee81ae624a27e8/steering/wsp_short.py) [tests/test_wsp_short.py](https://github.com/Fireman4740/workere/blob/356eabec785cc2ca849114bdddee81ae624a27e8/tests/test_wsp_short.py).

Documents associés : [bilan-w6-maven4-pilote-court.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-116"></a>

#### 116. W7 — extension ESC50

Expérience `6a41df0c-b7e6-443f-a930-b131abe1a68c`.

**Intervention.** Étendre le pilote à cinquante documents, caches réutilisés seulement si identité exacte ; mêmes familles de votes et vérification.

**Panel et comparaison.** Cinquante documents ESC prévus ; 53 lectures réutilisables au précontrôle.

**Résultat et portée.** Exécution enregistrée active à l’instantané ; aucun journal local récupérable ni résumé final. Aucun F1 annoncé.

**Contrôles et limites.** 38 tests CPU rapportés dans le protocole de lancement. Exposition de développement conservée ; la file GPU est séquentielle avec MAVEN. Statut actif ne prouve pas que toutes les inférences sont en cours.

**État des exécutions à l’instantané.** running :1. Référence de lecture : [run 7d071a42](../results/inventaire-executions.csv) (`7d071a42-c810-4e3b-8ba7-a74534a35b01`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/6dd0f9e3d951c05c46d86526aa606ff494ae62b0/steering/config.env) [steering/scale50_select.py](https://github.com/Fireman4740/workere/blob/6dd0f9e3d951c05c46d86526aa606ff494ae62b0/steering/scale50_select.py) [steering/wsp_scale50.py](https://github.com/Fireman4740/workere/blob/6dd0f9e3d951c05c46d86526aa606ff494ae62b0/steering/wsp_scale50.py) [tests/test_wsp_scale50.py](https://github.com/Fireman4740/workere/blob/6dd0f9e3d951c05c46d86526aa606ff494ae62b0/tests/test_wsp_scale50.py).

Documents associés : [protocole-w7-w8-50-documents.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


<a id="fiche-117"></a>

#### 117. W8 — extension MAVEN50 avec schéma de paire contraint

Expérience `2bf1b4a1-606e-44b5-b1a0-f90f99c7c8bb`.

**Intervention.** Étendre à cinquante documents et imposer un format de paire strict pour corriger l’échec de format du pilote W6.

**Panel et comparaison.** Cinquante documents MAVEN prévus, en file après ESC.

**Résultat et portée.** Exécution enregistrée active à l’instantané ; aucun journal local récupérable ni résumé final. Aucun résultat final disponible.

**Contrôles et limites.** Le changement de schéma peut résoudre un problème de forme ; son effet sur la qualité reste à mesurer. Ne pas confondre soumission active et acquisition effective du GPU.

**État des exécutions à l’instantané.** running :1. Référence de lecture : [run e8d91d5c](../results/inventaire-executions.csv) (`e8d91d5c-4c3d-4f9c-a78a-9f13898b138f`). Le statut ne remplace pas le résultat décrit ci-dessus.

Fichiers du commit de référence inspecté (liste ciblée, pas inventaire complet du pipeline) : [steering/config.env](https://github.com/Fireman4740/workere/blob/356eabec785cc2ca849114bdddee81ae624a27e8/steering/config.env) [steering/scale50_select.py](https://github.com/Fireman4740/workere/blob/356eabec785cc2ca849114bdddee81ae624a27e8/steering/scale50_select.py) [steering/wsp_scale50.py](https://github.com/Fireman4740/workere/blob/356eabec785cc2ca849114bdddee81ae624a27e8/steering/wsp_scale50.py) [tests/test_wsp_scale50.py](https://github.com/Fireman4740/workere/blob/356eabec785cc2ca849114bdddee81ae624a27e8/tests/test_wsp_scale50.py).

Documents associés : [protocole-w7-w8-50-documents.md](../SOURCES.md#annexes-locales).

[Identifiants et toutes les tentatives](../SOURCES.md#annexes-locales) · [Retour à la synthèse](synthese-resultats.md).


## 4. Travaux complémentaires qui ne sont pas de nouveaux nœuds d’inférence

### 4.1 Migration et parité du benchmark

Le travail de migration sépare chargement/candidats, votes et scoring de l’orchestration LLM. Les modules canoniques du benchmark reprennent le code historique après substitution des imports ; les anciens imports deviennent des réexports. Des tests comparent le texte au commit d’origine et confrontent les scores à un oracle qui charge séparément ce code.

Les contrôles portent sur les suites figées, le sampling, les folds, les checkpoints, les candidats, la corruption de rapports et les conventions de labels. Cela permet de tester qu’un producteur externe utilise le même évaluateur. La parité des scores historiques DeepSeek reste non démontrée faute de prédictions originales : des notes récapitulatives et des fixtures synthétiques ne suffisent pas. [Rapport de migration](../SOURCES.md#annexes-locales).

### 4.2 Diagnostics Mercury : calibration, ordre et erreurs

Les diagnostics du snapshot partiel repartent des probabilités et décisions sauvegardées. Ils examinent AP/AUROC, calibration par tranches, désaccord A/B, distance entre mentions, origine directe/propagée des annotations et exemples FP/FN échantillonnés. Les intervalles rééchantillonnent les documents ; la revue qualitative n’est pas une adjudication indépendante.

Ces analyses servent à formuler les variantes R1/R2. Elles ne montrent pas qu’une correction a déjà amélioré le F1. Le sous-ensemble partiel reste conditionné à l’arrêt de Mercury. [Analyse des erreurs Mercury](../SOURCES.md#annexes-locales).

### 4.3 Diagnostics de plafond de la fusion

Des relectures de caches DEV44 testent seuil global, seuils intra/inter, chaînes par surface ou tête lemmatisée, puis chaînes gold et seuil oracle par document. Les versions déployables choisissent leurs paramètres hors topic ; les oracles utilisent une information indisponible en production et sont étiquetés comme tels.

Le diagnostic de bruit sur les chaînes sert à préparer le gate J1R. Il ne transforme pas 85 % en loi universelle de précision minimale : son utilité dépend du bruit, du rappel, de la taille des groupes et de la règle de propagation. [Rapport Harness² et plafond](../SOURCES.md#annexes-locales).

### 4.4 Rectangles et profils de voisins dans le graphe causal

Un proposeur structurel cherche des rectangles incomplets et des profils de voisins analogues dans le graphe des prédictions. La paire cible est exclue des témoins ; seuls les liens initiaux servent de racines, avec quotas et absence de récursion. Les candidats sont classés sans gold, puis évalués a posteriori sur les annotations existantes.

Les contrôles construisent des témoins appariés par score/degré, testent plusieurs budgets top-k et la sensibilité au retrait du meilleur topic. Une seconde phase identifie un problème de classement dépendant des ensembles/orientations et en conserve une correction déterministe distincte. Des paquets de preuve, un schéma et un vérificateur textuel sont préparés, mais aucun LLM ne l’a appliqué et aucun humain n’a rempli le formulaire. Aucun F1 après vérification textuelle n’est donc mesuré. [Phase initiale](../SOURCES.md#annexes-locales), [phase de décision et contrôles](../SOURCES.md#annexes-locales).

### 4.5 Conception de concurrence d’attribution

Le dossier de conception distingue extraction d’une preuve, liaison de ses arguments aux occurrences, compatibilité de portée et abstention. Les premiers contrôles sont des audits de données et des exemples logiques construits. La réalisation automatique effectivement mesurée est la fiche CTB 46 ; les versions plus riches nécessitant une annotation humaine ne sont pas réputées exécutées. [Conception](../SOURCES.md#annexes-locales), [verdict du prototype](../SOURCES.md#annexes-locales).

### 4.6 Pilotes historiques Laya : trois formulations et recalibration

Avant les campagnes suivies ci-dessus, le dépôt documente un pilote Laya limité aux paires intra-phrase non ordonnées. L’état contient la phrase, deux mentions balisées et leurs surfaces. Trois formulations sont comparées : choix directionnel à trois classes, question binaire et formulation d’inférence textuelle. Le gate annoncé pour passer au corpus complet porte sur l’AUROC de développement ; le pilote MAVEN est partiel. Jev est implémenté comme backend mais n’a pas été testé dans cette étude.

Une seconde analyse réutilise les probabilités sauvegardées pour ajuster une température, puis biais+température, avec cross-fitting à deux folds documentaires. Les diagnostics regardent NLL/ECE, AP/AUROC, fréquence des options et discrimination des paires à l’intérieur des mêmes phrases. Le meilleur seuil sur le jeu inspecté est étiqueté oracle. Ces essais restent des faits rapportés par une note ancienne : leurs journaux source ne sont pas rattachés aux 104 runs de cet inventaire et leurs chiffres ne sont pas recertifiés ici. Les interprétations architecturales de la note ne sont pas de nouvelles conclusions de ce rapport. `notes/decision_models_eci.md` `baselines/decision_eci.py` `baselines/decision_eci_recal.py`

### 4.7 Réécriture du framework et sweeps post hoc historiques

La réécriture depuis agentere compare prompts, tours de démonstration, clés de cache et métriques reconstruits aux traces archivées des anciens runs. Des transports simulés testent reprise après interruption, réponses mal formées, concurrence réelle et réutilisation de CoT. Les comparaisons live de concurrence conservent les requêtes mais restent soumises à la variabilité de génération et au service : les anciens commentaires de causalité sur le drift ne valent pas une ablation contrôlée supplémentaire. `notes/REPORT.md`

Les sweeps post hoc ESC/MAVEN reconstruisent les décisions à chaque seuil de vote, puis testent expansion par coréférence gold, une étape de transitivité et fermeture au point fixe. Ils évaluent all/intra/inter et vérifient d’abord que le seuil majoritaire reproduit le résultat enregistré. Les chaînes gold en font des diagnostics oracle ; le meilleur seuil choisi sur les mêmes données est rétrospectif. Les prédictions historiques correspondantes ne sont pas toutes récupérables aujourd’hui, donc les notes ne remplacent pas un rejeu apparié contemporain. `notes/posthoc_sweep_e77431d6_dfe087fb.md`

### 4.8 Plans de fine-tuning et devis de corpus complet

Les plans d’entraînement Qwen et les devis de tests larges décrivent un travail possible, pas des résultats acquis. Le registre ne contient pas d’expérience d’entraînement correspondant à ces plans dans ce bilan. L’absence de nœud d’entraînement et les calibrations supervisées J3/J4 sont deux faits différents. [Plan de fine-tuning](../SOURCES.md#annexes-locales), [programme de tests](../SOURCES.md#annexes-locales).

## 5. Ce que les tests autorisent à conclure

La fusion R3 dispose d’une comparaison après gel et d’un avantage sur d1 dans TEST-S40 ; son avantage sur Luna seule reste incertain. Les gains DEV de seuils et de prompts n’ont pas tous passé leurs tests de transfert. J3 rappelle pourquoi il faut auditer le gold et les conventions d’orientation avant de lire un F1.

Les familles de structure/transport montrent aussi pourquoi il faut mesurer le rendement de chaque étage : sources utilisables, candidats, admissions avant filtre, ajouts finaux et TP/FP. Des citations exactes, des identités déclarées par un modèle ou des réponses JSON valides ne garantissent pas la vérité de la relation.

Les expériences d’efficacité répondent à des questions distinctes : N5 moyenne tous les sous-ensembles archivés ; N8 teste un préfixe frais ; N10 combine six lectures archivées ; N11 garde exactement la règle à sept et certifie l’arrêt. Elles ne peuvent pas être résumées par une unique affirmation « trois lectures suffisent ».

Les essais N4, N6 et N13 xhigh ont une limite de validité observée. Leur processus peut être terminé sans résultat comparatif primaire. La partie post hoc N13 est explicitement restreinte aux trois documents complets ; elle ne change pas ce statut. Les runs actifs restent un état daté et ne reçoivent pas ici de conclusion anticipée.

## 6. Reproduire la lecture sans relancer de modèles

Pour vérifier une fiche, partir de son identifiant d’expérience et de son run de référence, puis suivre la chaîne : **commit source → protocole/corps de requêtes → réponses brutes → gate → votes → décisions gelées → labels d’évaluation → TP/FP/FN → IC**.

Le code d’une ancienne expérience peut être absent du worktree courant ; lire son commit avec Git, sans changer de branche :

```sh
git show <commit>:<chemin-du-fichier>
```

Les scripts d’audit indépendants sont liés dans les rapports N1/N2/N3/N7/N8/N9/N10/N12 et N13. Les lancer dans une copie isolée de leurs entrées évite d’écraser les sorties publiées. Consulter leur environnement et leurs arguments avant exécution ; le présent rapport ne fournit pas une commande universelle d’inférence.

Pour contrôler un résultat numériquement, reconstruire d’abord les comptes sur le même ensemble de candidats, puis vérifier le F1 et l’unité du bootstrap. Pour J3, utiliser le gold symétrisé corrigé. Pour N5/N10, conserver la distinction moyenne des F1/F1 des comptes moyens. Pour N13, ne pas compter deux fois les orientations d’une même génération.

## 7. Annexes de traçabilité

**État courant du 12 novembre :** [117 expériences](../SOURCES.md#annexes-locales), [190 exécutions](../results/inventaire-executions.csv), [journaux](../SOURCES.md#annexes-locales), [écarts de sources](corrections-et-provenance.md), [contrôle documentaire](../results/verification-documentaire-source.json). Les annexes suivantes restent les archives techniques détaillées du 3 novembre.


- [Inventaire des 64 expériences, runs et commits](../SOURCES.md#annexes-locales).
- [Index lisible du code et des fonctions de test](../SOURCES.md#annexes-locales).
- [Index structuré du code et des tests, avec hashes de blobs](../SOURCES.md#annexes-locales).
- [Index de journaux : lignes, extraits ciblés et empreintes](../SOURCES.md#annexes-locales).
- [Instantané des expériences](../SOURCES.md#annexes-locales) et [des exécutions](../SOURCES.md#annexes-locales).
- [Contrôle du présent document](../SOURCES.md#annexes-locales).

Les index techniques sont des outils de recherche des preuves. Un extrait tronqué ne vaut pas absence de résultat dans le journal complet ; un fichier de test présent ne vaut pas exécution de ce test ; une ancienne description n’est pas prioritaire sur un rapport corrigé. Les chiffres J3 et les statuts des récentes continuations sont traités avec ces précautions dans les fiches.
