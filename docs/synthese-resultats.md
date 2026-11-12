# workere — synthèse des expériences, résultats et découvertes

**Mise à jour du 12 novembre 2026.** Ce document est le point d’entrée courant pour préparer une présentation du projet. L’inventaire couvre **117 nœuds d’expérience et 190 exécutions**, contre 64 et 104 dans le catalogue du 3 novembre. Un nœud peut contenir plusieurs reprises ; les variantes internes à un factoriel ne sont pas autant de nœuds distincts.

Pour accéder au détail : [les 117 fiches techniques](catalogue-experiences.md), [le tableau exportable des expériences](../results/inventaire-experiences.csv), [toutes les exécutions](../results/inventaire-executions.csv) et [l’index des preuves](../SOURCES.md#annexes-locales). La [trame de présentation](#trame-presentation) est en fin de document.

## 1. Les découvertes à présenter

1. **Le projet a identifié des références utiles, mais les améliorations récentes restent surtout exploratoires.** La fusion historique Luna × d1 reste une référence ESC ; sur CTB47, le vote Qwen à démonstrations et budgets variés reste la référence conservée. Aucun classement unique ne peut réunir ces résultats, mesurés sur des corpus et panels différents. Les comparaisons exactes figurent dans les sections correspondantes.
2. **La difficulté ne se réduit pas à produire une réflexion plus longue.** Les budgets accrus réduisent les fermetures forcées, sans gain F1 établi du vote documentaire ESC. En revanche, dans les décisions par blocs CTB, passer de 2k à 4k améliore les valeurs observées : l’effet dépend donc du protocole. [P2 et C2](#qwen-documentaire).
3. **Retrouver des candidats et bien les sélectionner sont deux problèmes distincts.** Les vérificateurs ne corrigent pas les liens absents de leur pool ; lorsque le pool s’élargit, les faux positifs supplémentaires peuvent annuler les vrais liens retrouvés. L’union n’est pas un progrès en soi. [C2/AP2 et W4](#verification-et-decomposition).
4. **La diversité doit se traduire par de meilleures décisions.** Démonstrations différentes, permutations, jugements et directions de poids donnent des sorties complémentaires, mais multiplier des vues corrélées ne garantit aucun gain à budget accru. [M0/M1, MV2 et W3](#poids-et-diversite).
5. **Une réflexion plus courte n’établit ni une meilleure causalité ni une économie de temps mesurée.** La direction de conclusion réduit les tokens sur plusieurs panels, mais W2 échoue au critère de gain de tâche et aux contrôles de spécificité ; W4 ne dépasse pas son témoin. [W1–W4](#poids-et-diversite).
6. **La qualité de la mesure fait partie du résultat.** Le projet a corrigé le scoring symétrique de J3, séparé les réponses finales de fragments de réflexion, audité la stabilité des lots et relevé des différences entre titres et périmètres réellement exécutés. Les nombres corrigés ont priorité sur les résumés antérieurs. [Écarts de sources et corrections](corrections-et-provenance.md).

Ces constats sont des conclusions sur les interventions effectivement testées. Ils ne démontrent ni un plafond général des LLM sans entraînement, ni l’impossibilité de nouvelles approches, ni un mécanisme universel de raisonnement.

## 2. Tâche, métrique et niveau de preuve

La tâche est l’**identification de relations causales entre mentions d’événements fournies**. Le système reçoit les mentions et l’univers candidat ; il ne détecte pas lui-même tous les événements du texte. Les campagnes principales discutent un micro-F1 binaire d’existence du lien, direction ignorée. Un bon résultat binaire ne démontre pas une bonne direction causale. Les détails de chaque scorer restent ceux du protocole de la fiche. [Contrat technique](catalogue-experiences.md#11-entrées-candidats-et-cible).

Les scores sont sur 100 et les différences en points. Le micro-F1 agrège TP, FP et FN sur tous les documents avant la division. Compter les deux orientations d’un lien symétrique peut doubler les comptes sans changer son F1 ; cela ne crée pas deux observations indépendantes.

| Niveau | Lecture correcte |
|---|---|
| Test de code ou replay CPU | Vérifie une propriété ou reproduit des comptes ; n’établit pas un progrès scientifique |
| Smoke ou panel minuscule | Vérifie la faisabilité et produit un signal local |
| Développement réutilisé | Compare des variantes et prépare une hypothèse ; les IC nominaux ne corrigent pas la sélection |
| Confirmation réservée à une série | Teste une recette figée sur des documents séparés de sa sélection ; vérifier aussi l’exposition antérieure dans le projet |
| Oracle | Utilise des annotations de la cible ; diagnostic de marge, hors performance autonome |
| Plan ou exécution sans résumé final | Aucun score final annoncé |

Un intervalle apparié rééchantillonne les documents, ou les sujets ESC quand précisé, pour les deux systèmes simultanément. Il mesure cette incertitude documentaire conditionnellement aux générations sauvegardées, pas toute la variabilité de nouveaux décodages. Un IC contenant zéro ne démontre pas l’équivalence. Le nombre de comparaisons et la réutilisation des panels comptent également.

**Les panels ne sont pas interchangeables.** DEV25, CONF40, DEV44, TEST-S40, RES30, CTB20/47 et MAVEN12 ne sont pas un même benchmark. Les anciens 88 documents dits « réservés » ont été inclus dans H1 ; ils ne forment plus une réserve globalement intacte. La confirmation W2 est réservée pour sa série au sein d’un corpus déjà étudié. [Audit de réserve](../SOURCES.md#annexes-locales), [protocole W2](../SOURCES.md#annexes-locales).

**Régimes à distinguer :** prompting et vote à poids gelés ; réglage de seuils sur annotations existantes ; sonde E4 apprise par régression logistique ; interventions S/E/W sur activations ou poids effectifs ; oracles de coréférence. « Sans fine-tuning de Qwen » ne signifie pas toujours « sans supervision », et les variantes W modifient les poids effectifs par une correction réversible sans gradients.

## 3. Campagnes historiques : formulations, fusion, juges et premières explorations

Les tableaux suivants reprennent la synthèse vérifiée du 30 octobre. Les observations historiques sont conservées ; les états plus récents sont précisés dans les sections suivantes. Les fiches 1–60 fournissent les protocoles et leurs sources. Les deltas de panels différents ne s’additionnent pas.

**Les références à conserver**

| Périmètre | Comparaison appariée | Conclusion |
|---|---|---|
| TEST-S40, 40 documents | d1 seul 38,59 ; Luna seule 46,02 ; fusion figée 48,64 | Fusion−d1 : +10,05, IC95 documentaire [6,66 ; 13,65]. Fusion−Luna : +2,62 [−1,37 ; 7,69]. La fusion est un point de départ utile ; sa supériorité sur Luna seule reste incertaine. |
| DEV44, 44 documents | Fusion figée 47,32 | Référence des branches suivantes. Sa valeur différente de TEST-S40 vient du périmètre évalué, sans constituer une baisse temporelle du système. |
| H1 ESC, 233 documents | Haiku seul 42,78 ; Haiku×d1 43,27 ; d1 seul 35,31 | La fusion Haiku dépasse d1, mais n’établit pas un gain sur Haiku seul. Aucun classement apparié avec Luna sur l’ensemble complet n’est établi ici. |
| H1 CTB, 183 documents | Haiku seul 62,57 ; Haiku×d1 47,69 | La fusion perd 14,88 points [−18,60 ; −11,20]. Pour cette tâche, conserver le lecteur seul comme témoin principal est directement motivé. |

Sources : [round 3](../SOURCES.md#annexes-locales), [résultat H1](../SOURCES.md#annexes-locales). Les références historiques DeepSeek 46,6 sur ESC et 63,3 sur CTB ne constituent pas des comparaisons appariées avec ces sous-ensembles ou protocoles.

**Première famille — formulations, seuils et structure**

| Piste testée | Résultat observé | Décision actuelle |
|---|---|---|
| Span/Mercury, Noul/Choice/contexte explicite | Sur un seul document : F1 0 / 58,33 / 40,00 / 47,06. Mercury historique reste partiel ; D15 donne 32,39. | Smokes et diagnostic seulement ; aucune victoire de corpus déduite de ces scores. |
| R1a, définition ESC ; R1b, mentions A/B balisées ; R1e, exemples récupérés | Sur DEV-SCREEN : F1 43,30 / 44,11 / 43,26, contre 42,80. Aucun ne passe le critère d’AP préenregistré. | Petits signaux de formulation, aucune promotion. |
| R1c, une question par paire non ordonnée | F1 41,93 contre 42,80 ; moitié moins de requêtes ; AP en baisse. | Économie réelle d’allocation, qualité préservée non établie. |
| R1d, plusieurs questions dans un appel | Le contrôle d’invariance échoue ; pas de test DEV complet. | Ne pas traiter les sorties comme des signaux indépendants. |
| R2a, symétrisation et seuil 0,65 choisi sur DEV | DEV : 39,26→42,04 ; sur CONF40 : 34,79→28,29, soit −6,50 [−9,94 ; −2,95]. | Abandonner cette configuration finale. Le gain DEV ne s’est pas transféré. |
| Coréférence gold, diagnostic oracle | DEV25 d1 : 39,26→60,88 avec seuil hors document et chaînes gold. Sur TEST-S40, fusion 48,64→54,60 sous un autre oracle. | Marge diagnostique, jamais une méthode disponible en déploiement ni un gain additionnable. |
| R2b, coréférence prédite et fermeture | DEV25 : F1 causal 29,27 à seuil 0,5, contre 39,26 ; 37,70 avec seuil hors document, contre 42,04. La fermeture augmente fortement les faux liens d’identité. | Arrêter la propagation actuelle. |
| T2 et J1, autres groupes et accord de deux modèles | T2 : précision des liens 54,86 % sur TEST-S40 ; J1 : 53,31 %, sous son minimum fixé de 85 %. | L’accord entre lecteurs ne fournit pas ici les groupes de précision exigée. Pas de propagation promue. |
| Rationales dans les démonstrations T1 | Luna 44,34 contre 46,02 ; Δ −1,67 [−3,85 ; 0,85]. | Pas de gain confirmé des rationales. |
| R4-A, consigne de sémantique événementielle | DEV44 fusion : −3,006 points. | Ne pas reprendre cette consigne telle quelle. |
| R4-B, exemples structurels et négatifs explicites | DEV44 : +1,106 [−0,676 ; 3,585], critère non atteint. | A motivé N3/N7/N9 ; ne reste pas une amélioration acquise. |
| Rectangles, profils et copies de relations | Top-100 historique : 15 vrais, contre 16 pour le score seul sans rectangle ; un classement alternatif des rectangles en trouve 17. Même un filtre oracle du top-100 historique ne donne que +1,18 point. | Arrêter comme programme autonome aux budgets étudiés. |

Sources : [bilan des premières expériences](../SOURCES.md#annexes-locales), [round 3](../SOURCES.md#annexes-locales), [J1/J2](../SOURCES.md#annexes-locales), [synthèse antérieure vérifiée](../SOURCES.md#annexes-locales), [décision rectangles](../SOURCES.md#annexes-locales).

**Deuxième famille — nouveaux juges, décomposition et ensembles**

Les comparaisons K utilisent le même DEV10, dont la référence fusion vaut 50,00.

| Piste | Résultat | Décision |
|---|---|---|
| Harness², édition automatique du harness | H2-D : +6,41 sur sept documents, surtout une symétrisation déjà connue ; H2-L : +0,43 incertain ; H2-F : +1,46 [−4,23 ; 8,12]. | Aucun nouveau gain robuste au-dessus de la fusion ; ne pas reprendre la boucle ouverte d’édition. |
| J2, juge sur bande de scores incertains | Principal DEV44 : 47,32→45,69. Variante secondaire avec poids et seuil ajustés : 49,31. | Le signal secondaire a déjà reçu sa suite J3 ; il n’est plus « en attente de confirmation ». |
| J3-A, règle de moyenne figée issue de J2 | RES30 corrigé : 36,07→36,35 ; +0,28, IC97,5 par topic [−3,40 ; 2,90]. | Gain non confirmé. |
| J3-B, calibration logistique supervisée | RES30 corrigé : 35,99 contre 36,07. | Pas de gain ; branche comportant un apprentissage supervisé. |
| J4-A/B, calibrations supervisées des scores, avec ou sans structure | DEV44 : 46,76 et 45,19, contre 47,32. | Échecs exploratoires ; hors programme futur sans entraînement. |
| KA, cascade de types de relations | 44,89 contre 50,00. | Arrêter cette cascade. |
| KB, proposeur centré sur l’événement, union à 2/3 | 46,74 contre 50,00. | Les vrais ajouts ne compensent pas les faux positifs. |
| KC, décision entre événements puis copie aux mentions | 39,42 contre 50,00 ; précision des liens de coréférence 38,13 %. | Arrêter cette version. |
| KD, ensemble des nouveaux signaux K | 42,42 contre 50,00. | La diversité combinée n’apporte pas un gain utile. |
| K1, décision Luna relation par relation | 40,41 ; combinaison figée avec la fusion 42,55, contre 50,00. | Ne pas répéter cette formulation générique par paire. |

Les scores J3 bruts des journaux sont invalides à cause d’une convention gold asymétrique ; seuls les résultats corrigés ci-dessus sont utilisables. La correction ne modifie ni les prédictions ni les règles. Elle change la représentation des labels pour respecter la convention symétrique prévue.

Sources : [Harness²](../SOURCES.md#annexes-locales), [J2](../SOURCES.md#annexes-locales), [J3 corrigé](../SOURCES.md#annexes-locales), [J4](../SOURCES.md#annexes-locales), [preuves K et R4](../SOURCES.md#annexes-locales). Les marqueurs des exécutions relues sont conservés dans [l’index de journaux actualisé](../SOURCES.md#annexes-locales).

**Troisième famille — explorations des agents et leurs prolongements**

| Piste | Résultat | Conclusion |
|---|---|---|
| M0, permutations d’IDs, masques de candidats et combinaison MP | Deux documents : référence 36,92 ; +O 44,12 ; +P 34,29 ; +M 37,14 ; +MP 42,86. Contraste principal M−O : −6,97. Une union O+MP postérieure atteint 49,32 avec davantage de calcul. | Les masques changent les relations retrouvées ; l’union favorable était post hoc et n’établissait pas un gain à coût égal. |
| M1, diversité complémentaire à allocation préenregistrée | Vingt documents/topics : P0 42,50 ; contrôle C 42,39 ; hybride H 41,09. H−C : −1,302 [−3,646 ; 0,170]. H ajoute 17 TP et 97 FP. | Signal pratique défavorable. Statut gelé INCONCLUSIF à coût égal : indice H/C 0,8483 et 18 troncatures. Arrêter l’union actuelle. |
| Alias virtuels A/R, agent 6 | Un document, 14 réponses valides : A 16,00 contre R 18,18 ; mêmes deux TP, trois FP nets supplémentaires ; aucun handle dans les arêtes émises. | Signal local défavorable ; pas de pilote complet. Faible priorité, hypothèse générale non réfutée. |
| Transport local P1 v1.1, agent 1 | 44 requêtes ; cinq troncatures ; 26 certificats structurels utilisables ; pas de comparaison finale des quatre bras. | Échec technique et de volume ; aucun gain F1 du mécanisme mesuré. |
| Transport N2, nouvelle version automatique courte | 62 réponses complètes ; 24/26 sources soutenues par le modèle ; neuf candidats ; zéro ajout final dans les quatre bras. | Delta F1 exactement nul par identité des prédictions, sans lecture du gold ; précision des ajouts indéfinie. Comparaison trop peu productive pour établir l’effet propre du transport. |
| Concurrence d’attribution CTB automatique | Sur 40 unités : Haiku 69,14 ; F 65,88 ; A 66,67. A−F à couverture égale : zéro. Une seule unité active la concurrence. | Aucun apport propre démontré ; arrêter ce prototype. |
| Placebos automatiques V0 | Six appels d’édition, six refus, zéro jeu cible+placebos accepté, aucune extraction sous transformations. | Faisabilité de cette construction insuffisante ; effet du contraste non mesuré, pas un F1 nul. |
| Qwen3.8 documentaire, premier point de contrôle historique | Les neuf premiers documents CTB complets donnaient71,43 ; ce point est antérieur aux reprises et aux pilotes de novembre. | Score conditionné à la complétion ; les états courants et les nouveaux panels Qwen sont décrits plus bas. |

Sources : [M0](../SOURCES.md#annexes-locales), [M1](../SOURCES.md#annexes-locales), [alias](../SOURCES.md#annexes-locales), [P1](../SOURCES.md#annexes-locales), [N2](../SOURCES.md#annexes-locales), [attribution CTB](../SOURCES.md#annexes-locales), [placebos](../SOURCES.md#annexes-locales), [Qwen partiel](../SOURCES.md#annexes-locales).

**Quatrième famille — N1 à N10 : ce que les nouvelles expériences ont changé**

N2 figure déjà dans la famille transport ci-dessus ; il n’est pas une deuxième dépense ou une deuxième étude dans les totaux.

| Étude | Question et périmètre | Résultat principal | Décision |
|---|---|---|---|
| N1 | Filtrer automatiquement les candidats ordinaires/MP par preuves et ancrage, DEV20 | H filtré 42,340 contre P0 42,497 et C filtré 42,950 ; H−C −0,610. Ajouts H : 5 TP/22 FP, précision 18,52 %. | NO-GO pour ce filtre avec cette admission MP. |
| N2 | Transport local, neuf substitutions | Zéro ajout final dans les quatre bras. | Faisabilité insuffisante ; ne pas relancer le même générateur lexical. |
| N3 | Exemples structurels + négatifs explicites, DEV12 | 45,473→49,513 ; +4,040 [−0,950 ; 10,428]. | Premier signal encourageant mais incertain ; sa réplication est N7. |
| N4 | DeepSeek documentaire contre Luna, douze documents | Luna 36/36 réponses complètes ; DeepSeek 24/36 tronquées à 32 768 tokens. | Comparaison de qualité bloquée ; pas de F1 primaire. |
| N5 | Une/trois/sept lectures, caches CONF40 | Moyenne des triples 40,779 contre sept 40,928 : −0,149 ; −57,14 % d’appels Luna. Une lecture perd environ 2,54 points. | Trois lectures restent une piste d’efficacité ; une seule écartée. |
| N6 | Question lexicale CTB contre question générique | 60/146 réponses valides ; 86 HTTP429. | Hypothèse de qualité indéterminée. |
| N7 | Réplication N3, seize documents/topics et deux calendriers | Écarts −3,247 et +4,329 ; moyenne gelée +0,541 [−1,614 ; 3,372]. | Aucun gain robuste ; pas de promotion du paquet N3. |
| N8 | Trois versus sept lectures fraîches, préfixe réellement partagé, huit topics | 35,685 contre 36,364 : −0,679 [−3,174 ; 0,292]. | Non-infériorité à une marge d’un point non établie. |
| N9 | Factoriel choix aléatoire/structurel × positifs seuls/négatifs explicites, dix topics, deux calendriers | 240 réponses complètes. Négatifs seuls +0,503 ; structurel seul −1,041 ; paquet complet −1,152 point contre témoin. Tous les intervalles corrigés incluent zéro. | Aucun candidat ; aucune variante ne gagne dans les deux calendriers. Clore cette version. |
| N10 | Six lectures contre moyenne des vingt triples, caches N7, seize topics | Ordinaire +0,161, IC corrigé [−1,539 ; 2,538] ; contrastif +0,544 [−0,246 ; 1,460]. | Aucun candidat retenu ; augmenter partout le nombre de lectures n’a pas de bénéfice établi. |

Pour N5, la différence fondée sur les comptes moyens vaut −0,152, IC95 [−0,931 ; 0,660] ; elle est distincte de la moyenne des F1. Pour N10, les intervalles sont corrigés pour trois comparaisons, au niveau nominal 98,3333 %. Les IC des lignes ne sont donc pas tous construits de la même façon.

N10 compare aussi six lectures contrastives aux six ordinaires : +1,955, intervalle corrigé [−1,274 ; 5,006]. Ce contraste reste incertain et ne remplace pas la question primaire des deux calendriers de N7.

Les bras contrastifs de N7 gardent les mêmes corps de prompt entre calendriers, mais leur F1 change de 3,487 points. Le contrôle change aussi les démonstrations. Le calendrier et l’ordre de soumission restent des facteurs possibles : on ne peut pas attribuer tout l’écart au seul décodage.

Résultat nouveau : [bilan N9 audité](../SOURCES.md#annexes-locales), [audit des réponses brutes](../SOURCES.md#annexes-locales).

Sources : [N1–N6](../SOURCES.md#annexes-locales), [N7/N8](../SOURCES.md#annexes-locales), [N10](../SOURCES.md#annexes-locales), [registre N9/N10](../SOURCES.md#annexes-locales), [protocole N9/N10](../SOURCES.md#annexes-locales).



## 4. N11–N15 : calcul exact, représentations et validité

| Étude | Ce qui a été effectivement observé | Découverte et limite |
|---|---|---|
| N11, arrêt exact sur caches | Audit d’un arrêt anticipé qui doit préserver le résultat du vote, sans génération nouvelle | Une économie doit être mesurée en appels/tokens évités sous un contrat exact ; cela ne constitue pas un gain F1. Voir fiche61. |
| N12, représentations des occurrences | Smoke de quatre documents ; comparaison de vues documentaire, locale et tabulaire | Le résultat dépend de la validité et de la complétion ; aucune amélioration générale validée. Voir fiche62. |
| N13, Qwen local | Les trois derniers smokes terminés contiennent63/72,69/72 puis68/72 réponses valides ; tous sont `BLOCKED_VALIDITY`, sans lecture du gold | « Run terminé » ne signifie pas « résultat scientifique complet ». Pas de baseline CTB50 finale déduite de ces essais. |
| N14, consigne × exécution bornée | Quatre documents, trois représentations et deux calendriers ; couverture complète non acquise | Les diagnostics D0 sur trois documents et deux liens positifs ne permettent pas de conclure sur CTB entier. |
| N15, contrat court | Six documents,36/36 sorties valides ; medium52,17 contre sans réflexion50,00 | Panel exposé trop petit pour promouvoir une variante. |
| D0/E6, MBR sur N14/N15 | Sur N15, MBR exact par lot reproduit la majorité ; union brute7/12 liens, cinq absents | Le MBR n’invente pas les liens absents de toutes les hypothèses ; sa borne oracle n’est pas une méthode déployable. |

Sources : [fiches 61–73](catalogue-experiences.md#fiche-61), [bilan d’implémentation](../SOURCES.md#annexes-locales), journaux N13 [run e163b810](../results/inventaire-executions.csv) (`e163b810-e9f7-419d-b769-168265cf4c49`) [run 2d6766a2](../results/inventaire-executions.csv) (`2d6766a2-3d10-4a32-a19f-e8009e35efbc`) [run 09873773](../results/inventaire-executions.csv) (`09873773-2908-459d-bced-dc436272fb32`) et N15 [run 1d1a1daf](../results/inventaire-executions.csv) (`1d1a1daf-1388-45de-87c2-4fdd756e10b2`).

L’ancienne reprise « CTB seul » reste marquée en annulation dans l’inventaire, sans métrique finale complète validée. Son long temps de suivi ne constitue pas une durée GPU mesurée. Les plans FULL CTB183/ESC233/MAVEN710, créés ultérieurement, sont trois autres nœuds sans lancement ; ils ne complètent pas rétrospectivement cette baseline.

## 5. S1/S2 et E1–E5 : ce que disent les interventions internes

S1/S2 modifient des directions extraites des prompts hint/demo. Ils changent beaucoup les textes de réflexion, mais rarement les décisions finales sur les documents communément complets. Les doses plus fortes, la couche 40, la direction opposée et un témoin aléatoire n’établissent pas un bénéfice spécifique. Plusieurs scores historiques excluaient des documents tronqués : ces valeurs ne sont pas des performances sur le panel entier. [Revue et limites](../SOURCES.md#annexes-locales), [fiches 66 et70–72](catalogue-experiences.md#fiche-66).

Pour E1/E2/E3/E5, le tableau emploie le replay strict : **47 documents, réponse après le dernier `</think>` uniquement, sans réponse finale traitée selon le contrat du scorer**. Les matrices de l’audit ont été recomptées pour cette mise à jour. Elles diffèrent de certains anciens paragraphes, y compris de corrections intermédiaires. Ne pas reprendre leurs chiffres sans préciser le périmètre.

| Étude sur CTB47 | Contrôle F1 | Intervention F1 | Lecture défendable |
|---|---:|---|---|
| E1, décodage et réflexion bornée |40,38|Température0,6 :44,04 ; budget 8k :50,00|La fermeture au budget récupère des réponses finales ; supériorité générale non confirmée |
| E2, arrêt par continuations en accord |33,33|32,69|Pas de gain F1 ; l’identité rapportée sur une ancienne intersection ne vaut pas sur les 47 documents |
| E3, retrait de la composante d’hésitation |35,29|Intensité0,5 :33,66 ; intensité 1 :25,26 ; aléatoire 32,99|Aucun bénéfice spécifique démontré |
| E5, édition de têtes |25,53|Édition33,33 ; têtes aléatoires35,64|Le témoin aléatoire fait mieux en valeur centrale ; pas de mécanisme de décision identifié |

Source prioritaire : [audit strict avec matrices documentaires](../SOURCES.md#annexes-locales), confronté aux journaux E1 [run a5f034d1](../results/inventaire-executions.csv) (`a5f034d1-5e13-4d2f-9b7a-3daf00cafe79`), E2 [run 5ce8a7ea](../results/inventaire-executions.csv) (`5ce8a7ea-342a-40ca-814b-01de968890a9`), E3 [run 865d312c](../results/inventaire-executions.csv) (`865d312c-0ad3-4d0a-a46e-d3964470826f`) et E5 [run 25b64a47](../results/inventaire-executions.csv) (`25b64a47-33b5-44b8-bdb4-1b189df73bfb`). Les anciennes sorties des journaux ne sont pas silencieusement réécrites ; [les divergences sont documentées](corrections-et-provenance.md).

E4 est une expérience différente : une **sonde supervisée** apprend une régression logistique sur les états résiduels. Après réflexion, elle donne51,61 contre Qwen7 à44,44 sur les mêmes38 documents, soit+7,17 [−4,29 ; +18,97]. Neuf documents sans traces compatibles restent hors de ce bras. Le résultat indique un signal décodable, sans supériorité confirmée et sans comparer proprement avant/après sur un panel identique. Dans R1, sa moyenne retombe à40,00 malgré AUROC0,890. [run 3aef44d2](../results/inventaire-executions.csv) (`3aef44d2-eb82-41a1-bef0-f7b14a99482f`) [run dd5cebc4](../results/inventaire-executions.csv) (`dd5cebc4-f7de-400d-b1d4-75b2372ad530`).

**Inférence limitée :** ces expériences n’établissent pas que l’hésitation mesurée soit la cause principale des erreurs causales. Elles ne prouvent pas non plus que le raisonnement n’a aucun rôle. La dispersion des contrôles exige des comparaisons A/A et des moteurs figés ; elle ne peut être attribuée à la seule composition des lots sans test supplémentaire.

<a id="qwen-documentaire"></a>
## 6. Qwen documentaire : diversité, budget et transfert

### 6.1 CTB47 : la référence conservée est le vote N1+P3

Toutes les lignes concernent les mêmes47 documents ; la baseline Qwen7 vaut58,91. Les variantes R/N/P de cette section sont celles de Qwen en novembre, distinctes des R1/R2 d1 et N1/N2 Luna d’octobre.

| Recette | F1 | Ce qui a changé et conclusion |
|---|---:|---|
| R1-Qwen, cinq lectures échantillonnées |55,64|Même démonstration, température 0,6 ; la sonde ne bat pas le vote, la tête du vérificateur prévue manque |
| N1-Qwen, cinq démonstrations distinctes |59,70|Meilleur en valeur centrale que R1, mais contraste+4,06 [−1,94 ; +10,89] |
| N2-Qwen, relecture des propositions R1 |60,15|Gain central sur ses entrées, incertain ; pas reproduit par les deux relectures suivantes |
| P1, relecture des propositions N1 |56,06|La relecture ne maintient pas le bénéfice ponctuel de N2 |
| P3, cinq lectures à20k |57,14|Moins de fermetures forcées, sans progrès du vote à cinq lectures |
| Réunion N1+P3, dix lectures |**61,19**|Contre Qwen7 :+2,28 [−2,09 ; +8,44] ; référence de travail, pas victoire confirmée |
| P4, relecture des propositions P3 |57,14|Mêmes comptes agrégés que ses entrées, sans gain F1 |
| VC1, filtre élargi à47 documents |60,61|Le petit gain du filtre sur CTB20 ne se maintient pas sur l’extension |
| MV2, majorité de cinq vues |56,25|Les vues corrélées ne surpassent pas le vote documentaire |

Sources : [run dd5cebc4](../results/inventaire-executions.csv) (`dd5cebc4-f7de-400d-b1d4-75b2372ad530`) [run 9876c9f5](../results/inventaire-executions.csv) (`9876c9f5-35ae-4b15-b691-3d377cac16b2`) [run e1f30751](../results/inventaire-executions.csv) (`e1f30751-ad2b-4aeb-acde-37cc7d690781`) [run f5d5f69e](../results/inventaire-executions.csv) (`f5d5f69e-1c35-455f-80bd-9cf89f600071`) [run 3d8a86a3](../results/inventaire-executions.csv) (`3d8a86a3-8b79-4aab-acaa-96b5ddbb6f82`) [run b2dc2418](../results/inventaire-executions.csv) (`b2dc2418-a64f-498f-b116-4a095360b975`) [run 89900552](../results/inventaire-executions.csv) (`89900552-7067-4e64-926f-625bcf9201fb`) [run 35ecb7c3](../results/inventaire-executions.csv) (`35ecb7c3-ac28-45fe-a470-122bb84e3913`). [Rapport VC1](../SOURCES.md#annexes-locales).

La variante R3-Qwen élargissant les règles aux buts et à l’habilitation a été arrêtée après son premier passage : F1 29,46, TP66/FP242/FN74. C’est un signal défavorable à cette formulation, sans résultat du vote à trois passages initialement prévu. [run fdfdbcc2](../results/inventaire-executions.csv) (`fdfdbcc2-5a14-4ba0-9e1c-17c9ccfa8593`).

### 6.2 ESC40 : la correction directe reste une piste, sans gain confirmé

| Recette sur TEST-S40 | F1 binaire autonome | Conclusion |
|---|---:|---|
| R2-Qwen, trois lectures à10k |41,81|Sous les références historiques appariées |
| P2, mêmes lectures à24k |42,38|+0,58 [−3,64 ; +5,29] contre R2 ; moins de fermetures ne suffit pas à établir un progrès |
| VC2, réparation directe figée |46,07|+3,68 [−0,10 ; +7,48] contre P2 ; développement réutilisé |
| MV2, majorité de cinq vues |46,74|+4,35 [−0,88 ; +9,00] contre P2 ; sélection exploratoire, gain incertain |
| Luna7 historique, mêmes documents |46,02|Comparateur sauvegardé, pas un composant du nouveau système Qwen |
| Fusion historique Luna × d1 |48,64|Reste au-dessus en valeur centrale ; aucun dépassement confirmé par les variantes Qwen |

Sources : [run 6ef55acc](../results/inventaire-executions.csv) (`6ef55acc-155e-4b5a-873b-48c878858e15`) [run 42e7e53b](../results/inventaire-executions.csv) (`42e7e53b-3529-4fda-a1e9-506fe34b732c`) [run 52629af7](../results/inventaire-executions.csv) (`52629af7-60c7-40f1-ae61-51dbc4d538a2`) [run 35ecb7c3](../results/inventaire-executions.csv) (`35ecb7c3-ac28-45fe-a470-122bb84e3913`) ; [comparateurs historiques](../SOURCES.md#annexes-locales).

Les versions avec propagation par chaînes ECB+ gold restent **oracles**, donc hors du tableau autonome. Le gain d’une méthode Qwen sur P2 ne vaut pas gain sur Luna ou sur la fusion. Le score DeepSeek historique sur ESC233 ne se compare pas directement à TEST-S40.

### 6.3 MAVEN12 : le vote simple reste le témoin

Le pilote termine avec vote 29,47 ; filtre29,79 ; réparation26,54 ; filtre+ajouts26,92. Le filtre retire un seul faux lien. MV2 obtient24,78 avec cinq vues,28,06 par consensus et27,64 avec la règle asymétrique. Les intervalles restent larges ; aucun gain de combinaison n’est établi. Les vues ne sont complètes que pour339/2 530 paires, les autres utilisant le repli prévu. Une lecture sur 60 est invalide. [run 71e9dace](../results/inventaire-executions.csv) (`71e9dace-b52a-4774-8332-a429a640a587`) [run 35ecb7c3](../results/inventaire-executions.csv) (`35ecb7c3-ac28-45fe-a470-122bb84e3913`).

<a id="verification-et-decomposition"></a>
## 7. Vérification, fiches, décomposition et graphes

Les résultats de cette famille servent surtout à distinguer **la couverture des candidats**, **leur discrimination** et **le coût du jugement**.

| Campagne | Comparaison utile | Résultat interprétable |
|---|---|---|
| V1/V2, probabilités de labels sur toutes les paires | CTB47, vote N1+P3 conservé | F1 brut29,37/29,58 ; meilleur hybride V2 47,27 contre 61,19. Un bon AUROC ne garantit pas le F1 sur les liens rares. |
| C1, contexte local/complet et réflexion0/2k | CTB20/ESC12, deux ordres | La réflexion améliore les blocs par rapport à la sortie directe ; le contexte complet n’a pas d’avantage stable. Forte sensibilité à l’ordre. |
| C2, fiches réelles/permutées et réflexion4k | Même panel, contrôles C1 réutilisés | Sans fiche, CTB : 53,85/58,82 contre 40,00/50,85 à2k ; toujours sous vote 65,00. Les fiches ne battent pas clairement les fiches permutées. |
| C3 API, direct/preuve | CTB20 : vote 65,00 ; ESC12 : P2 35,50 | CTB : 54,17/59,57 ; ESC : 41,05/39,69. La preuve ne bat pas le direct de façon stable. |
| AP1, décomposition contre direct à budget égal | ESC12 | Décomposé39,20 contre direct40,44 ; delta−1,24 [−5,28 ; +1,15], entrée totale2,32 fois plus grande. Pas de gain démontré de la décomposition. |
| AP2, liens unanimement rejetés | CTB20/ESC12 | Zéro vrai lien récupéré, un faux ESC ajouté ; certains liens gold étaient pourtant dans les candidats interrogés. |
| AP3, répétition technique |32 décisions ESC | Sept changent avec les lots réorganisés ; cause non isolée. Ce n’est pas une nouvelle mesure F1 du corpus entier. |
| AP4, extension de la réparation figée | Huit documents ESC ajoutés aux douze initiaux | Gain central+1,93 [−4,81 ; +5,66] sur les huit ; pas de confirmation. Total20 :47,45 contre 43,70, avec données de sélection incluses. |
| QG1/QG2, choisir ou éditer un graphe | CTB20/ESC20, référence après AP4 | QG1 :52,17/46,61 ; QG2 :54,55/47,12, contre 65,00/47,45. Aucun ne bat la référence conservée. |

Sources : journaux V1 [run c2a7c92c](../results/inventaire-executions.csv) (`c2a7c92c-7bf3-496d-b239-d343bc6140e2`), V2 [run 682c9c50](../results/inventaire-executions.csv) (`682c9c50-aa02-4611-a125-39ccd5893626`), [C1](../SOURCES.md#annexes-locales), [C2](../SOURCES.md#annexes-locales), [C3](../SOURCES.md#annexes-locales), [AP1](../SOURCES.md#annexes-locales), [bilan AP](../SOURCES.md#annexes-locales), [bilan QG](../SOURCES.md#annexes-locales).

Le filtre C2 paraissait favorable sur CTB20 parce qu’il retirait **un seul faux lien**. Sur les 27 documents ajoutés de VC1, il retire aussi un vrai lien ; le score total ne progresse plus. C’est un exemple à montrer pour distinguer un mécanisme plausible, un effet local et une amélioration qui se transfère.

**Inférence :** dans ces essais, les renseignements supplémentaires et les contraintes de preuve n’apportent pas une séparation assez fiable des vrais et faux liens pour améliorer régulièrement le F1. Cela ne réfute pas toute représentation d’événements ou toute vérification par preuve ; cela clôt les versions mesurées.

<a id="poids-et-diversite"></a>
## 8. W1–W6 : édition des poids, économie de tokens et contrôles

La correction W est un ajout réversible de rang1 au modèle quantifié. Les deux directions sont « conclusion » et « rappel » ; quatre témoins aléatoires sont conservés. L’extraction et la sélection utilisent des traces et, pour certaines constructions, les annotations existantes. Aucun gradient d’entraînement ne transforme ce protocole en absence de supervision. [Protocole](../SOURCES.md#annexes-locales).

| Étape et panel réel | Résultat | Ce qu’on peut conclure |
|---|---|---|
| W1, CTB20 de sélection | Base70,00 ; conclusion 0,10 :73,68 ; meilleur aléatoire 76,54. Réflexion−37,69 %. | Candidate éligible à W2, sans preuve de supériorité spécifique. La direction rappel 0,10 est défavorable. |
| W2, vingt documents réservés avant W1 | Base63,41 ; conclusion 72,73 ; delta+9,31 [−1,72 ; +22,62] ; meilleur aléatoire 76,92. Réflexion−48,90 %. | Critère primaire et spécificité non satisfaits. Série arrêtée sans promotion ; économie de tokens secondaire. |
| W3, recombinaison CPU | Vote des six lectures :72,97 sur W1,63,41 sur W2, à environ5,7×/5,4× les tokens d’une base | La complémentarité ne devient pas un gain fiable ; absence de témoin à même nombre de lectures sans intervention. |
| W4, CTB : 47,96 variantes | Sans intervention55,64 ; avec 54,14 ; référence 61,19 ; meilleure nouvelle variante59,54 | Aucune des 95 variantes ajoutées ne bat la référence. Steering−sans−1,50 [−7,31 ; +4,02]. |
| W5, **ESC4**, titre hérité « ESC40 » | Base historique33,33 ; nouvelles lectures sans 44,00, avec 37,21 ; meilleure combinaison54,17 | Maximum parmi96 variantes corrélées sur quatre documents exposés. Aucun score ESC40/ESC233 ni effet propre confirmé. |
| W6, **MAVEN4**, titre hérité « MAVEN24 » | Vote sans 0,00 ; avec 36,73 ; meilleure réparation41,94 | Le journal final compte16/20 lectures sans et9/20 avec invalides. Contraste confondu par le format, pas une preuve de gain du steering. |

Sources : [W1/W2 et vérifications](../SOURCES.md#annexes-locales), [W3](../SOURCES.md#annexes-locales), [W4](../SOURCES.md#annexes-locales), journaux W5 [run aa041b15](../results/inventaire-executions.csv) (`aa041b15-ebb2-4cd1-8bbf-079aa72ff648`) et W6 [run 95133280](../results/inventaire-executions.csv) (`95133280-621a-4bb2-87cf-29aef6dca1ae`). Le décompte d’invalides W6 diffère d’une note antérieure : [écart explicite](corrections-et-provenance.md).

Dans W4, la vérification examine100 paires, mais seules44des 70relations gold non orientées sont présentes dans le pool proposé. Les26 autres restent inaccessibles à ce vérificateur. Il s’agit du plafond de rappel de **ce pool**, pas d’un plafond de F1 du modèle. Les cinq nouvelles lectures avec intervention émettent31,84 % de tokens de moins ; aucune non-infériorité F1 ni réduction équivalente du temps n’est établie.

## 9. Travaux encore ouverts et résultats qui n’existent pas encore

État à l’instantané documentaire, sans nouvelle relance ni annulation effectuée pour ce rapport :

| Travail | État observé | Ce qui manque pour un résultat présentable |
|---|---|---|
| W7 ESC50 | Exécution enregistrée active ; journal local non disponible au contrôle | Résumé final, panel complet, validité, comptes et comparaison appariée. Aucun score extrapolé d’ESC4. |
| W8 MAVEN50 | Exécution enregistrée active ; prévue après ESC, schéma de paire contraint | Distinguer attente et inférence, vérifier la correction du format et produire les mesures finales. |
| Ancienne baseline CTB seule | Annulation encore en cours dans le suivi | Pas de résultat final complet validé ; ne pas supposer le calcul actif à partir de cet état. |
| FULL CTB183 | Nœud planifié, zéro run | Implémentation du contrat complet et contrôle long contexte avant lancement. |
| FULL ESC233 | Nœud planifié, zéro run | Même exigence ; figer la recette et identifier l’exposition des documents. |
| FULL MAVEN710 | Nœud planifié, zéro run | Même exigence ; séparer pilote12 et698 autres, sans indépendance globale présumée. |

Références actives : [run 7d071a42](../results/inventaire-executions.csv) (`7d071a42-c810-4e3b-8ba7-a74534a35b01`) [run e8d91d5c](../results/inventaire-executions.csv) (`e8d91d5c-4c3d-4f9c-a78a-9f13898b138f`) ; ancienne baseline [run b41468ee](../results/inventaire-executions.csv) (`b41468ee-d81e-4d4c-9df7-1f51309dbde9`). Ces liens établissent l’identification des runs, **pas une mesure de résultat**. [Protocole W7/W8](../SOURCES.md#annexes-locales), [plan des corpus complets](../SOURCES.md#annexes-locales).

Les valeurs DeepSeek63,3/46,6/31,1 concernent les anciens corpus complets CTB183/ESC233/MAVEN710. Seuls les scores finaux historiques sont disponibles pour la campagne complète prévue ; aucune comparaison appariée nouvelle n’est possible avec ces seuls nombres. Ce sont des repères descriptifs, avec révision et scorer historiques à vérifier. Aucun score Qwen de pilote ne doit leur être opposé comme preuve de supériorité.

## 10. Coûts et limites à conserver dans la présentation

Il n’existe pas ici de total financier homogène et vérifié pour les 190 exécutions. Les campagnes utilisent des services payants, des GPU locaux, des caches, des reprises et parfois des temps de suivi périmés. Additionner ces durées ou multiplier un taux inconnu produirait un coût fictif.

La campagne AP0–AP4 rapporte5 374 réponses logiques,5 600 685 tokens de réflexion,255 062de réponse et14 602 908d’entrée pour ses étapes achevées. Elle exclut les votes historiques, C2/C3, les smokes et le surcoût des tentatives échouées : ce n’est pas le coût de déploiement complet. Le temps persistant6h10min45inclut analyses et attentes. [Périmètre de coût AP](../SOURCES.md#annexes-locales).

Pour tout futur transparent « qualité/coût », rapporter ensemble : population évaluée, documents/lectures valides, F1 et incertitude, coût total avec les sources du vote, coût incrémental, tokens d’entrée/réflexion/réponse et temps actif distinct du temps d’attente. Une réponse servie depuis le cache reste une inférence nécessaire à un nouveau déploiement.

Les diagnostics historiques de bruit entre passages complètent les bootstraps documentaires, mais ne donnent pas une marge universelle identique sur tous les panels et moteurs. Ils restent archivés dans les notes de variance ; les intervalles des tableaux ne doivent pas être décrits comme couvrant toute la stochasticité future.

<a id="trame-presentation"></a>
## 11. Trame pour une présentation de 10–15minutes

| Partie | Message à faire passer | Support à reprendre | Limite à dire oralement |
|---|---|---|---|
| 1. Question | Peut-on identifier les liens causaux entre événements d’un document sans fine-tuning lourd ? | Tâche, entrées et métrique, section2 | Mentions et candidats sont fournis ; direction hors critère principal |
| 2. Démarche | Explorer formulation, représentation, lecture multiple, vérification et interventions internes | Catalogue117 fiches et familles historiques |117 nœuds ne signifie pas117 tests indépendants |
| 3. Références | Les lecteurs et leur fusion constituent des témoins utiles | Tableau historique et section6 | Corpus, panels et régimes de supervision différents |
| 4. Couverture | Les liens jamais proposés restent hors d’atteinte du filtre | AP2 et pool W4 | Borne du pool précis, pas plafond du modèle |
| 5. Sélection | Ajouter des explications ou des vues peut accroître les faux positifs | C2, AP1 et QG | Séparer signal local et confirmation |
| 6. Calcul | Davantage de raisonnement ne donne pas automatiquement un meilleur vote | P2/P3 et C1/C2 | Effets propres au protocole ; moteur parfois modifié avec le budget |
| 7. Intervention | Raccourcir les traces est possible, améliorer la tâche de façon spécifique reste non établi | W1/W2/W4, témoins aléatoires | Tokens ≠ temps ; W2 ne passe pas les critères |
| 8. Bilan et suite | Conserver les témoins, fermer les versions négatives, attendre les extensions et une validation propre | Sections9–10 | Pas de score anticipé ni de réserve supposée intacte |

Formulation de clôture possible : « Nous avons surtout appris à distinguer la couverture, la discrimination et le coût. Plusieurs interventions changent le comportement ou le score local ; leur transfert et leur bénéfice propre doivent être démontrés séparément. »

Pour une présentation courte, choisir les comparaisons répondant à la question du public et garder les 117 fiches en annexe. Le tableau des tentatives échouées sert à expliquer la reproductibilité, pas à compter des hypothèses scientifiques réfutées.

## 12. Traçabilité de cette mise à jour

L’instantané distingue101 exécutions terminées,49échouées,25annulées,2 annulations non confirmées à distance,10 suivis perdus,2 exécutions actives et1annulation en cours. Ces catégories décrivent le suivi technique, pas101résultats scientifiquement concluants. Des sorties valides existent dans certains suivis perdus ; certains runs terminés ont au contraire échoué au gate de validité.

187 journaux locaux sont localisables et ont été parcourus par l’indexeur ; trois ne le sont pas : les deux extensions W7/W8 et une soumission ancienne échouée au transport. Les passages utiles des résultats mis en avant ont été confrontés aux rapports et exports corrigés. Ce travail ne réexécute pas toutes les inférences, tous les tests historiques ni chaque bootstrap. Les recomptages effectués concernent les matrices structurées de l’audit strict et la cohérence documentaire.

L’[inventaire des expériences](../SOURCES.md#annexes-locales) conserve les identifiants, descriptions, branches et runs ; l’[index des journaux](../SOURCES.md#annexes-locales) conserve les chemins locaux, tailles, SHA-256 et lignes de preuve. Le [contrôle documentaire](../results/verification-documentaire-source.json) vérifie couverture, identifiants, liens et recomptages. Les descriptions sont un historique de travail qui peut contenir des états anciens : le journal final et le protocole effectif priment.

Les anciens rapports datés restent des archives, avec leurs périmètres et dates. La synthèse du 29 octobre avant révision est conservée dans [son archive](../SOURCES.md#annexes-locales). La présente rédaction n’a déclenché aucune génération de modèle, aucun entraînement, aucune annotation humaine et aucune modification d’expérience.
