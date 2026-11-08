# C1: contexte et réflexion sur blocs de paires

## Hypothèse et portée

Sur le même checkpoint local Qwen3.8-27B W4A16, une décision exhaustive par
blocs de douze paires pourrait mieux exploiter le contexte qu'une réponse
binaire à un token. Ce n'est pas une reproduction de KnowQA ou MEFA. Aucun
paramètre de Qwen n'est entraîné. Le jeu constitue du développement déjà vu.

Plan factoriel apparié : contexte local ou document entier × thinking éteint
ou borné à 2 048 tokens. Deux ordres déterministes des paires sont appliqués
aux mêmes blocs. La réponse finale est un JSON contraint comportant A, B ou N
pour chaque paire ; une réponse invalide s'abstient sur tout le bloc, dont les
positifs manqués restent des faux négatifs. Le texte de réflexion n'est jamais
utilisé par le scorer. Le F1 principal mesure la présence d'un lien ; un F1
orienté CTB est calculé séparément. Les données ESC archivées ne permettent pas
de scorer la direction.

## Sélection et comparaisons

20 documents CTB choisis dans les 47 avec baseline Qwen, par quintiles de
longueur, et 12 documents ESC de 12 sujets distincts. Sélection déterministe
par métadonnées, sans lecture des labels : graine 20261107. Les deux contextes
couvrent les mêmes paires, y compris celles jamais proposées par les votes.
Les fenêtres locales utilisent une segmentation heuristique fondée sur la
ponctuation et conservent toutes les occurrences des mentions. Elles sont donc
approximatives, surtout pour ESC ; leur contenu est vérifiable dans les
requêtes sauvegardées.

Les références sont Qwen7 et N1+P3 sur CTB, puis Qwen P2, Luna7 et la fusion
figée sur ESC. Pas de propagation par coréférence gold. Les intervalles sont
appariés par document ; les 12 documents ESC viennent de 12 sujets distincts,
donc ces unités sont également distinctes par sujet dans ce pilote. Les quatre
contrastes du plan factoriel sont affichés pour chacun des deux ordres.

## Reprise et coût

Les requêtes et les paramètres du moteur sont figés dans un manifeste.
Reprise possible seulement si les empreintes des requêtes, des données, des
paramètres et de l'identité du checkpoint correspondent. Les poids sont
identifiés par noms, tailles et dates de modification ; ce ne sont pas des
empreintes cryptographiques des poids. Les tokens d'entrée incluent la seconde
lecture du contexte après réflexion. La durée est mesurée sur les lots et
répartie entre blocs ; le temps de chargement du modèle est enregistré à part.

Le moteur et la sortie JSON subissent un smoke test GPU avant la génération.
Les 50 premiers blocs de chaque groupe donnent une mesure du débit, puis chaque
groupe terminé affiche un score provisoire sur tous ses documents. Le score final
est produit seulement après couverture complète. Le contexte trop long et les
sorties mal formées restent visibles et comptent comme abstentions.

## Suite conditionnelle

C2 ajoutera les fiches d'événements au protocole C1 retenu, avec contrôle des
arguments permutés. C3 évaluera la fusion seulement après avoir identifié un
signal complémentaire. C4 exige d'abord une borne de sélection calculée sur des
graphes de calibration disponibles ; C5 est un diagnostic mécaniste indépendant
préalable à toute édition de poids. Ces étapes n'ont pas encore de vainqueur ni
de résultat et ne sont pas lancées par avance sur la base des labels du pilote.

## C2 : fiches d'événements et budget par bloc

Les résultats C1 orientent cette suite : en contexte local, 117/168 blocs CTB
et 531/546 blocs ESC ont atteint le plafond de 2 048 tokens. Les paires, textes,
ordres et résultats C1 de ce contexte sont réutilisés après vérification des
empreintes ; aucune génération C1 n'est refaite. Le contexte local a été retenu
pour son coût plus faible et ses résultats de développement, sans conclure à
une supériorité générale sur le document entier.

Quatre conditions : fiches à 2 048 tokens ; mêmes fiches permutées entre
événements du document à 2 048 ; sans fiche à 4 096 ; fiches à 4 096. Le
contrôle direct sans fiche à 2 048 vient de C1. La comparaison fiches réelles
contre fiches permutées maintient le modèle, les blocs, les ordres et le budget.
Le dépassement de budget, les réponses invalides et tous les documents restent
dans le dénominateur. Les intervalles appariés sont calculés par document ; les
12 documents ESC proviennent de 12 sujets distincts.

Le même Qwen extrait une seule fiche par événement depuis le texte, sans label
causal ou coréférence gold : acteur, participant affecté, citation de preuve et
statut de l'événement. Les champs textuels non retrouvés littéralement dans le
document sont écartés ; sans citation retrouvée, le statut devient inconnu. La
présence de la citation ne prouve pas sa bonne attribution à l'événement. Les
fiches et leur qualité sont sauvegardées ; si plus de 10 % des lots de fiches
échouent à produire un objet valide, l'inférence par paire s'arrête.

Le coût des fiches est mesuré par corpus et partagé entre les trois conditions
qui les utilisent. Les sorties partielles et les identités des entrées, du
moteur et du checkpoint sont vérifiées à la reprise. Ce sont des expériences
exploratoires sur des documents déjà utilisés pour sélectionner les méthodes.

Quatre règles de combinaison fixées avant la génération sont également évaluées
sur CPU : accord entre les deux ordres ; ajout de cet accord au vote Qwen ;
filtrage des seuls liens rejetés par les deux ordres ; et majorité entre le vote
Qwen et les deux ordres. Ces nombreuses comparaisons restent exploratoires.

## C3 : arbitrage ciblé, une paire à la fois

Les premières conditions complètes de C2 à 4096 tokens récupèrent davantage de
liens, mais leur précision CTB reste sous celle du vote N1+P3. C3 conserve ce
vote CTB et Qwen P2 sur ESC comme points de départ. Il réutilise les deux ordres
`plain_4096` de C2 pour sélectionner uniquement les paires en désaccord avec
le vote. Une sortie C2 invalide ne constitue pas un rejet. Aucune annotation
gold, fiche C2 ou fréquence de vote n'est transmise au modèle d'arbitrage.

Deux conditions appariées : `direct`, une classification de la paire avec
réflexion ; `evidence`, le même jugement avec extraits citant les événements,
passage soutenant le lien et explication concurrente. Les règles d'annotation
restent celles de chaque corpus. Chaque condition utilise deux présentations
de la même paire, A/B puis B/A, 1024 tokens de réflexion et 512 de réponse.
Le contexte est la fenêtre locale de la seule paire. Cela teste aussi une
allocation plus concentrée du calcul que les blocs C2 à douze paires ; seule
la comparaison evidence/direct isole l'effet de la consigne de preuve.

Contraste principal fixé : F1 du système `evidence/consensus_repair` moins celui
de `direct/consensus_repair`. Comparaison supplémentaire au vote de départ.
Une décision change seulement si les deux présentations produisent le même
label après normalisation de l'orientation. Des sorties invalides, citations
non ancrées ou orientations contradictoires font conserver le vote. Les
citations des deux événements doivent contenir leurs balises complètes dans
une sous-chaîne exacte du contexte. Pour un lien positif, le passage soutenant
le lien doit aussi être présent. Ce contrôle établit une provenance textuelle,
pas la validité sémantique du lien. Une ablation sans contrôle des citations et
une règle d'ajouts seuls sont secondaires, sans sélection du meilleur seuil.

Le routage et les prompts sont construits avant de lire les labels pour les
diagnostics. Un plafond oracle est publié seulement pour mesurer la marge
maximale de réparation dans les paires routées. Il ne pilote ni les requêtes
ni la sélection d'une règle. Les paires hors du routage restent inchangées :
C3 ne peut pas retrouver un lien rejeté par le vote et les deux ordres C2.

Tous les 20 documents CTB et 12 ESC restent dans les métriques, y compris les
documents sans désaccord ou avec erreurs de format. Bootstrap apparié par
document (12 sujets ESC distincts), comptage des ajouts et suppressions vrais
ou faux, coût incrémental incluant les deux présentations. Le coût de déploiement
inclut aussi les générations amont N1/P3/P2 et les deux ordres C2. Aucun
entraînement, pas de réglage du seuil sur ce pilote déjà exposé. Une promotion
exige un gain face au vote et une confirmation élargie figée ensuite.

## C3 API batch : migration demandée vers le serveur persistant

Le run natif C3 est interrompu à la demande de l'utilisateur après 912 réponses
sauvegardées. Ses sorties restent dans son répertoire, sans être mélangées à
celles de l'API. Ce n'est pas un échec scientifique du protocole.

Le nœud C3 API, frère de C3 sous C2, conserve exactement les requêtes,
baselines, schémas et budgets 1024 + 512. Un tokenizer CPU construit les mêmes
préfixes ; deux appels HTTP bruts `/v1/completions` conservent l'arrêt sur
`</think>` puis la réponse JSON contrainte. Deux lots de seize requêtes sont
servis simultanément. La concordance des identifiants, indices de réponses,
empreintes des entrées et du checkpoint est contrôlée. Les erreurs HTTP font
échouer le run plutôt que de devenir des labels négatifs. Les réponses HTTP
complètes sont mises en cache, y compris entre les deux étapes de génération.

Le conteneur `qwen38-27b-rtx3090-batch-1` est le seul moteur propriétaire de la
GPU. W4A16 conservé, W4A8 désactivé, cache KV FP8, état récurrent FP16, cache de
préfixes activé, scheduling asynchrone, 32 séquences actives et contexte 32768.
Ces choix de calcul et de scheduling constituent une variante numérique et de
service ; l'API n'est pas supposée reproduire bit à bit le moteur natif dont le
cache KV était BF16. Aucun poids n'est entraîné ou édité. La configuration du
serveur et les identités des poids/tokenizers sont archivées avec le run.

La mesure principale reste evidence moins direct avec consensus des deux
orientations, puis comparaison au vote de départ. Toutes les sorties finales
et tous les documents entrent dans le scoring. Le temps de chargement du serveur
est séparé du temps de la campagne. Le score oracle n'est jamais une prédiction.

Suite prioritaire : analyser C3 API, comparer Terra 6.1 low sur les mêmes entrées
avec des crédits API Pro activés, puis tester l'arbitrage Qwen/Terra uniquement
si le modèle apporte un signal utile. Toute combinaison sera figée avant une
extension CTB47/ESC40 et une confirmation indépendante par document/sujet.
