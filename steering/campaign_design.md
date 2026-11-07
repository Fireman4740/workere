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
