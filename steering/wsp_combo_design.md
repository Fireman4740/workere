# W4–W6 : maximiser le F1 avec steering, votes et vérification

Demande : msg_163e2156. Trois nœuds frères sous W2, qui fournit une modification
des poids figée, techniquement vérifiée et moins coûteuse en tokens. W2 n'est
pas promu comme gagnant de qualité. Ses documents réservés ne servent à aucun
réglage supplémentaire ; sa série de confirmation reste close.

## Corpus et références

| Pilote | Panel | Référence historique recalculée |
|---|---|---|
| W4 | CTB47 entier, distinct des documents d'extraction | N1+P3 : 61,1940 F1 |
| W5 | ESC TEST-S40 entier | P2 Qwen local : 42,3818 F1 |
| W6 | MAVEN24 : ancien12 + extension12 | Ancien12 : 29,4737 F1 ; nouvelle baseline sur24 |

Les douze nouveaux documents MAVEN sont tirés avec seed20261111, trois dans
chaque quartile de longueur, en excluant l'ancien12. Les cinq démonstrations
train sont celles du pilote MAVEN précédent. Même révision du jeu, vérifiée
par SHA256. Les démos CTB/ESC excluent tous les identifiants et textes évalués ;
une démo ESC provient d'un autre sujet que le document auquel elle est montrée.
Les sujets ne sont pas tous inédits dans le projet : aucun statut de test
indépendant n'est revendiqué.

## Lectures et fusions

Deux bras : original et conclusion cc0.10, avec exactement les tenseurs W1/W2,
les couches32,34,…,48 et alpha0,10. Aucun apprentissage, nouvelle direction,
recherche de dose ou contrôle aléatoire choisi a posteriori. Cette campagne
mesure l'utilité pratique et ne teste pas la spécificité mécanistique.

Cinq démonstrations distinctes par document, identiques entre bras ; greedy,
8192 tokens de réflexion maximum, puis4096 tokens de réponse maximum. Les
nouvelles réponses utilisent toutes le même décodage en deux étapes, avec
JSON contraint. Le décodage diffère de W2 ; la comparaison appariée pertinente
est donc la nouvelle baseline, pas son ancien score. Une réponse invalide
compte vide et reste dans le dénominateur fixe de cinq ou dix lectures.

Calculs sans nouvelle inférence : base5, cc5, vote mixte10 ; CTB/ESC ajoutent
le vote historique, ses unions/intersections avec les votes5, la majorité de
trois familles, et le vote sur le support historique original augmenté de
cinq lectures base, cinq lectures cc ou des dix lectures nouvelles. Pour CTB,
ce dernier groupe correspond aux votes15 et20. Les anciens dénominateurs
valides et le traitement symétrique ESC sont conservés et décrits dans les
caches. Seuil fixe50%, égalité positive ; aucun balayage de seuil.

## Vérification sélective et ablations

Au plus24 paires candidates par document, parmi l'union des propositions.
Priorité aux désaccords entre systèmes, puis au support le plus proche de50% ;
égalité départagée par hash fixe. Aucun gold n'entre dans ce classement. Le
nombre de paires hors quota est publié. Aucune proposition nouvelle absente
de toutes les lectures ne peut être créée par cette vérification.

Blocs de huit paires, texte source complet, deux présentations inversant les
événements, et deux juges : modèle original puis cc. Chaque appel dispose de
4096 tokens de réflexion et512 de réponse JSON. Les juges ne voient ni les
votes ni les réponses des autres juges. Un changement exige des réponses
valides concordantes après remise en direction canonique. L'abstention
conserve la décision du vote. Les mêmes jugements alimentent les ablations :

- filtre seul par le juge original ;
- réparation ajouts/suppressions par le juge original ;
- ajouts seuls par le juge original ;
- réparation par le juge cc ;
- réparation avec accord des quatre jugements ;
- ajouts seuls avec accord des quatre jugements ;
- ajouts à quatre accords et suppression à deux rejets du juge original.

Chaque politique s'applique directement au vote initial ; aucune cascade ne
réutilise ses propres ajouts pour inventer de nouvelles observations.

## Lecture des résultats et coûts

Objectif : F1 micro de présence causale symétrisée sur les candidats officiels,
comme les références historiques. Rapporter TP/FP/FN, précision, rappel,
prédictions hors candidats et F1 alternatif qui les pénalise ; aucune
coréférence gold n'est utilisée. Résultats par document conservés. MAVEN
ancien12 et extension12 sont séparés en plus du total24.

Contrastes fixés : cc5 contre base5, puis cc5+réparation par juge original
contre base5+la même réparation. Chaque variante CTB/ESC est aussi comparée à
la référence historique sur les mêmes documents. Bootstrap apparié10000,
par document pour CTB/MAVEN et par sujet pour ESC. Intervalles exploratoires
non corrigés ; le maximum parmi de nombreuses variantes corrélées n'est pas
une confirmation indépendante. Aucun gagnant n'est promu automatiquement.

1110 lectures nouvelles ; au maximum1332 appels de vérification. Les coûts
incluent les tokens historiques pour les méthodes qui en dépendent. Une
variante vérifiée doit financer les dix lectures utilisées par le routage
commun, même si son vote final n'en utilise que cinq. Les tokens émis ne sont
ni une mesure de temps mur ni un coût complet de prefill/extraction.

Une RTX3090, modèle local W4A16, batch4, contexte24576, vLLM0.32.0. Préflight
des longueurs avant toute prise GPU, sans troncature silencieuse. Test du
changement des logits et du retour exact au modèle original avant génération.
Ordre CTB → ESC → MAVEN, verrou GPU partagé ; plafonds actifs18h/24h/18h,
hors attente, qui sont des limites et non des prévisions. Caches persistants
avec manifestes, identités des requêtes et contrôle de reprise. Une limite
atteinte laisse les sorties partielles explicitement incomplètes.
