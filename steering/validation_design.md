# Validation des combinaisons sur Qwen local — 9 novembre 2026

Les trois nœuds frères prolongent AP4, qui conserve le vote CTB et identifie la
correction directe ESC comme candidat exploratoire. Le filtre conservateur C2
est réintroduit explicitement : CTB20 passe de 65 à 66,6667 par suppression
d'un seul faux lien non orienté. Ce résultat n'est pas une confirmation.

## Contraste commun figé

- A : vote seul.
- B : supprimer un lien initial si les deux écrans C2 valides le rejettent.
- C : corriger les désaccords entre C2 et vote après accord des deux jugements
  directs, y compris sur la direction interne du jugement.
- D : appliquer B aux positifs initiaux et ajouter aux négatifs initiaux
  uniquement après accord positif des deux jugements directs. Les règles ne
  se réappliquent pas séquentiellement à leurs propres sorties.

Tout désaccord ou réponse invalide préserve la référence. Tous les documents
et paires restent dans le dénominateur. Aucune étiquette d'évaluation ne sert
à construire les requêtes, choisir les routes ou régler un seuil. Les scores
sont la présence causale binaire symétrisée, pas une revendication de direction.

## Jeux et réutilisation

VC1 : CTB47, vote N1+P3 archivés. Replay exact du filtre CTB20 avant toute
inférence, puis extension27 séparée et total47. Les réponses C2 plain4096 et
les jugements AP1 directs sont importés uniquement si messages, schéma et
budgets sont identiques. Le budget final C2 original est 384, pas 512.

VC2 : ESC40 TEST-S40, vote P2 archivé. Réutilisation des 12 écrans C2 originaux,
des huit écrans AP4 et de tous les jugements directs identiques. Compléter les
20 autres documents. Rapporter séparément initial12, extension28 et total40.
Le panel AP4 de huit documents a un ordre et un budget final différents : il
est préservé tel quel et tracé, sans le présenter comme une réplique C2 exacte.

Les anciens écrans C2 ont été obtenus en mode natif et les compléments sont
servis par l'API avec un autre protocole numérique. Cette hétérogénéité est
déclarée ; huit prompts exacts par corpus font l'objet d'un audit API frais.
Cet audit ne sert pas à choisir les réponses favorables. Un gain sur les
extensions ou sur l'ensemble réutilisé n'est pas une confirmation indépendante.

VC3 : MAVEN-ERE test, révision 830a58ee7a092d5ca27b4a8c28c18e2c82abf814.
Pilote12 : trois documents par quartile de longueur en tokens, tirage seedé
sans annotations. Nouvelle baseline Qwen API à cinq lectures greedy, cinq
démonstrations distinctes issues du train, 8192 tokens de réflexion et 2048
de réponse structurée. Le train est exclu par identifiant et texte du test.
Une lecture invalide compte vide, majorité fixe 3/5. Les quatre combinaisons
se comparent à cette baseline sur les mêmes documents ; les références
historiques d'autres modèles ne remplacent pas une baseline Qwen appariée.
La génération ne reçoit ni les relations test ni la coréférence gold.

## Serveur, coûts et arrêt

Même checkpoint Qwen3.8-27B W4A16, serveur batch existant, entrée maximale
16384 et contexte32768. Les longueurs sont contrôlées avant chaque étage sans
troncature silencieuse. Deux appels HTTP de seize prompts maximum ; lease
commun avec AP. Aucun chargement GPU local, entraînement ou appel payant.

Chronomètres persistants par corpus pris après le lease : plafonds CTB8h,
ESC16h, MAVEN12h, distincts d'une prévision de durée. File CTB puis ESC puis
MAVEN, attente hors lease. Une interruption garde les caches et ne publie pas
un F1 incomplet comme résultat complet. Tous les appels sont repris par
identité ; les coûts importés sont distingués du coût incrémental. Le coût
complet de déploiement doit aussi inclure les votes et écrans historiques.

F1 micro, précision/rappel, TP/FP/FN, changements vrais/faux, bootstrap apparié
par document CTB/MAVEN et par sujet ESC, 5000 tirages. Les comparaisons A–D
sont exploratoires et leur multiplicité est déclarée. Pas de choix automatique
du meilleur score comme preuve. Un gain réparti, son intervalle et le coût
guident ensuite une confirmation figée. L'évaluation complète 183/233/710 est
une étape suivante conditionnelle, pas lancée aveuglément. L'exposition des
documents doit être auditée avant toute revendication de test indépendant.

La récupération structurelle des démonstrations reste le tour suivant : ne
pas la mélanger à ce contraste de combinaison ni répéter V2 à l'identique.
