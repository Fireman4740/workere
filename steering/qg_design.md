# QG1–QG2 : comparaison de propositions causales anonymes

9 novembre 2026. Même modèle Qwen3.8-27B W4A16, même serveur batch local,
aucun entraînement, aucune modification de poids, aucun appel externe.

AP4 a fixé une politique de référence : vote historique CTB, correction directe
ESC. Son avantage ESC reste exploratoire et l’instabilité observée dans AP3
n’est pas résolue. Les nouveaux nœuds héritent de cette politique et de ses
prédictions ; ils ne présentent pas AP4 comme un gagnant confirmé.

## Hypothèse et contraste

Comparer des alternatives explicites peut aider Qwen à sélectionner les bons
liens parmi les propositions déjà disponibles. Le contraste principal est :

- **QG1 / select** : choisir une proposition complète parmi plusieurs graphes.
- **QG2 / edit** : examiner exactement les mêmes propositions, puis décider
  séparément chaque lien pour permettre une recombinaison.

Même budget de 2 048 tokens de réflexion et 256 de réponse par vue. Ce sont des
plafonds égaux ; les coûts effectifs peuvent différer et seront rapportés.

Cette expérience diffère des vérificateurs V1/V2 à un token, du jugement direct
AP1 et des relectures documentaires N2/P1/P4 : les propositions sont anonymes,
sans nombre de votes, et comparées par petits blocs structurés autour d’un
événement. Les deux variantes sont des nœuds frères sous AP4.

## Données et construction sans gold

CTB : les 20 documents initiaux de la campagne AP, référence à 65,00 F1.
ESC : les 20 documents d’AP4, référence corrigée à 47,45 F1. Il s’agit de
développement réutilisé, sans confirmation indépendante. Le score CTB47
historique à 61,19 n’est pas une référence directement comparable à CTB20.

Réutiliser les prédictions conservées AP4, le vote Qwen historique et les deux
ordres C2 plain4096 (avec le screening AP4 pour les huit ESC supplémentaires).
Une réponse de screening invalide conserve la référence sur cette paire.
Ne pas utiliser de chaînes de coréférence gold ni les labels pour générer les
requêtes. Un test vérifie l’invariance des requêtes à leur modification.

Former l’union des liens proposés, puis des blocs disjoints de huit paires au
maximum autour de l’événement de plus grand degré restant. Aucun lien n’est
interrogé deux fois dans un ordre. Chaque système fournit sa projection sur le
bloc ; les projections identiques sont dédupliquées. Ajouter le graphe vide
comme alternative explicite, soit deux à cinq propositions distinctes. Les
paires absentes de toutes les propositions restent inchangées : cette méthode
ne peut pas découvrir un lien qui n’apparaît jamais dans ce pool.

Présenter les graphes sous G0…Gk et les paires sous indices locaux. La seconde
vue inverse à la fois l’ordre des paires et celui des propositions ; le décodage
revient aux paires canoniques avant comparaison. Deux décisions concordantes
sont nécessaires pour modifier un lien ; désaccord ou réponse invalide conserve
la référence. On mesure la présence causale binaire dans les deux directions,
sans revendiquer une amélioration de la direction. Aucune clôture transitive.

## Évaluation et répétabilité

Tous les documents et toutes les paires du scorer restent dans le dénominateur.
F1 micro, précision, rappel, TP/FP/FN, ajouts/suppressions vrais et faux,
accord des vues, abstentions et coût incrémental. Comparaison appariée contre
la politique conservée et contre le vote historique. Le second bras terminé
publie également le contraste entre bras. Bootstrap par document CTB et sujet
ESC, 5 000 tirages, interprétation exploratoire.

Un audit sur huit blocs par corpus répète exactement les prompts dans un nouvel
espace de cache, avec de plus petits lots. Il mesure les changements binaires
après repli, sans prétendre isoler leur cause. Le gain principal reste issu de
la première exécution ; l’audit ne sert pas à choisir les réponses favorables.
L’union des propositions est scorée après génération uniquement comme diagnostic
de couverture du rappel, pas comme une borne de F1 atteignable.

## Exécution et décision de suite

API du conteneur existant, deux lots HTTP de seize prompts au maximum et verrou
commun avec les campagnes antérieures. Identité du modèle/serveur vérifiée,
replay exact des comptes AP4, contrôle préalable de tous les contextes et
smoke test des nouveaux schémas. Pas de troncature des entrées ou d’exclusion
silencieuse. Reprise à partir des réponses terminées et cache HTTP atomique.

Plafond persistant : six heures par bras, douze heures au total. Une interruption
au budget conserve les sorties et ne publie aucun F1 final incomplet. Les coûts
des votes/C2/AP archivés sont exclus du coût incrémental mais restent nécessaires
pour évaluer le coût de déploiement complet.

Après les deux résultats, retenir au plus un candidat par corpus. Une hausse
exploratoire d’au moins deux points et répartie sur au moins trois documents
justifie d’envisager une confirmation, pas de la déclarer acquise. Examiner aussi
les intervalles et la répétabilité. En l’absence de gain, ne pas ajouter des
lectures identiques automatiquement. Aucune extension ultérieure n’est soumise
par ces scripts.
