# AP0–AP4 : campagne Qwen locale, 24 heures

Protocole demandé le 8 novembre 2026. Même checkpoint Qwen3.8-27B W4A16,
API locale du conteneur batch existant ; aucun entraînement, aucune édition
de poids, aucun appel OpenAI/OpenRouter. Les références externes sont des caches.

## Contrat et progression

Commande héritée : `bash steering/run_pilot.sh`. `CAMPAIGN=AP` et `AP_STAGE`
dans la configuration de chaque branche choisissent le stade. Les résultats
persistants sont sous `$WORK/out/AP/$AP_GROUP/<stage>`, indépendants du répertoire
temporaire de lancement. Les identités de données, serveur et requêtes sont
vérifiées. Chaque réponse conserve le run et le commit qui l'ont produite.

AP0 attend la clôture de C3 API `00a20ccc-434d-4d4c-a11f-66fbfb86f72c`, puis
rejoue son scoring sur une copie sans modifier la source. Le budget global
commence à ce moment. AP1_direct et AP1_decomposed sont deux nœuds frères ;
AP2–AP4 sont prêts dans le code mais leur lancement et parent sont décidés à
la lecture des résultats, en descendant sur le résultat établi pertinent.

Un verrou partagé sérialise les contrôleurs AP. Chaque contrôleur soumet au
plus deux lots de seize prompts ; le serveur garde sa limite de 32 séquences.
Les plafonds sont persistants : AP0 1h, AP1 **6h partagées par les deux variantes**,
AP2 5h, AP3 2h, AP4 8h, global 24h (marge 2h). Les attentes de dépendances ne
lancent pas de modèle. L'arrêt au budget préserve les réponses et produit
`budget_exhausted` ; il ne publie pas un F1 final sur une sélection de réponses
terminées. Une reprise ne remet pas les compteurs de temps à zéro.

## Expériences et décisions

- **AP0** : vérification exacte des TP/FP/FN par document, qualité des sorties,
  coût amont connu, registre conservateur d'exposition. Aucun document CTB183
  ou ESC40 n'est revendiqué indépendant. Une correction C3 n'est candidate
  que si elle améliore le score du corpus ; égalité entre méthodes départagée
  vers direct.
- **AP1 direct** : 2 560 tokens de réflexion + 512 de JSON par orientation.
- **AP1 décomposé** : faits, mécanisme, explication concurrente indépendants,
  puis synthèse ; chaque appel 512 + 256 tokens, soit également 3 072 tokens
  maximum par orientation. Les entrées répétées et la latence sont comptées.
  Les citations non ancrées deviennent UNKNOWN ; cela n'impose pas N. Une
  étape syntaxiquement invalide fait s'abstenir la décision concernée.
  Les deux variantes utilisent les 396 routes C3 et les deux orientations.
- **AP2** : au plus 128 CTB + 256 ESC paires négatives dans le vote et les deux
  ordres C2. Moitié réservée aux indices lexicaux figés, puis moitié uniforme
  parmi les paires restantes, sélection hashée et tourniquet document/distance.
  Les catégories insuffisantes cèdent leurs quotas à l'échantillon uniforme.
  La distance est estimée par le même découpage de phrases heuristique que C2.
  Une réponse C2 invalide n'est pas un rejet. Meilleure méthode AP1 admissible
  par corpus, sinon direct C3 1024+512. Ajouts seulement, après accord des deux
  orientations, sans modification des règles d'annotation.
- **AP3** : référence, réparation, couverture et combinaison sont comparées
  sur CPU. Une méthode AP1 et les ajouts AP2 deviennent admissibles avec au
  moins +2 F1 et trois documents avec davantage de corrections utiles que
  nuisibles. À défaut de réparation AP1, C3 positif peut être conservé comme
  candidat exploratoire. La meilleure politique par corpus est figée avant
  la génération de stabilité et toute extension. Le test de stabilité porte
  sur au plus 32 paires par corpus, mêmes prompts, lots réorganisés, espace de
  cache distinct imposant de nouvelles réponses ; aucune validation indépendante.
- **AP4** : extension CTB47/ESC40 si la projection conservatrice du coût tient
  dans le budget, sinon +10 CTB/+8 ESC sélectionnés par quantiles de longueur.
  La sélection est persistée et ne change pas à la reprise. Les deux ordres
  de screening C2 plain4096 sont générés sur les nouveaux documents ; le
  coût du routage est inclus. Les quotas de couverture augmentent au prorata
  du nombre de documents. Aucun réglage sur les nouveaux résultats. Si aucune
  politique ne bat la référence, aucune génération d'extension.

## Mesures, limites et reprise

F1 binaire micro sur tous les documents et l'univers candidat canonique,
TP/FP/FN, changements de liens non orientés, abstentions, plafonds, tokens et
temps réel. Bootstrap apparié par document CTB et par sujet ESC, 5 000 tirages.
Les intervalles et gates restent exploratoires après sélection sur développement.
Le contraste primaire AP1 est décomposé moins direct à budget maximal égal.

Le vote N1+P3 archivé est binaire : sa direction n'est pas définie. Le F1
orienté supplémentaire porte donc explicitement sur les seules paires CTB
jugées, et ne doit pas être présenté comme le F1 orienté du système complet.
ESC n'a pas de gold orienté exploitable dans cet export. Les coûts historiques
incomplets restent marqués manquants, jamais remplacés par zéro.

Les sorties des deux étapes HTTP sont mises en cache ; un échec de scoring
ne justifie pas de régénérer les réponses. Les JSONL ne récupèrent qu'une ligne
finale interrompue ; doublons et identifiants étrangers sont rejetés. Les erreurs
HTTP font échouer l'étape, et ne deviennent pas des labels N. Les dépassements
de contexte font échouer la préparation/inférence sans troncature silencieuse.

Validation : 33 tests CPU, dont neuf propres à AP et un parcours synthétique
AP1→AP4, plus les tests C1/C2/C3 réutilisés. Les nombres synthétiques des tests
ne sont pas des résultats scientifiques. Les nouveaux schémas font un smoke
test sur l'API avant AP1. Aucune dépendance ajoutée.

Suite après AP4 : confirmation sur des documents/sujets dont l'absence
d'exposition est auditée ; aucune affirmation de gain indépendant avant cela.
