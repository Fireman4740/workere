# Pilote N15 — contrat de génération sur CTB6

Statut : préparé et validé localement, aucune génération N15 lancée.

## Question et périmètre

Comparer Qwen local avec raisonnement medium et avec `enable_thinking=false`,
sans modifier les poids. Garder exactement les mêmes prompts documentaires
historiques, candidats et démonstrations. Les deux conditions ont un plafond
8192 tokens. Le plafond lui-même diffère de N14 : le témoin medium est donc
régénéré, et ne sera pas présenté comme une reproduction du medium/65536.
Le réglage sans thinking peut encore produire du raisonnement dans le contenu
final car le prompt historique est conservé ; mesurer le temps réellement gagné.

Six documents : wsj_0006, ABC19980304.1830.1636, wsj_0533, wsj_0344,
wsj_1035, wsj_0313. Ils totalisent 189 paires non orientées et 12 positives.
Sélection intentionnelle de développement : courts (44 à 434 tokens bruts),
avec résultats historiques faciles, intermédiaires, difficiles et discordants.
Ce choix utilise des résultats déjà exposés et ne permet aucune confirmation
indépendante. Le document difficile géant de N14 n'est pas évalué ici : aucune
conclusion sur sa faisabilité ou sur les longs documents ne sera tirée.

Les F1 Qwen et Haiku par document sont conservés dans
`historical-context.json`, avec empreinte du CSV source. Ils proviennent d'un
protocole historique différent à sept lectures ; ils fournissent du contexte,
pas le témoin expérimental. Leur support est orienté. Ne pas moyenner les F1
par document pour obtenir le micro-F1 du nouveau pilote.

## Exécution

- 6 documents × 3 démonstrations × 2 conditions = 36 générations maximum.
- Température zéro ; trois démonstrations différentes, pas trois répétitions
  prétendument indépendantes d'un prompt identique.
- Requêtes appariées, ordre mélangé déterministe, une seule en vol.
- Entrée <=16384 tokens vérifiés au tokenizer réel, sortie <=8192 ;
  plafond de sortie total 294912 tokens, et non estimation de consommation.
- Contrôleur limité à une heure, délai d'une requête <=600 secondes.
  Un timeout client ne garantit pas l'annulation immédiate du décodage serveur.
- Première erreur, troncature ou omission de réponse : arrêt sans retry,
  score primaire absent. Aucun échec converti en négatif.
- Aucun changement du serveur, aucun fournisseur externe, aucun fine-tuning.
- Pas de mélange avec des caches au contrat différent ; les historiques évitent
  de régénérer Haiku et les variantes précédentes.
- Ne pas lancer tant que le steering S1 utilise la carte et que le serveur
  Qwen habituel sur 127.0.0.1:18020 n'est pas disponible. Ne pas arrêter S1.

## Analyse gelée

Majorité 2/3 par paire non orientée, une voix par lecture même si les deux sens
sont produits. Les 36 réponses doivent être valides. Geler les décisions avant
lecture du gold. Rapporter TP/FP/FN et micro-F1 global et comptes par document,
temps, tokens de complétion et validité. Comparer le gain F1 et le coût mesuré,
sans promouvoir automatiquement une variante sur seulement six documents.
Pas de seuil optimisé sur ce petit lot ; aucune extension automatique.

## Prochaine expérience scientifique préparée : faits et rattachement

Une fois le contrat retenu, comparer sur ces mêmes documents :
A document brut ; B brut + passages sources sélectionnés ; C brut + fiches
factuelles d'occurrences (participants, action, temps, négation/modalité).
C doit citer des segments exacts du texte et ne doit jamais inventer une
identité événementielle ni propager des liens entre groupes.

Préparation une fois par document, ensuite cache des fiches partagé entre les
lectures ; le témoin B reçoit les mêmes citations que C, ce qui isole davantage
l'effet de leur structuration. Maximum 6 appels de préparation + 54 appels
lecteurs (6 docs × 3 lectures × 3 bras). Réutiliser les 18 appels du témoin A
N15 uniquement si le contrat complet et les corps sont strictement identiques,
sinon produire un témoin frais. Ne pas rejeter silencieusement un document dont
la préparation échoue ; consigner l'échec et ne pas publier de F1 primaire
incomplet. Cette deuxième expérience reste un protocole, pas du code exécuté.

La couverture exhaustive par paires et les grands ensembles de modèles seront
reportés : le premier pilote doit établir une génération locale fiable et
mesurer le coût avant de multiplier les appels.

## Vérifications locales

Préflight des 36 requêtes et six tests : couverture des candidats,
dédoublement des orientations, refus d'une réponse manquante ou tronquée,
refus d'un cache de prompt différent, refus d'un identifiant réponse dupliqué,
micro-F1 calculé à partir des comptes. Syntaxe shell et diff vérifiés.
La tokenisation du template et les sorties réelles restent à tester sur la box.
