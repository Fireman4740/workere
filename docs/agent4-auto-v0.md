# Agent4 V0 automatique — protocole gelé avant appels

27 octobre 2026. L’utilisateur autorise explicitement la suite **sans annotation humaine**,
avec un plafond déjà fixé de **2 USD pour tout le pilote**. Cette modification est
prospective ; aucun certificat humain ni booléen de revue humaine n’est fabriqué.
Les éditions admises seront `AUTOMATIC_PROVISIONAL_NOT_HUMAN_REVIEWED`.

## Population inchangée

- Conserver les 12 paires sélectionnées sans gold, seuil b>=0,7, au plus deux par document.
- Six paires (quatre documents, topics 37/41) ont leurs snapshots gelés disponibles :
  elles seules sont techniquement exécutables. Les six autres restent au dénominateur
  d’attrition. Aucune substitution selon les textes, les labels ou les résultats.
- Aucun champ gold n’est exporté ni chargé ; les démonstrations gold hors topic sont
  celles des requêtes historiques, sans nouveau tirage.

## Étapes et bornes

1. Un appel Luna batch `high`, sortie <=16384, pour proposer la cible, trois placebos et
   des citations littérales par paire techniquement éligible. Le modèle peut refuser.
2. Contrôle automatique strict : ancres littérales exactes, toutes les mentions protégées,
   offsets recalculés, quatre textes distincts, même famille de retrait/neutralisation,
   volumes +/-20% (tolérance d’un token) et proximité comparable (écart de distance
   <=max(300 caractères, 25% de la longueur documentaire)). Pas d’insertion pure,
   de changement de ponctuation des phrases, de renommage lexical ou de dissociation.
3. Un appel distinct au même modèle gelé pour auditer chaque cas structurellement
   admissible, **aveugle aux rôles cible/placebo** (A-D mélangés), au gold et aux sorties
   de l’extracteur. Sortie <=8192. Il doit constater une preuve originale documentée,
   une cible affaiblissant ce support et trois placebos préservateurs ; toute incertitude
   entraîne l’abstention. Les preuves résiduelles sont signalées, pas assimilées à un FP.
   Ce juge automatique ne constitue pas un audit humain ni une preuve de fidélité interne.
4. Sur les cas retenus provisoirement : original x2, cible x1, placebo 1/2/3 x1,
   permutation bijective des IDs x1. Utiliser **exactement** la requête Luna gelée,
   mêmes prompt, démo de passe 0 hors topic, paramètres et parseur. Sortie <=32768.
   La première lecture originale et les variantes sont mélangées dans une vague ;
   la seconde lecture originale est une vague distincte à cache/receipt distinct.
5. Calculer uniquement les votes, s, contraste, écart de renommage et stabilité.
   Verrouiller fichiers et hashes sans joindre le gold. Aucune règle ADD/REMOVE,
   aucun nouveau score d1 ni F1 système.

Maximum : 6 appels éditeur + 6 auditeur + 42 tâches de score = **54 tâches de modèle**,
à l’intérieur du plafond global de 84 tentatives et 2 USD. Les requêtes identiques
pour plusieurs paires d’un même document sont mutualisées **à l’intérieur d’une vague**,
pas entre les deux répétitions originales. Rapporter tâches développées et appels réels.

Les appels éditeur/auditeur sont comptés dans le même registre persistant que
l’extracteur. Prix supérieurs issus du catalogue actuel, réserve d’entrée conservatrice
0,225 USD/M, sortie 0,375 USD/M, marge 25%. Chaque stage entier est refusé si ses
réservations dépassent le solde de 2 USD. Aucun retry de génération automatique.
Coût inconnu conserve la réservation ; aucun coût inconnu n’est annoncé comme nul.
Un POST à issue incertaine n’est jamais renvoyé automatiquement.

## Exécution et gel

Commande héritée inchangée : `bash experiments/jev/run.sh`, sur le backend SSH
configuré ; le script de cette branche appelle `agent4_v0.automated_run`.
Les fichiers d’entrée, prompts auxiliaires et sources du transport sont scellés dans
`agent4_v0/data/auto-v0/protocol.json`. Le registre et les reçus sont persistants sous
`~/.local/share/openresearch/workere-decisions/agent4-auto-v0-2usd` du serveur.
Ce registre devient l’autorité de dépense pour l’autorisation utilisateur de 2 USD ;
le registre local précédemment créé contient zéro tentative et ne doit pas servir
à un second lancement concurrent.

Aucun texte envoyé pour édition/audit ne contient b, votes d’origine, gold ou
prédiction de l’extracteur. Le refus de l’éditeur et les contrôles échoués sont
journalisés, sans nouvelle formulation ni sélection d’une autre paire.

## Interprétation et arrêt

- Si aucune édition complète ne passe : résultat de faisabilité négatif de ce mécanisme
  automatisé ; ne pas relancer des prompts jusqu’à obtenir un cas acceptable.
- Si quelques cas passent : contrastes descriptifs très discrets (1–2 lectures), pas de
  test de signal, pas d’enrichissement FP/TP ni de gain causal revendiqué.
- Si les placebos/renommages expliquent la réaction : l’isolation du signal n’est pas établie.
- Budget, sortie invalide, troncature ou modèle inattendu : arrêt sans convertir l’erreur en zéro.

Le pilote ne permet pas une confirmation indépendante. Décision générale attendue :
INCONCLUSIVE sur l’utilité marginale, avec diagnostic explicite d’attrition/instabilité.
Une éventuelle étude R/J/T/F serait un nouveau protocole à budget égal et nouvelle
confirmation, pas un réglage sur les cas présents.
