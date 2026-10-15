# Pipeline décisionnel v1

Une décision classe la relation entre deux mentions d'événements. Le fournisseur de
mentions et le générateur de candidats réutilisent le loader historique. Le modèle
voit le document canonique et uniquement les deux mentions choisies : identifiants,
indices de tokens, offsets caractères, surfaces et phrases. Il ne reçoit aucune
relation de référence, démonstration annotée ou statistique des labels du test.

Le protocole v1 est `relation_gold_mentions`, explicitement oracle pour les mentions.
Ce n'est pas une mesure d'extraction d'événements ou de prédiction bout en bout.

## Installation et commandes

Depuis la racine, Python >=3.12 et uv :

```bash
uv sync --project benchmark --locked --extra hub
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli dry-run --config experiments/jev/esc_span_noul_smoke.yaml --output /tmp/span-plan.json
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli run --config experiments/jev/esc_span_noul_smoke.yaml --output /tmp/span-receipt.json
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli run --config experiments/jev/esc_mercury_noul_smoke.yaml --output /tmp/mercury-receipt.json
```

L'environnement doit fournir `OPENROUTER_API_KEY` (`OPENROUTER` accepté). Aucun secret
ne doit être copié dans le YAML. Le script enregistré `experiments/jev/run.sh` exécute
le YAML `active.yaml` sur le serveur de calcul configuré. Les commandes ci-dessus
sont l'interface du package ; les expériences suivies utilisent ce script à partir
d'un commit immuable, pas un processus manuel non enregistré.

Les fichiers `esc_noul_v1.yaml`, `esc_choice_v1.yaml`, `esc_context_v1.yaml` ciblent JEV
1.14 sur le développement ESC complet. **Ces configurations refusent les appels
payants**, avec budget zéro et `allow_paid: false`. Pour les tester sans API, ajouter
`--mock` à `run`. Une sortie mock ne mesure pas les performances du modèle.
Les fichiers `esc_mercury_{noul,choice,context}_smoke.yaml` permettent les trois
variantes avec Mercury gratuit. Span est limité à Noul ; Choice est refusé avant
inférence. Aucun modèle de substitution payant n'est sélectionné automatiquement.

Les modèles gratuits sont `respan/span-01-lite` et `inception/mercury-decide:free`.
Ils utilisent `POST /api/alpha/decisions`, sans JSON généré en texte libre. La version
réellement servie, les réponses structurées, l'usage et le coût déclaré sont conservés.
Les quotas gratuits peuvent interrompre un run ; ses décisions déjà réussies restent
en cache. Réexécuter la même configuration reprend ces décisions sans les repayer.
Un run interrompu n'est jamais présenté comme une évaluation complète.

## Trois variantes, un facteur à la fois

- **A Noul** : question causale demandée, probabilité positive, seuil `p >= 0.5`.
  Sortie `causal` ou `norel`, **aucune direction inventée**. La métrique principale
  est le F1 binaire historique ; les scores directionnels ne sont pas interprétables.
- **B Choice** : mêmes documents, mentions, candidats et état de base ; une question
  à trois classes `CAUSES`, `CAUSED_BY`, `NO_RELATION`, mappées exactement aux labels
  historiques. La classe structurée choisie est conservée avec ses probabilités.
- **C Context** : même question et seuil que A ; l'état ajoute le même document avec
  uniquement les tokens des deux événements marqués et leurs phrases explicites.

Comparer A/B/C avec **le même modèle**, le même split et la même liste de documents.
Le smoke prédéfini contient un document complet choisi pour sa petite taille, sans
consulter ses labels. Ses 156 paires ordonnées sont toutes conservées. Il ne représente
pas la distribution ESC complète. Le développement corrigé a 25 documents, donc le
pilot de 50 documents distincts n'est pas réalisable sur ce split. `max_examples: 50`
est seulement un plafond ; le dry-run indique le nombre réellement disponible.

## Réévaluer et analyser sans API

Remplacer `RUN` par le répertoire indiqué dans le reçu :

```bash
python3 -m benchmark reevaluate RUN --output /tmp/rescored.json
python3 -m pipelines.jev.cli rethreshold RUN --threshold 0.7 --output-dir /tmp/threshold-07
python3 -m pipelines.jev.cli analyze RUN --output-dir /tmp/analysis
python3 -m benchmark compare-runs RUN_SPAN RUN_MERCURY --metric binary --output /tmp/comparison.json --markdown /tmp/comparison.md
```

L'analyse produit la matrice de confusion, distributions conditionnelles des
probabilités, courbe précision-rappel, sensibilité à 21 seuils, listes FP/FN, audit des
positifs hors candidats, baseline toujours négative et baselines aléatoires contrôlées
(taux positif 0.5, seeds 13/42/2026). Chaque baseline garde ses prédictions et métriques.
Les seuils sont explorés hors ligne ; aucun seuil n'est choisi automatiquement. Toute
sélection doit rester sur le développement, puis être figée avant le test. Le bootstrap
rééchantillonne les documents complets. Un seul document donne un intervalle dégénéré ;
il ne permet pas de conclure sur la généralisation. La variance entre seeds nécessite
plusieurs runs ; une seed du framework ne force pas le déterminisme du fournisseur.

Les scores intra/inter du banc désignent les relations dans une même phrase ou entre
phrases du **même document**, pas des relations entre documents.

## Cache et remplacement du modèle

La clé inclut fournisseur, endpoint, ID, version demandée, mode mock/live, état,
questions et version servie attendue si spécifiée. Le seuil est hors clé : sa
modification réutilise les probabilités. SQLite ne conserve que les réponses validées.
Le journal des tentatives est écrit immédiatement, même lorsqu'un document échoue.
Le budget réserve une borne par tentative ; timeout, retries, RPM, limite d'appels et
taille de contexte sont configurables. Un contexte trop long est refusé sans troncature.
La limite de dépenses est par client/run, pas un budget global partagé entre campagnes.

Pour un autre modèle compatible Decisions, changer `models`, `client.model`,
`client.model_version`, et éventuellement endpoint/provider. Renseigner
`expected_served_version` pour refuser une autre version réellement servie. Le client
ne présume pas qu'un endpoint arbitraire est gratuit. Pour une API différente,
implémenter un nouveau `RelationPredictor` (voir `benchmark/interfaces.py`) et
l'enregistrer avec `@register`, puis le sélectionner dans le YAML. L'évaluateur ne
change pas. `benchmark/examples/jev_plugin.py` montre aussi l'intégration de
probabilités déjà enregistrées par un modèle externe.

## Historique et limite de validation

Les fonctions historiques sont inchangées et vérifiées contre `dev`. Les fichiers
originaux de `e77431d6` manquent encore : configuration exacte, rapport, documents et
traces. Les notes décrivent un ESC corrigé, 233 documents train, DeepSeek-v4-flash,
7 passes, prompt lean coref et F1 binaire rapporté 46.6. Un smoke dev avec un autre
protocole d'inférence n'est pas directement comparable. Le comparateur refuse les
suites différentes ; la migration historique reste **non validée** jusqu'au rejeu
avec les artefacts originaux. Le chiffre des notes n'est jamais traité comme une
nouvelle mesure.

Des préparations sans API sont fournies pour CTB et MAVEN-ERE :
`experiments/jev/causal_timebank_noul_mock.yaml` et `maven_ere_noul_mock.yaml`.
Elles reprennent les dépôts historiques non corrigés du `config.yaml`, explicitement
épinglés, sans les remplacer silencieusement par les dépôts corrigés des anciennes
notes Laya. CTB utilise son split train historique (pas de dev inventé) ; MAVEN utilise
dev. Les dry-runs sur trois documents vérifient le chargement et le contexte ; ils ne
mesurent aucune performance modèle et ne sont pas des baselines du benchmark corrigé.

Les clients utilisant le même cache SQLite partagent aussi les créneaux de débit,
par fournisseur et endpoint, entre processus du même hôte. Ainsi deux variantes
concurrentes à 20 RPM partagent 20 requêtes/minute au lieu d'en envoyer 40. Cette
attente est enregistrée séparément de la durée réseau de chaque tentative. Des
machines ou caches distincts n'ont pas de limite commune : les coordonner séparément.

L'analyse exporte aussi `pair-probabilities.csv` : une ligne par décision avec les
identifiants, surfaces, classe choisie, probabilité positive, confiance de classe,
scope et version servie. Les listes d'erreurs contiennent les offsets en tokens et
les phrases correspondantes, réservés à l'analyse évaluateur. Un diagnostic compare
les deux ordres A/B sans modifier leurs prédictions.

Pour une chaîne entièrement composable sur prédictions enregistrées, charger
`pipelines.jev.postprocessor` et sélectionner
`postprocessors: [{plugin: jev_noul_threshold, version: "1", params: {threshold: 0.3}}]`
après le pipeline monolithique `saved_predictions`. Ce composant peut promouvoir
une ancienne décision négative ou rejeter une positive à partir de `p(causal)` ;
un simple filtre sur la confiance de la classe choisie ne permet pas ces deux sens.
Choice garde sa classe structurée ; une projection binaire explicite est disponible
avec la commande `rethreshold`, sans prétendre reproduire sa règle multiclasse.
