# Pilote de transport local de preuve ECI

Sidecar indépendant : aucun import du runner de baseline, aucun entraînement, aucune modification de `eci/`, de la fusion ni du seuil **0,39**. Commande d’entrée :

```bash
uv run --locked python -m proof_transport --help
```

**Ne pas lancer `experiments/jev/run.sh` pour ce pilote : c’est encore le lanceur hérité de la baseline.** Aucun appel fournisseur n’a lieu dans les commandes de préparation, de requêtes, de replay ou de smoke. La commande `execute` est la seule entrée réseau d’inférence ; elle exige une autorisation externe explicite. Les tests ne contactent pas de modèle.

## Contrat et variantes

- **A0** : identité exacte des occurrences, sans certificat.
- **A1** : compatibilité factuelle, ≥2 accords observés discriminants et portée connue compatible, sans preuve positive de liaison. UNKNOWN n’est jamais un accord.
- **A2** : source vérifiée, E immuable, B positif couvrant la mention alternative, rôles/incident/temps/lieu/granularité, recherche de concurrents sur toutes les mentions et phrases, portée préservée, abstention sur manque ou ambiguïté. Un contrôle mécanique ne prouve pas une implication sémantique : l’audit humain est indispensable.
- **A3 contrôlé** : groupes T2 réellement sauvegardés, mêmes sources vérifiées et propositions que les autres bras. Une substitution ne passe que si cause et alternative partagent un groupe et si l’effet est dans un autre groupe. T2 laisse les scores intra-groupe inchangés : il ne les supprime pas, mais ne propage pas non plus leur maximum. Aucun groupe nouveau ni fermeture récursive.

Pour tous : cause seule remplacée, ≤2 propositions par source, ≤1 ajout par source, ≤5/document, déduplication non orientée. Les positifs initiaux restent inchangés. Les ajouts sont génération 1 et non réutilisables. Le classement commun est score source décroissant puis hash de candidat ; aucun label n’y intervient.

`native-t2` reproduit séparément le maximum historique entre groupes sur tous les scores initiaux. Ce diagnostic non plafonné n’est pas A3 contrôlé et ne compte pas comme un cinquième bras réglable. Un cache absent ou ambigu est **UNAVAILABLE**, pas un groupe vide ou une prédiction négative supposée.

## Données et deux gels

Le rapport de conception nécessite une orientation source et un proposeur commun avant de connaître les véritables alternatives. Il serait incorrect de fabriquer dès maintenant 240 paires de transport.

1. `prepare-local` gèle les données **sans gold**, les prompts, les règles, les empreintes du code/lockfile et au plus 120 racines. Le fichier `candidate-slots.pending.jsonl` décrit seulement une **capacité maximale** ; ce n’est pas un registre de candidats évaluables.
2. Après extraction et vérification/récupération communes, `freeze` gèle les **vrais candidats**, au plus 240, avec un hash de certificat source par proposition. Ce gel précède tous les jugements de substitution et toute jointure au gold.

Les champs d’inférence sont construits par liste blanche depuis les exports mixtes historiques. Les IDs restent des chaînes, les deux ordres des scores sont obligatoires, les candidats couvrent exactement toutes les combinaisons ESC. Les surfaces, premiers tokens et index de phrase doivent coïncider avec les suites historiques. Les champs `gold`, `annots`, `relations`, `coref_clusters` et `mention_type` ne passent pas l’adaptateur.

Les citations emploient des offsets Unicode semi-ouverts dans **`" ".join(tokens)`**, pas dans `text` balisé ni dans le XML d’origine. Les mentions discontinues conservent leurs segments. Toute citation déclarée est vérifiée exactement ; aucun appariement flou ou réparation silencieuse.

## Reproduction locale sans paiement

Depuis le clone qui contient les branches historiques :

```bash
ART=/chemin/vers/les/artefacts/workere
OUT="$ART/transport-preuve-locale/pilote"

uv run --locked python -m unittest tests.test_proof_transport -v
uv run --locked python -m proof_transport audit-test-s40 \
  --pairs "$ART/jev-v1/round3/error-analysis/pairs.csv" \
  --raw-rows "$ART/jev-v1/round3/error-analysis/test-s40-gold-rows.jsonl" \
  --output "$OUT/test-s40-audit.json"

uv run --locked python -m proof_transport prepare-local \
  --repo "$PWD" --artifacts "$ART" --output "$OUT/dev44-plan"
uv run --locked python -m proof_transport requests \
  --plan "$OUT/dev44-plan" --stage extract \
  --output "$OUT/extract.requests.jsonl"
uv run --locked python -m proof_transport native-t2 \
  --plan "$OUT/dev44-plan" --output "$OUT/t2-native.label-free.json"
uv run --locked python -m proof_transport smoke --output "$OUT/smoke"
```

L’audit TEST-S40 lit explicitement le gold pour **reproduire une mesure**, jamais pour choisir les sources ou les candidats. DEV44 est un ensemble de développement consommé ; les anciens 88 documents ne sont **pas** une réserve vierge. Le smoke est synthétique, sans labels et avec faux reçus marqués `synthetic_fixture`. Il ne mesure aucune performance ECI. Ces reçus sont refusés sur des documents réels.

Tous les artefacts scientifiques sont write-once. Un changement de code, prompts, modèle, seuil, règles, documents ou score invalide le plan. Refaire une nouvelle préparation plutôt qu’écraser un gel ; aucune adaptation aux résultats gold n’est permise. Les tests peuvent réécrire leurs seuls fichiers temporaires pour vérifier cette protection.

## Tarifs et autorisation — aucune autorisation implicite

Une lecture du catalogue public n’est pas un appel d’inférence :

```bash
curl --fail https://openrouter.ai/api/v1/models > /tmp/transport-model-catalog.json
uv run --locked python -m proof_transport pricing \
  --catalog /tmp/transport-model-catalog.json --output "$OUT/pricing.json"
```

Le modèle proposé est **`openai/gpt-6-luna`**, révision attendue `gpt-6-luna-20260922`, raisonnement `high`, sans température imposée et sans repli vers un autre modèle. Il s’agit du transport synchrone, pas de l’ancien tarif batch. Le snapshot archive le slug canonique, les prix et les paramètres supportés. L’exécution refuse un catalogue daté de plus de 24 h ou une révision canonique différente. Les identifiants réellement servis et l’éventuelle absence de confirmation de révision dans la réponse sont archivés séparément ; le code ne les invente pas.

Budget global maximal : **10 USD et 1 500 000 tokens**, premiers plafond/réservation non admissible = arrêt, pas de variante partielle annoncée gagnante. Une seule base SQLite doit être partagée entre tous les stades : son chemin absolu est lié à l’autorisation. Réservations durables avant envoi ; coût fournisseur et tokens réels réglés ensuite ; tarif conservateur explicite seulement si le fournisseur ne rapporte pas de coût. Un appel de statut inconnu n’est jamais repayé automatiquement. Aucune relance cachée. Une erreur de transport conserve sa réservation ; une erreur de JSON/longueur conserve aussi son coût payé.

L’opérateur doit obtenir l’accord explicite de l’utilisateur **avant** de créer une autorisation réelle. Exemple **non autorisant**, à ne pas traiter comme une permission :

```json
{
  "approval": "NOT_AUTHORIZED",
  "approved_by": "",
  "plan_hash": "empreinte du plan.json",
  "config_hash": "empreinte du config.json",
  "pricing_hash": "empreinte du snapshot de prix",
  "model": "openai/gpt-6-luna",
  "max_usd": 10.0,
  "max_billed_tokens": 1500000,
  "expires_utc": "date UTC explicite future",
  "ledger_path": "/chemin/absolu/unique/budget.sqlite"
}
```

Ne jamais inscrire la clé API dans ce fichier, dans le code ou dans un artefact. Elle est lue uniquement depuis `OPENROUTER_API_KEY` au moment de l’envoi. Les tests injectent un transport factice, sans envoi et sans autorisation utilisateur persistante.

## Chaîne expérimentale après autorisation

Exécuter les requêtes groupées d’extraction, puis produire le stade commun de vérification et récupération :

```bash
# Seulement après accord explicite et création de l’autorisation liée au plan.
uv run --locked python -m proof_transport execute \
  --plan "$OUT/dev44-plan" --requests "$OUT/extract.requests.jsonl" \
  --pricing "$OUT/pricing.json" --approval /chemin/prive/approval.json \
  --ledger /chemin/absolu/unique/budget.sqlite --output "$OUT/extract.receipts.jsonl"

uv run --locked python -m proof_transport requests \
  --plan "$OUT/dev44-plan" --stage propose \
  --extraction-receipts "$OUT/extract.receipts.jsonl" --output "$OUT/propose.requests.jsonl"
# execute des requêtes propose avec LE MÊME ledger et la même autorisation.

uv run --locked python -m proof_transport freeze \
  --plan "$OUT/dev44-plan" --extraction-receipts "$OUT/extract.receipts.jsonl" \
  --proposal-receipts "$OUT/propose.receipts.jsonl" --output "$OUT/dev44-registry"
uv run --locked python -m proof_transport requests \
  --plan "$OUT/dev44-plan" --stage judge --registry "$OUT/dev44-registry/registry.json" \
  --output "$OUT/judge.requests.jsonl"
# execute des requêtes judge avec LE MÊME ledger et la même autorisation.

uv run --locked python -m proof_transport replay \
  --plan "$OUT/dev44-plan" --registry-dir "$OUT/dev44-registry" \
  --extraction-receipts "$OUT/extract.receipts.jsonl" \
  --proposal-receipts "$OUT/propose.receipts.jsonl" --judge-receipts "$OUT/judge.receipts.jsonl" \
  --output "$OUT/dev44-run"
```

Le programme refuse les reçus manquants/dupliqués, les IDs inattendus et les artefacts altérés. Il archive requêtes exactes, reçus et réponses fournisseur brutes, prompts, modèle annoncé/servi, justifications, coût, tokens et temps par appel. Plus de 5 % d’items logiques avec citations invalides bloque le passage au stade suivant ou le scoring ; le seuil est technique, fixé avant toute mesure. Une source refusée ne supprime pas son positif de baseline.

## Diagnostic de dépendance au témoin (≤20 requêtes, même budget)

Avant la jointure au gold, un échantillon de candidats trié par hash est utilisé
pour retirer E du certificat en gardant le document et les mentions. Le modèle
ne devrait plus déclarer l’obligation `causal_reuse` soutenue. Le contrôle porte
sur ses réponses brutes, pas sur le refus structurel tautologique d’un E vide.
Ce sont des diagnostics, jamais de nouvelles prédictions ni un cinquième bras.
Aucun échange de rattachement vers un rival n’est effectué automatiquement : ce
contre-test plus sémantique reste à l’audit humain.

```bash
uv run --locked python -m proof_transport requests \
  --plan "$OUT/dev44-plan" --stage diagnostic --registry "$OUT/dev44-registry/registry.json" \
  --output "$OUT/diagnostic.requests.jsonl"
# execute avec LE MÊME ledger et le budget global déjà autorisé, sans recharge.
uv run --locked python -m proof_transport inspect-diagnostics \
  --plan "$OUT/dev44-plan" --registry "$OUT/dev44-registry/registry.json" \
  --receipts "$OUT/diagnostic.receipts.jsonl" --output "$OUT/diagnostic"
```

Les coûts de ces contrôles sont inclus dans le total campagne si les reçus sont
fournis à `evaluate --diagnostic-receipts`. Maximum total prévu : 220 requêtes
groupées des stades principaux +20 diagnostics, toujours sous les mêmes plafonds.

## Audit aveugle et évaluation séparée

Exporter **tous** les candidats, sans labels, score initial, variante ni acceptation. Après les jugements, inclure le véritable transport proposé pour vérifier E/B, les concurrents et la portée :

```bash
uv run --locked python -m proof_transport blind-audit \
  --plan "$OUT/dev44-plan" --registry "$OUT/dev44-registry/registry.json" \
  --judge-receipts "$OUT/judge.receipts.jsonl" --output "$OUT/blind-audit"
```

Le fichier de correspondance d’IDs reste dans l’enclave d’évaluation. Chaque annotateur renvoie `audit_id`, `source_valid`, `binding_valid`, `scope_valid`, `proof_valid`, avec `YES|NO|UNKNOWN`, indépendamment du benchmark. Prévoir une double lecture d’un sous-échantillon et une adjudication aveugle des désaccords ; conserver les avis initiaux. Sans cet audit, les erreurs de portée acceptées et la validité des preuves restent **non mesurées**, pas égales à zéro.

Les commandes suivantes exigent un run complet scellé avant la première exportation gold :

```bash
uv run --locked python -m proof_transport export-evaluation-gold \
  --plan "$OUT/dev44-plan" --run-dir "$OUT/dev44-run" \
  --arrow /chemin/cache-train.arrow /chemin/cache-dev.arrow \
  --output "$OUT/evaluation-private/dev44-gold.json"
uv run --locked python -m proof_transport evaluate \
  --plan "$OUT/dev44-plan" --registry-dir "$OUT/dev44-registry" --run-dir "$OUT/dev44-run" \
  --gold "$OUT/evaluation-private/dev44-gold.json" \
  --human-audit /chemin/annotations-audit.jsonl \
  --audit-id-map "$OUT/blind-audit/private-audit-id-map.json" \
  --diagnostic-receipts "$OUT/diagnostic.receipts.jsonl" \
  --output "$OUT/evaluation"
```

Sorties : comparaison JSON/CSV, TP/FP ajoutés, F1 exact avant/après, neutralité marginale propre au jeu, U, zéro-vote récupérés et dénominateurs, erreurs de source au gold, diagnostic d’identité selon les chaînes gold, erreurs de portée et validité selon l’audit humain, raisons d’abstention, coûts, effets par topic et six comparaisons appariées avec bootstrap par topic. A3 est mesuré uniquement sur les documents réellement couverts ; les comparaisons qui l’impliquent utilisent l’intersection. Ne pas comparer son F1 de sous-ensemble à celui des 44 documents.

Les coûts des stades communs sont attribués à chaque bras pour sa rentabilité mais ne sont pas resommés pour le total campagne. Le coût/TP vaut non disponible si aucun TP. Les sources fausses au gold et les identités selon chaînes gold sont des diagnostics de benchmark, pas une preuve d’invalidité textuelle.

## Décision

Les règles de poursuite sont fixées avant labels : volume cible ≥50 ajouts et ≥20 TP pour A2, précision exploratoire ≥35 % et utilité >0, avantage F1 sur les témoins et intervalles appariés favorables, audit humain complété, budgets respectés, révision servie vérifiée et absence de revendication de réutilisation lorsque E est retiré dans le diagnostic pré-déclaré. Elles sont volontairement conservatrices ; un manque de puissance donne **indéterminé**, pas une preuve d’équivalence. Les tableaux de développement ne constituent **jamais** un gain confirmé. Une confirmation exigerait un jeu réellement indépendant.

Tant que les premières réponses autorisées n’existent pas, le statut correct est **préparé, en attente d’autorisation ; efficacité non mesurée**. Ni un smoke synthétique ni les anciens résultats T2/J1 ne permettent de remplir une table de gains A0–A3 sur ce nouveau registre.
