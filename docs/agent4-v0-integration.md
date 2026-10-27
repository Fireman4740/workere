# Agent 4 — intégration prudente du pilote placebo V0

Date : 27 octobre 2026. **Décision : INCONCLUSIVE. Aucun appel modèle, coût modèle 0 USD, aucun contraste réel, aucun F1 nouveau.** Le préflight hors ligne est exécuté ; le pilote payant est bloqué (`BLOCKED_HUMAN_REVIEW`, `BLOCKED_BUDGET`, snapshots incomplets et disponibilité modèle non vérifiée). Ce n’est ni un échec scientifique du contraste, ni un GO.

## 1. Dépôt, kit et contrat gelé

- Dépôt : worktree de cette session, base `a253e5e507532fc257249f9446f7b36918a5dbdf` ; branche nouvelle `integration/agent4-v0-preflight`. Aucune expérience historique, configuration de référence ou commande de run n’est modifiée.
- Le chemin fourni `/Users/mathiscarlesso/Desktop/eci_agent4_next` est absent. La copie retrouvée est **`/Users/mathiscarlesso/Desktop/eci_agent4_pilote_v0`**, contenant les fichiers annoncés. `agent4_v0/kit/` conserve à l’identique `pilot.py`, README, tests et cinq fixtures/générateurs ; huit fichiers contrôlés par SHA-256. Aucun fichier `out/*MOCK*` n’est importé comme résultat réel.
- Référence historique inspectée : worktree `chat_6c8b45c4-a7f7-4139-80bb-f8cfdbc57a83`, branche J2, commit `ccdaf3a…` (identité complète dans l’artefact `reference-contract.json`). Le plugin réellement employé par les caches est `pipelines/jev/llm.py::LunaDocumentPredictor` et son transport `pipelines/jev/batch.py::BatchChatClient`, et non un nouvel appel au runner générique.
- `eci/voting.py`, `eci/runner.py`, `eci/tools.py`, `eci/parse.py` et `prompts/esl_standard_eci_lean_coref.yaml` sont inchangés et ont les mêmes hashes que dans ce worktree historique. Le prompt SHA-256 est `88a87ca71ac011935ca572af05ed6531545460ebae398ee87c06cd31e32a827b`.
- Contrat : document entier, une démonstration gold d’un autre topic, `openai/gpt-6-luna:batch`, version logique `gpt-6-luna-20260922`, raisonnement `high`, sortie maximale 32 768 tokens, parseur `eci.parse.parse_triples`. Aucun remplacement de modèle ni synthèse de nouvelles démonstrations.
- La fusion binaire non orientée à 0,39 et les comptes TEST-S40 fournis restent une **référence acquise**, pas un résultat produit ici.

Le code ajouté se limite à l’export des formats, la préparation/revue des entrées, l’interface de transport gelé et les contrôles de faisabilité. Aucun entraînement, calibrateur, variation de seuil ou recherche bibliographique nouvelle.

## 2. Caches, univers des candidats et offsets

Sources accessibles :

1. `experiments/jev/round4/controls.json` du worktree historique : 21 832 lignes ordonnées sur DEV44, champs `doc,a,b,luna,d1` utilisés. Le champ gold présent dans cette source mixte n’est jamais consulté pour la sélection ou les interventions.
2. `experiments/jev/round4/suites/<doc>.json` : tokens, mentions, spans, phrases, texte balisé et paires candidates. Le convertisseur emploie une liste blanche de champs et n’importe pas les annotations ni les chaînes gold dans son export.
3. Manifestes `document-selection.json`, `round3/round3-selection.json`, `round4/selection.json` et registre RES30 : exclusions explicites.
4. Archive locale `jev-v1/esc-improvement-plan/luna-dev25/source-evaluations.tar.gz` : snapshots des requêtes dans `*/decisions.jsonl`, avec pass index, démo et requête exacte. Les anciennes réponses ne sont pas réutilisées comme répétitions V0.

**Frontière sans gold :** les JSON historiques contiennent des champs gold et sont désérialisés par le lecteur JSON ; seuls les champs autorisés sont copiés ou utilisés. Aucun évaluateur historique ni `Scorer` n’est importé. Un test remplace les valeurs gold synthétiques par un objet inutilisable : baseline et sélection restent identiques octet pour octet. Ce n’est pas une preuve d’absence de toute exposition historique des développeurs aux labels ; c’est le contrôle de cette procédure de sélection.

Pour chaque document :

- reconstruction du même texte `<ID surface>` à partir des tokens/spans, comparaison exacte à la suite gelée ;
- vérification des deux sens de **chaque** candidat et refus d’un sens inverse absent (pas de copie silencieuse) ;
- export d’une paire canonique non orientée avec `b=(vote_ab+vote_ba+d1_ab+d1_ba)/4` ;
- offsets Unicode Python `[start,end)` sur le texte brut à espaces simples, plus table des offsets tokens et plages de phrases ;
- refus explicite des mentions discontinues/chevauchantes que le kit caractère ne sait pas représenter. Aucun document de l’export présent n’a été rejeté pour cette raison.

L’adaptateur rebâtit le balisage directement depuis les offsets transformés ; il ne retokenise pas les événements. Il vérifie toutes les surfaces, la bijection des IDs, la paire exposée et l’univers complet des candidats. L’original doit reproduire la requête historique, les conditions éditées n’en changent que le texte final. Toute édition réelle devrait également faire contrôler les frontières de phrases, la portée, les participants et les anaphores par l’auditeur : le contrôle automatique ne certifie pas ces propriétés sémantiques.

## 3. Cohorte scellée et exclusions

**DEV44 n’est pas accepté en bloc.** Il contient des documents déjà employés en CONF40 ou TEST-S40. Tous ces documents, les 118 documents du `holdout_rest` et les réserves RES30/88 sont exclus, même si un autre plan les a ensuite appelés « développement ».

Résultat de l’export : **10 documents de développement déjà exposés**, 3 758 paires non ordonnées, dont **126 à b ≥ 0,7**. Le sélecteur du kit, hash fixé `agent4-v0-20261027`, choisit **12 paires sur les 10 documents et 8 topics**, au plus deux par document. Pas de sélection à partir des 138 FP connus.

| Topic | Paires sélectionnées |
|---|---:|
| 1 | 1 |
| 14 | 1 |
| 20 | 1 |
| 22 | 1 |
| 30 | 1 |
| 32 | 1 |
| 37 | 4, réparties sur deux documents |
| 41 | 2, réparties sur deux documents |

Hashes scellés :

- `baseline.jsonl` : `05bd71e0318c5b686cbab69ac2030b6f7ed4a49aa67fc82ccb4d9b74d29b5865`.
- `selection.json` : `d566bb765a41a1251ca95f67b632ba78b214c1d13f4d23e2300f066854b87503`.

Les quotas par document, exclusions intégrales, source SHA et identité du dataset sont dans `split-manifest.json`.

## 4. Réserve : exposition H1 désormais vérifiée

Les fichiers du premier agent `audit_local.py` / `audit-local.json` sont accessibles dans `empreinte-placebo-2026-10-27/`. L’audit initial ne prouvait que l’inclusion dans train233.

Cette phase croise **le manifeste H1 `experiments/jev/haiku/data/esc-suite.json` et le journal complet du run H1 `a7f7bbd5-b115-4926-850a-af6eb7f44776`** : les 88 IDs sont dans le manifeste **et chacun a une entrée documentaire H1 `status=completed` dans le journal**. L’exposition H1 n’est donc plus seulement déduite d’un nombre agrégé de documents. L’audit sauvegarde les listes et hashes dans `reserve-audit.json`.

Conclusion : **les 88 ne sont pas intacts**. La chronologie détaillée des accès aux labels et des décisions adaptatives n’a pas été reconstruite ; aucune indépendance confirmatoire n’est revendiquée. Aucun de ces documents n’entre dans la cohorte V0.

## 5. Éditions et validation indépendante : porte fermée

Sorties produites, sans appels d’édition à un modèle :

- 12 fiches `evidence-proposals.jsonl` : citations littérales autour des deux mentions, offsets et contextes d’occurrences lexicales alternatives. Une répétition lexicale n’est pas déclarée coréférence ; ces contextes ne sont pas certifiés comme supports causaux exhaustifs.
- 60 fiches `edit-proposals.jsonl` : cible, trois placebos, permutation d’IDs pour chacune des 12 paires.
- 12 paquets de revue `human-review-packets.jsonl`, sans labels ni scores de modèle.
- `edit-audit.json` : **0/12 éligibles, 12/12 en attente d’approbation**, donc `tasks.jsonl` vide.

La génération locale de spans n’est qu’une heuristique de connecteurs non protégés : **2 cibles et 2 placebos non vides sur 48 variantes sémantiques** ; 44 variantes n’ont pas de span proposé. Aucune paire ne dispose d’un jeu complet cible + trois placebos. Les permutations couvrent toutes les mentions avec les IDs numériques existants, sans renommer les entités.

**Ne pas interpréter ces 44 absences comme 44 éditions sémantiquement invalides.** Elles indiquent l’insuffisance de la proposition automatique conservatrice. Aucune revue humaine indépendante n’a été réalisée dans cette session ; aucun champ `approved` n’a été attribué à une édition réelle. Les supports distribués/redondants, l’innocuité et l’appariement en position restent à auditer. Il est donc impossible de conclure STOP sur un taux d’invalidité humaine, comme de conclure GO.

`agent4_v0.review` exige un reçu externe liant les hashes des propositions, une identité humaine différente du proposeur et des attestations d’indépendance/aveuglement aux labels **et** aux sorties. Le reçu est une attestation, pas une preuve cryptographique de compétence sémantique. Les fixtures synthétiques pré-approuvées du kit ne satisfont pas cette porte réelle.

## 6. Démonstrations, disponibilité et coût

- Snapshots complets des sept passes retrouvés pour **4 documents sélectionnés** : `37_1ecbplus.xml`, `37_9ecbplus.xml`, `41_4ecbplus.xml`, `41_5ecbplus.xml` — soit 6 paires sur 12.
- Snapshots absents des sources inspectées pour les six autres documents sélectionnés. Ils ne sont pas reconstruits silencieusement à partir de nouveaux tirages ou de nouveaux gold.
- **28 requêtes originales** ont été reconstruites hors ligne et comparées à leurs snapshots, avec égalité complète des objets requête ; les chaînes et paramètres sont inchangés. C’est une mesure de parité de format, pas 28 appels modèle ni 28 nouvelles prédictions.
- V0 prévu : une démonstration fixée, celle de la passe 0, identique dans les six conditions et dans les deux répétitions originales. Ainsi la comparaison n’est pas confondue par un changement de démo. Cette V0 **n’est pas** la référence à sept lectures par condition.
- V0 maximal : 2 lectures original + 1 pour cible, chacun des trois placebos et renommage = **7 appels par paire, 84 appels maximum pour 12 paires**, sans reprise payante automatique. Borne de sortie totale : 2 752 512 tokens, pas une consommation mesurée. Les tokens d’entrée incluent document, démo et prompt.

**Aucun plafond monétaire effectif n’a été autorisé.** Les prix historiques et les budgets d’autres expériences ne sont pas une enveloppe réutilisable. Le fichier `budget-template-UNAPPROVED.json` laisse montant, tarifs supérieurs, frais, source/date et registre à renseigner, avec `approved=false`. Le plafond réel en USD est donc **non déterminé**, et non zéro ; la dépense réellement engagée par cette phase est zéro.

Le code calcule, si ces entrées sont autorisées :

`Cmax = somme[(borne_entree_utf8_et_framing * prix_entree_max + 32768 * prix_sortie_max) / 1e6 * facteur_frais_max]`.

C’est une borne conditionnelle à la validité du contrat tarifaire et du plafond de sortie, pas un tarif inventé. Il faut vérifier que les prix couvrent raisonnement et frais éventuels. Le registre SQLite persistant, commun aux reprises du même scope, réserve atomiquement le pire coût avant chaque appel, conserve les réservations incertaines, interdit le double envoi d’une même tâche et borne globalement le scope à 84 tentatives. Chaque tâche a un cache/receipt de transport distinct pour ne pas dédupliquer les deux répétitions originales. Aucun suffixe de prompt n’est ajouté pour forcer une répétition.

Un identifiant d’authentification alternatif est présent dans l’environnement, mais cela ne démontre ni quota, ni crédits, ni disponibilité actuelle de Luna. Aucun appel de santé potentiellement facturable n’est effectué. `--execute` demeure refusé tant que les portes échouent ; aucun repli vers un modèle plus accessible.

## 7. Interface, journaux et scoring

- `python -m agent4_v0.adapter …` est **dry-run par défaut** ; `--dry-run` est aussi explicite. Aucun transport n’est importé en ce mode.
- `--execute` requiert revue liée aux hashes, budget autorisé avec prix sourcés, registre persistant, snapshots, sources du transport épinglées et credential. Le client historique batch est importé seulement après ces portes. Pas de transport synchrone de secours, pas de réutilisation d’anciennes réponses comme nouvelles lectures.
- Requêtes ordonnées par shuffle hash fixé ; démo non adaptée aux éditions. Journalisation de la requête, ID démo, réponse brute, modèle servi, parse, usage, coût/réservation et erreurs ; tokens et frais repris des reçus, frais inconnus non assimilés à zéro. Les échecs gardent leur réservation et ne sont jamais transformés en votes négatifs.
- Le parseur gelé s’applique, avec le dernier label par paire et filtrage à l’univers fourni. Une omission valide est zéro ; absence de JSON, troncature ou modèle inattendu est manquante/invalide.
- `agent4_v0.score` enveloppe le score du kit, ajoute les votes orientés et un gel des fichiers avant tout gold. Il refuse un jeu vide ou marqué MOCK. **Ce scoring n’a pas été exécuté sur de vraies réponses**, puisqu’il n’y en a pas.

Aucun `predictions.jsonl`, `scores.jsonl`, contraste réel ou F1 nouveau n’est fabriqué dans les artefacts V0. La sortie fictive contraste=1 du kit reste un test logiciel explicitement MOCK.

## 8. Reproduction

Depuis la racine du dépôt, ne pas réutiliser un dossier scellé comme sortie :

```bash
REF=/Users/mathiscarlesso/.local/share/openresearch/worktrees/75e891b6-a1c1-41a2-a06f-99092098f31e/chat_6c8b45c4-a7f7-4139-80bb-f8cfdbc57a83
ART=/Users/mathiscarlesso/.local/share/openresearch/files/workere/jev-v1
OUT=/tmp/agent4-v0-reproduction
uv run --locked --no-sync python -m unittest tests.test_agent4_v0 -v
uv run --locked --no-sync python -m unittest discover -s agent4_v0/kit -p tests.py -v
uv run --locked --no-sync python -m agent4_v0.convert_cache \
  --reference-root "$REF" \
  --reserve-register "$ART/new-signal-judges/res30/reserve-register.json" --out "$OUT"
uv run --isolated --no-project --with pyyaml python -m agent4_v0.preflight \
  --out "$OUT" --reference-root "$REF" \
  --archive "$ART/esc-improvement-plan/luna-dev25/source-evaluations.tar.gz" \
  --prompt prompts/esl_standard_eci_lean_coref.yaml \
  --reserve-register "$ART/new-signal-judges/res30/reserve-register.json" \
  --h1-manifest /Users/mathiscarlesso/.local/share/openresearch/worktrees/75e891b6-a1c1-41a2-a06f-99092098f31e/chat_dde16780-d6aa-4731-afeb-737127fd99e3/experiments/jev/haiku/data/esc-suite.json \
  --h1-log /Users/mathiscarlesso/.local/share/openresearch/run-logs/a7f7bbd5-b115-4926-850a-af6eb7f44776.log
uv run --locked --no-sync python -m agent4_v0.adapter \
  --tasks "$OUT/tasks.jsonl" --baseline "$OUT/baseline.jsonl" \
  --selection "$OUT/selection.json" --snapshots "$OUT/frozen-snapshots.jsonl" \
  --out "$OUT/adapter-dry-run"
```

Les tests propres à cette intégration sont stdlib ; `--no-sync` évite d’installer les dépendances lourdes du pipeline inchangé. Le petit outil de lecture YAML utilise l’environnement isolé documenté. Aucun manifeste de dépendances ni lock n’est modifié.

Après réception d’une **vraie revue externe** et de spans complets :

```bash
python -m agent4_v0.review --baseline "$OUT/baseline.jsonl" \
  --selection "$OUT/selection.json" --proposals "$OUT/edit-proposals.jsonl" \
  --receipt /chemin/recu-humain.json --out /nouveau/dossier/revu
# Adapter en dry-run avec les tâches/certificat du dossier revu et une enveloppe autorisée.
# --execute reste une action explicite après inspection de tous les blockers et du coût plafond.
# Après de vraies sorties uniquement :
python -m agent4_v0.score --tasks /nouveau/dossier/revu/tasks.jsonl \
  --predictions /nouveau/dossier/appels/predictions.jsonl \
  --baseline "$OUT/baseline.jsonl" --selection "$OUT/selection.json" \
  --certificate /nouveau/dossier/revu/review-certification.json --out /nouveau/dossier/scores
```

Ne pas lancer une commande comportant des chemins illustratifs sans les remplacer. Le transport payant complet n’a pas été testé contre le service ; les tests couvrent son interface et les portes hors ligne, pas la facturation réelle.

## 9. Tests, essais et limites de livraison

- Tests du kit inchangé : 8.
- Tests d’intégration : 20, incluant la persistance du plafond entre instances et la limite globale de 84 tentatives ; tous réussis.
- Smoke synthétique du kit : sélection/préparation/fausses prédictions/scoring et verrou, dans un répertoire temporaire, sans API. Jamais une mesure ESC.
- Contrôles de parité : 28 requêtes historiques originales, mention/offset/candidats, aucune modification du cœur gelé.
- Non exécuté : suite complète des pipelines dépendant de services/datasets/environnements lourds, appel live, revue humaine, validation sémantique, nouveaux scores, jointure gold, décision ADD/REMOVE et F1.

Le dossier initial `v0/` est conservé comme préflight de développement du convertisseur. `v0-final/` ajoute le relevé explicite de contextes alternatifs et le manifeste de snapshots : **baseline et sélection sont identiques octet pour octet entre les deux**, sans nouvel essai de seuil, de prompt ou de modèle. Les changements de code et le défaut de chemin initial du kit sont consignés ; les journaux et hashes finaux décrivent le code livré.

## 10. Décision et conditions de reprise

**INCONCLUSIVE / BLOCKED**, pas GO : effectif modèle nul, aucune paire humainement éligible, jeux de placebos incomplets, six documents sans snapshots et budget non autorisé. La pauvreté des propositions automatiques incite à une revue de faisabilité de l’édition **avant de financer des appels** ; elle ne permet pas de trancher l’hypothèse du contraste.

Prochaines portes, dans cet ordre : compléter/auditer humainement les supports et les trois placebos ; importer le reçu aveugle ; retrouver les snapshots manquants sans remplacer opportunément les paires ; fixer une enveloppe USD explicite avec tarifs/frais vérifiés ; inspecter le dry-run puis éventuellement exécuter. Ne pas étendre aux bras R/J/T/F avant une décision V0 informée. Aucun plan R/J/T/F n’est lancé ni présenté comme validé.

Pour une future règle seulement : ajouts utiles si `t/n > 24,3187 %` à la référence fournie ; retraits utiles si plus de `75,6813 %` des retraits sont des FP. Un simple enrichissement de +10 points en FP n’implique pas un gain de F1. Ces règles algébriques ne sont pas des gains de ce pilote.
