# workere / ESC — Agent 4 : pilote de faisabilité V0

**État : outillage hors ligne créé et testé ; aucun appel modèle ni nouvelle mesure ESC.**

Ce kit transforme le rapport « Empreinte de dépendance documentaire avec placebos » (27 octobre 2026) en protocole de faisabilité à faible coût. Il **ne** relance pas le protocole final de l'agent (84 appels par paire *pour le seul bras F*), ne modifie pas la fusion gelée et ne revendique pas un gain F1.

## Pourquoi ce pilote et pas directement les 60 + 60 paires ?

Le protocole complet F emploie deux paquets de sept lectures par condition (original, ablation cible, trois placebos, permutation), soit `6 conditions × 2 paquets × 7 lectures = 84 appels Luna par paire`, indépendamment du coût du repérage et de l'audit des preuves. Sur 60 paires : 5 040 appels pour un bras, sans les bras de comparaison. Avant d'investir, nous devons montrer : (1) qu'on peut éditer le texte sans casser les mentions et l'identité ; (2) qu'il existe des placebos véritablement comparables ; (3) que l'empreinte n'est pas une réponse triviale aux perturbations.

Le kit propose un **smoke test à 12 positifs initiaux confiants** (`b >= 0.7`), choisis à l'aveugle sur des documents de développement, si disponibles et préalablement autorisés. Six conditions : original, cible, trois placebos, permutation d'IDs. La V0 demande `2` lectures de l'original et `1` lecture des cinq autres conditions : **7 appels documentaires par paire au maximum**, soit **84 appels maximum pour 12 paires**, hors génération/revue des éditions et retries. Les 1–2 lectures donnent une mesure grossière, PAS une réplication fidèle des sept passages de référence, PAS un test statistique, PAS un score de fusion nouveau.

Les ajouts de paires et l'amélioration de F1 **ne font pas partie** de cette V0. Un pilote de comparaison R/J/T/F de puissance suffisante est une étape ultérieure conditionnelle.

## Contenu

- `pilot.py` : outil CLI local **sans aucun accès réseau ni modèle**.
- `tests.py` : tests du protocole et du calcul.
- `examples/baseline_synthetic.jsonl` : un document inventé, avec offsets et scores fictifs.
- `examples/edits_synthetic.jsonl` : éditions fictives, pré-marquées comme approuvées **uniquement** pour vérifier le fonctionnement du code.
- `examples/gold_synthetic.jsonl` : vérité terrain inventée, utilisée séparément et **uniquement après** calcul/scellement des scores fictifs.
- `examples/build_fixture.py`, `examples/build_mock_predictions.py` : générateurs de faux jeux pour test logiciel.
- `out/` : manifeste, tâches, prédictions **MOCK**, sorties et audits du test synthétique. **Ne jamais citer ces nombres comme performance ECI.**
- `AGENT4_IMPLEMENTATION_PROMPT.md` : mission complète pour l'agent de développement qui possède le dépôt et les caches.

## Exécution sur le jeu synthétique (déjà testée)

Depuis ce dossier :

```bash
python3 -m unittest -v tests.py
python3 examples/build_fixture.py
python3 pilot.py select --baseline examples/baseline_synthetic.jsonl --out out/selection.json --n-positive 1
python3 pilot.py template --baseline examples/baseline_synthetic.jsonl --selection out/selection.json --out out/edits_template.jsonl
python3 pilot.py prepare --baseline examples/baseline_synthetic.jsonl --selection out/selection.json --edits examples/edits_synthetic.jsonl --out-tasks out/tasks.jsonl --out-audit out/audit.json
python3 examples/build_mock_predictions.py
python3 pilot.py score --tasks out/tasks.jsonl --predictions out/predictions_MOCK.jsonl --out out/scores_MOCK.jsonl
python3 pilot.py evaluate --scores out/scores_MOCK.jsonl --selection out/selection.json --gold examples/gold_synthetic.jsonl --out out/evaluation_MOCK.json
```

**IMPORTANT :** la cible est conçue pour faire passer un faux score de 1 à 0 tandis que les placebos maintiennent 1 ; le contraste = 1 dans cet exercice artificiel. **Ce résultat est généré, pas mesuré.**

## Formats d'entrée, à adapter aux caches réels

`baseline.jsonl` : **une ligne par document**, sans aucune information gold ni label, avec les `mentions` sous forme d'offsets Unicode `[start,end)` (indexation Python) et une liste unique de **paires non ordonnées**.

```json
{"doc_id":"d1","topic_id":"topicA","split":"dev","text":"...","mentions":[{"id":"e1","start":0,"end":3,"surface":"..."}],"pairs":[{"a":"e1","b":"e2","baseline_score":0.81}]}
```

Ceci est un **schéma illustratif** : les champs sont à remplir avec les vraies données ; l'exemple n'est pas valide tel quel (e2 absent). Le `split` doit être `dev` ou `pilot` par défaut. La conversion à partir du dépôt doit préserver les frontières de mentions, le même univers de paires et le score `b` d'origine. Elle ne doit pas lire le gold, ni accéder aux 88 documents contestés comme réserve pour une confirmation. Le script refuse les champs de label dans son entrée opérationnelle. Les vrais IDs et offsets doivent être vérifiés.

`edits.jsonl` : **une ligne par variante et paire sélectionnée**, préparée par un éditeur indépendant puis révisée à l'aveugle du gold et des réactions du modèle :

```json
{"doc_id":"d1","pair":["e1","e2"],"variant":"target","review_status":"approved","reviewed_blind_to_gold":true,"edits":[{"start":100,"end":107,"before":"...","after":"..."}]}
```

À fournir pour `target`, `placebo_1`, `placebo_2`, `placebo_3` et `rename_ids`. Les `start/end` pointent vers **le document original** et la surface `before` est obligatoire. Les éditions de longueur variable sont permises, avec recalcul des offsets ; aucune mention ne peut être touchée. Pour `rename_ids`, fournir `id_map` bijective couvrant **toutes** les mentions, sans toucher au texte.

**Le contrôle automatique n'est que structurel.** Le booléen `reviewed_blind_to_gold` est une déclaration d'une revue réellement effectuée ; ne pas le renseigner à `true` sans revue indépendante. Le programme ne peut pas certifier la conservation de la sémantique, l'innocuité des placebos, les preuves distribuées ou l'absence d'indices résiduels. Les variantes non révisées n'entrent pas dans `tasks.jsonl`.

`tasks.jsonl` : documents transformés + mentions remappées + paires à transmettre à l'extracteur Luna **avec exactement le prompt, les démonstrations hors topic et le parseur gelés dans le dépôt**. Le connecteur d'appels n'est pas inclus ; **aucun appel modèle n'a été lancé**.

`predictions.jsonl` : réponses du modèle **déjà parsées** par l'adaptateur d'origine, exactement une par `task_id` :

```json
{"task_id":"<id provenant de tasks.jsonl>","status":"ok","predicted_ordered_pairs":[["e1","e2"],["e2","e1"]],"input_tokens":1234,"output_tokens":200}
```

Les réponses non parseables / manquantes doivent être marquées comme telles (`status != ok`) et provoquent l'abstention, **jamais un vote négatif**. Une omission dans une réponse valide reste un zéro, selon la convention de Luna. Les deux orientations sont distinguées, puis agrégées en `s_L`, sans les interpréter comme observations indépendantes. Les tâches `rename_ids` utilisent les IDs remappés et doivent être remises dans l'espace original pour les décisions finales.

`score` produit les six scores par paire, les différences, le contraste placebo `median(s_placebos) - s_target`, l'écart de renommage et l'instabilité des lectures de l'original. Il écrit un verrou SHA-256. `evaluate` n'utilise le gold **qu'après** ce verrou, pour un diagnostic descriptif sur les paires sélectionnées. En V0, aucune règle ADD/REMOVE n'est appliquée ; par conséquent aucun F1 système n'est calculé.

## Ordre des opérations sur le vrai dépôt

1. **Préflight et garde-fou budgétaire** : vérifier split/IDs, présence des caches, version Luna, quotas et crédits. Auditer l'hypothèse de la réserve de 88 documents en vérifiant les journaux et pas seulement le train233.
2. **Sélection sans gold** : adapter un exporteur de cache vers `baseline.jsonl`, puis sélectionner au maximum 12 positifs (si plusieurs topics disponibles, favoriser la diversité et déclarer le nombre obtenu).
3. **Éditions révisées** : repérer supports et alternatives, écrire cible + trois placebos comparables, puis revue à l'aveugle. Les invalides restent comptés et jamais remplacés opportunément après observation des réactions.
4. **Budget** : calculer le nombre de tâches et le coût plafond réel. **Aucun appel payant** tant que les entrées, le contrôle de dépenses et la revue ne sont pas validés. V0 : au plus 84 appels de scoring si les 12 cas sont tous éligibles ; modèle, tokens et budget économique doivent être consignés.
5. **Connecteur gelé** : exécuter les tâches sur Luna, pas sur un autre modèle plus disponible, en préservant le prompt et les démonstrations. Intercaler les conditions pour réduire les effets temporels. Collecter les coûts.
6. **Sceller puis évaluer** : charger les sorties parsées, vérifier les échecs, calculer les contrastes sans labels, sceller, puis seulement consulter le gold. Classer le résultat : faisabilité, absence de spécificité, ou inconclusif. Il ne sera pas une confirmation de F1.

## Critères d'arrêt

- Plus de 20 % des paires sont invalides ou sans placebos appariables : revoir le mécanisme d'édition **avant** une campagne.
- Une part substantielle des placebos touche une preuve résiduelle ou modifie les participants/temps/lieux : ne pas lancer l'API.
- Les réponses sont fréquemment non parseables ou la permutation d'IDs provoque autant de réaction que l'ablation : signal non isolable, arrêt/diagnostic.
- Pas d'information nouvelle par rapport à la répétition ou à un juge de même coût, dans le pilote étendu : arrêter cette piste.

## Seuil d'amélioration du F1 pour une future règle de retrait

À la référence TEST-S40 (464 TP, 473 FP, 507 FN, F1≈48,64), une règle de **retrait** doit retirer **plus de 75,68 % de faux positifs** parmi toutes les prédictions qu'elle retire pour améliorer le F1. Un enrichissement de 10 points en FP par rapport à un contrôle n'est donc **pas suffisant** à lui seul. Le premier pilote peut montrer un mécanisme de dépendance ; seule une confirmation indépendante d'une règle gelée peut établir une utilité opérationnelle.

## Accès aux sources

Le rapport joint affirme qu'un audit local a trouvé les 88 IDs de réserve dans train233 ; les fichiers `audit_local.py` et `audit-local.json` évoqués n'étaient **pas** joints à la conversation. Cette affirmation est donc attribuée à l'agent et non revérifiée ici. Le dépôt `workere` et les caches ESC mentionnés n'étaient pas présents dans les pièces disponibles, ni retrouvés parmi les dépôts GitHub accessibles : pas de nouvelles prédictions réelles.
