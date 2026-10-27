# M0 — planification uniquement

Ce paquet ne réalise **aucun appel LLM** et n'utilise aucun label pour planifier.

## Utilisation

```bash
python m0.py synthetic.jsonl --out example_plan --seed 42 --passes 7
python m0.py real_documents.jsonl --out real_plan --seed 42 --passes 7
```

Schéma JSONL d'entrée (une ligne par document) :

```json
{"doc_id":"d1","topic":"t1","mentions":[{"id":"e1","sentence":0},{"id":"e2","sentence":1}],"pairs":[["e1","e2"]]}
```

Fichiers produits : `plan.jsonl` (28 entrées par document : 4 bras × 7 passages) et `summary.json` (tailles des blocs). Les identifiants `DEMO_SLOT_1` à `DEMO_SLOT_7` sont des emplacements, **pas** des démonstrations réelles. Tous les passages ont `NOT_SENT`.

La transformation de permutation est fournie comme mapping structuré ; il reste à l'appliquer à **tous** les champs du prompt, y compris à la démonstration, avec un contrôle d'invariance avant envoi. Le script n'est pas un exécuteur et n'implémente pas le parseur historique, le journal d'exposition ou la mesure F1. Le pilote doit comparer les 4 bras à coût équivalent, avec sorties vides valides, journal des expositions E et propositions V, et application de la règle de sélection prédéfinie.

## Attention aux tailles

Le M0 produit des blocs complets même s'ils dépassent le contexte disponible. Ne pas les tronquer silencieusement. Auditer d'abord taille et tokens, puis choisir un sous-échantillon de développement sans labels ou signaler l'impossibilité technique. Le bras O et P autorise toutes les paires ; l'énumération explicite de ces listes peut augmenter le prompt par rapport à l'extraction libre historique : pour une comparaison rigoureuse, ne pas sérialiser les listes dans O et P et conserver le prompt documentaire de référence.

## Résultats disponibles

Tests d'invariance locaux : 100 graphes synthétiques pseudo-aléatoires validés ; deux petits documents exemples. **Aucun F1 de M0 n'a été mesuré.** Les données et caches `workere` ne sont pas inclus.
