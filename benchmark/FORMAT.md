# Contrat public, version 2

Une ligne JSON représente un document. Le contrat est validé par `schema.py`.
`workere-document/v2` contient `id`, `tokens`, `sentences` (intervalles de tokens
`[début, fin)`), `lang`, `text` et `token_char_offsets`. Le texte canonique est
`" ".join(tokens)` : les offsets caractères concernent ce texte, pas le fichier
source avant sa tokenisation. Ils comptent les caractères Unicode Python, pas les
octets ni les unités UTF-16. Les IDs sont des chaînes non vides sans virgule.

```json
{
  "schema": "workere-prediction/v2",
  "id": "example",
  "entities": [
    {"id": "a", "token_indices": [0], "char_offsets": [[0, 4]], "label": null, "confidence": 0.9},
    {"id": "b", "token_indices": [2], "char_offsets": [[12, 20]], "label": null, "confidence": 0.9}
  ],
  "relations": [
    {"source": "a", "target": "b", "label": "causes", "confidence": 0.82}
  ],
  "metadata": {}
}
```

Cet exemple suppose les tokens `Rain caused flooding` : l'offset de `flooding` est
`[12,20)`. Un événement discontinu possède plusieurs indices et segments caractères.
Les indices doivent être triés, uniques, non vides et dans le document. Les types
d'entités sont optionnels ; les métriques d'extraction actuelles sont non typées.
Les relations reprennent exactement `causes`, `causedby`, `norel`. `causal` est la
sortie binaire sans direction. Aucun label hors convention n'est silencieusement
corrigé par le scoring historique. Une confiance est `null` ou finie dans `[0,1]`.
Le scoreur évalue les classes enregistrées ; il n'applique aucun seuil implicite.

Les prédictions JEV ajoutent dans `metadata.decision_scores` une entrée par candidat :
`source`, `target`, `positive_probability`, `cache_key`, `cache_hit`, `served_model`.
Choice ajoute `class_probabilities` et le `provider_confidence` indépendant s'il est
fourni. `relations[].confidence` est la probabilité de la classe choisie. En Noul,
c'est `p` pour `causal`, `1-p` pour `norel`. Une probabilité n'est pas une garantie de
calibration. Le seuil est enregistré séparément.

Le protocole `relation_gold_mentions` (nom interne `relations_gold`) donne au modèle
les mentions et candidats de référence, sans relation de référence. Les protocoles
`entity_extraction` et `end_to_end` ne donnent que le document. Le pipeline peut
également être monolithique, avec la même sortie. Les endpoints inconnus restent
mesurables comme erreurs et ne sont pas supprimés par une validation permissive.
Les documents absents sont refusés ; un résultat vide explicite est une prédiction.

Le répertoire d'un run conserve la suite privée, le YAML source et résolu, les
entrées publiques, les prédictions, les métriques, les erreurs, le journal immédiat
`decisions.jsonl`, l'usage, les versions et empreintes du code. Le journal conserve
les questions et réponses structurées brutes, jamais les credentials. La suite
contenant les annotations est réservée à l'évaluateur. Les plugins Python sont du
code de confiance : le contrat ne remplace pas une isolation du système de fichiers.
