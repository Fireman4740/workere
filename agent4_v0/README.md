# Agent 4 V0 — interfaces d’intégration

Voir [le rapport d’intégration et les commandes](../docs/agent4-v0-integration.md).

Le kit d’origine est conservé dans `kit/` ; il ne fait aucun appel API. Le résultat
MOCK contraste=1 de sa fixture est généré, jamais mesuré. Les nouvelles interfaces
sont séparées, sans changement de la référence gelée :

| Module | Rôle |
|---|---|
| `convert_cache` | Allowlist des signaux DEV, exclusions confirmatoires/réserves, offsets vérifiés, sélection hash avec le kit |
| `preflight` | Citations/propositions non approuvées, snapshots historiques et exposition H1 ; aucun réseau |
| `review` | Reçu humain externe, indépendant et aveugle ; hashes et préparation stricte V0 |
| `adapter` | Dry-run par défaut ; transport batch historique épinglé, budget persistant avant tout appel explicite |
| `score` | Contrastes et votes orientés sans gold ; refuse les jeux vides/MOCK ; aucun F1 |

Tests logiciels :

```bash
uv run --locked --no-sync python -m unittest tests.test_agent4_v0 -v
uv run --locked --no-sync python -m unittest discover -s agent4_v0/kit -p tests.py -v
```

Les données réelles sont des artefacts locaux, pas des fixtures commitées. Il faut
les caches et worktrees historiques pour reproduire leur conversion. Les sorties
scellées ne doivent pas être écrasées.

## Revue externe

Le reçu à fournir à `review` doit lier la somme SHA-256 du fichier complet des
propositions ; chaque décision référence le `proposal_sha256` exact. Schéma :

```json
{
  "reviewer_type": "human",
  "reviewer": "IDENTITE_REELLE_A_RENSEIGNER",
  "independent": false,
  "blind_to_gold": false,
  "blind_to_model_outputs": false,
  "proposals_sha256": "HASH_A_RENSEIGNER",
  "decisions": [
    {
      "proposal_sha256": "HASH_DE_LA_PROPOSITION",
      "decision": "uncertain",
      "semantic_checks_complete": false,
      "comment": "Revue non encore effectuée"
    }
  ]
}
```

**Ce gabarit n’est pas une revue.** Ne renseigner les attestations et une décision
`approved` qu’après une vraie revue humaine indépendante du proposeur. Toutes les
propositions doivent avoir une décision. Les propositions sans spans ou sans
placebos doivent d’abord être complétées et rescéllées dans une nouvelle version ;
le reçu doit alors viser cette nouvelle version. Documenter les changements.

## Budget

Aucune enveloppe n’est fournie ni inventée. L’exécution demande un fichier
`approved=true`, une autorisation explicite, un plafond USD positif, les prix/frais
supérieurs avec source/date, un `ledger_path` absolu persistant, un scope neuf
préfixé `agent4-v0-` et les hashes des sources historiques du transport. Les
anciens budgets ne sont pas réutilisés. Ne pas confondre 84 appels maximum pour
**tout le V0** avec 84 appels **par paire** du protocole complet.

Le dry-run ne prouve pas l’accès actuel au modèle. Le chemin live s’arrête sur
modèle inattendu, réponse invalide, coût non conforme ou état d’envoi incertain ;
aucune erreur ne devient un vote négatif et aucune relance payante automatique
n’est faite. Les caches des répétitions originales sont distincts afin de ne pas
simuler une répétition par relecture d’une réponse historique.
