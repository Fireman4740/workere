# M0 — continuation expérimentale, 27 octobre 2026

Le ZIP original et `m0.py` restent inchangés. Le nouveau code implémente les fonctions qui manquaient : préparation d'entrées publiques, vraies démonstrations hors topic, rendu/remappage cohérent, contrôles, exécuteur batch borné, journal E/V et mesure séparée.

## Contrat et phases

Commande héritée et fixe : `bash experiments/jev/run.sh`, exclusivement via le lanceur d'expériences configuré. Le script détecte la configuration M0 versionnée, exécute les tests puis le pilote dans un environnement isolé Python 3.13 (bibliothèque standard, pas d'ajout aux dépendances du projet).

1. **Nœud de faisabilité hors ligne** : vérifier les 233 documents publics ESC et les prompts des huit documents de développement. Aucun appel de génération, aucun F1 de méthode.
2. **Enfant smoke** : deux documents de développement, choisis sans labels avant les appels, quatre bras O/P/M/MP, sept passages chacun : 56 requêtes logiques. Même transport batch, modèle demandé Luna, effort high, limite de sortie 32768 et calendrier des démonstrations. Plafond de réservation 2 USD ; 0,025 USD par requête soit 1,40 USD réservés. Aucun retry payant ni fallback modèle. Le catalogue et les credentials doivent passer les contrôles avant POST. Les prix de catalogue doivent rester sous les plafonds de réservation 0,10/0,50 USD par million de tokens entrée/sortie ; aucun rabais batch n'est présumé. Ce sont des plafonds de planification, pas un coût mesuré.
3. **Suite éventuelle** : les six autres documents ne sont pas lancés automatiquement. Examiner le smoke : conformité, modèle servi, coût, précision des ajouts et inflation de FP. Aucun E1/E2 avant bilan. Un résultat sur deux documents n'établit ni gain global ni réfutation statistique définitive.

Chaque nœud ayant répondu est gelé ; toute suite est un enfant, pas une modification rétroactive.

## Données et choix sans gold

- Révision ESC : `e6418e8747ac1c7fecf66d56ae41fbde7fd5967a`.
- `data/public_train233.jsonl` : whitelist texte intégral, mentions/spans/phrases, candidats originaux. Ni relations, ni coréférence, ni origine causale gold.
- Huit documents de DEV44 **déjà consommé**, quatre strates de taille de candidats, deux documents par strate et topics distincts ; seed 42. Smoke aux extrêmes de ces strates. Pas de nouvelle confirmation ni réserve intacte.
- Démonstrations : banque historique de 20 exemples vérifiée contre le registre DEV44 ; sept tirages hors topic avec exactement `Random('42::' + doc_id)`, identiques entre bras. Leurs réponses gold sont des démonstrations autorisées, pas les labels des paires évaluées.
- `data/evaluation_only.json` : scores Luna/d1 historiques, décision gelée à 0,39 et labels ; ouvert pour l'évaluation après sauvegarde et hash des générations, du journal et des prédictions.
- Préparation reproductible : `uv run --isolated --no-project --with pyarrow --with pyyaml python eci_m0_pilot/prepare_data.py --arrow <cache-train.arrow>`. Les empreintes et le commit source figurent dans `data/provenance.json`.

## Déviations déclarées / points de vigilance

- Le ZIP équilibre chaque strate à une unité près mais fait tourner ses restes indépendamment : l'équilibre **total** à une unité près n'est pas garanti. Le planificateur n'est pas réécrit ; les écarts sont mesurés. Une correction serait une variante séparée.
- Les bras O/P ne sérialisent pas leurs listes de candidats. Seuls M/MP affichent leurs masques. Le document complet et l'ordre lexical restent inchangés.
- Clarification commune aux quatre bras : une sortie vide est valide même avec mentions. La référence gelée n'est pas recalculée avec cette clarification ; O est un contrôle documentaire contemporain, pas une reproduction byte-identique de l'ancien prompt.
- Le parseur historique est conservé, avec gardes communs : schéma manquant, ID inconnu, troncature ou tableau final ambigu sont invalides, pas des négatifs. Le parseur historique pouvant préférer un tableau antérieur non vide à un `[]` final, ce cas est mis en quarantaine sans récupération opportuniste.
- Remappage séparé des namespaces du document, des démonstrations et des exemples de format ; tous les sites d'IDs structurés sont remappés. Les nombres du texte lexical ne sont jamais remplacés. Déduplication après inversion.
- La livraison autorisée et valide est distinguée de la soumission et de la planification. Une omission valide n'est pas un rejet explicite ; `norel` explicite alimente R. Pas d'ajout hors masque. `V/E=1/1` n'est jamais fusionné comme vote historique sur sept.
- Admission unique : baseline UNION propositions valides parmi les anciens zéro-vote négatifs sous baseline. Aucun retrait, recalibrage, entraînement ni propagation.
- Égalité de sept appels n'est pas égalité de calcul. Mesurer entrée, sortie, détails cache/raisonnement lorsqu'exposés et coût batch total. Les coûts batch ne sont pas faussement répartis en coûts par bras. Le proxy token pondéré est indiqué comme proxy. Un écart O/M >5 % interdit l'étiquette « coût égal » ; pas de gain de méthode avant contrôle d'enveloppe séparé si nécessaire.
- Les aliases de modèle servis autorisés reproduisent la validation historique ; ils ne prouvent pas l'identité des poids derrière un alias. Cette limite est déclarée et aucun autre modèle n'est substitué.

## Conservation et reprise

Le payload est hashé ; le reçu de soumission est écrit avant l'unique POST dans un répertoire durable du serveur. Un POST à issue inconnue n'est jamais renouvelé automatiquement. Les GET de suivi peuvent être réessayés. Les réponses et prompts sans credentials sont exportés en gzip/base64 dans le journal suivi, avec SHA256, puis réhydratés en artefacts locaux. Les tokens de raisonnement ne sont pas ajoutés une deuxième fois au total de sortie.

Tests : `uv run --isolated --no-project --python 3.13 python -m unittest discover -s eci_m0_pilot -p 'test_*.py' -v`. Les tests synthétiques ne sont pas des mesures de performance.
