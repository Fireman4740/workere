# Plan d'expériences — battre Mercury Noul (et DeepSeek) sur ESC

Point de départ : paquet `esc-mercury-15-v1`, 15 documents train, 8 174 paires, 630 positives. Baseline Mercury Noul figée **P 32,09 / R 32,70 / F1 32,39**, reproduite exactement hors ligne (ΔF1 = 0,0 ; évaluateur à lancer avec Python ≥ 3.10).

Objectif : un système **figé sur dev**, puis mesuré une seule fois sur un ensemble de confirmation jamais inspecté. Il doit battre en appariement (1) Mercury Noul et (2) le pipeline DeepSeek historique rejoué sur les mêmes documents.

## 0. Rôles des ensembles (pré-enregistrés)

Sélection par seed sur les identifiants et le nombre de mentions uniquement, **sans labels** : [document-selection.json](document-selection.json), [script](select-documents.py).

| Ensemble | Docs | Paires ordonnées | Rôle | Interdit |
|---|---:|---:|---|---|
| **D15** `esc-mercury-15-v1` | 15 (train) | 8 174 | diagnostic, non-régression, comparaisons appariées exploratoires | choisir prompt, seuil ou règle |
| **DEV-SCREEN** | 10 (5 par topic 37/41, seed 2026) | 6 894 | tri rapide des variantes du round 1 | rapporter comme résultat final |
| **DEV-FULL** | 25 (topics 37, 41) | 19 028 | confirmation des finalistes, réglage seuil/agrégation en LODO | — |
| **CONF-40** | 40 (train \ D15, 2 par topic × 20, seed 2027) | 19 174 | **une seule** évaluation finale, appariée | toute inspection avant le gel |

DEV est le split de développement standard de Gao (topics 37 et 41). `41_9ecbplus.xml` a déjà été vu (smoke) ; il reste dans DEV-FULL mais pas dans DEV-SCREEN.

## 1. Contrats fixes

- Protocole `relation_gold_mentions`, candidats historiques complets (deux ordres, tous les négatifs), F1 binaire micro sur paires ordonnées, intra/inter rapportés.
- Modèle Mercury `inception/mercury-decide-20260930`, gratuit, `allow_paid: false`, 20 RPM, cache SQLite partagé.
- Seuil 0,5 tant qu'aucun réglage n'est fait sur dev ; tout réglage est fait en leave-one-document-out (LODO) sur DEV-FULL.
- **Coréférence ECB+ = information gold.** Tout résultat qui l'utilise est rapporté dans une colonne « +coref oracle » séparée, comme la baseline historique à 50,6.
- Un facteur change par nœud. Probabilités, requêtes et réponses toujours sauvegardées. Prédictions exportées au format `pairs` du paquet et scorées par `evaluate.py`.

## 2. Budget d'appels (20 RPM ≈ 1 200 appels/h, hors retries et quotas)

| Ensemble | Ordonné | Non ordonné (1 appel/paire) |
|---|---:|---:|
| DEV-SCREEN | 6 894 ≈ 5,7 h | 3 447 ≈ 2,9 h |
| DEV-FULL | 19 028 ≈ 15,9 h | 9 514 ≈ 7,9 h |
| CONF-40 | 19 174 ≈ 16,0 h | 9 587 ≈ 8,0 h |
| D15 | 8 174 ≈ 6,8 h | 4 087 ≈ 3,4 h |

Les quotas journaliers gratuits sont inconnus. Prévoir des reprises via le cache. Le passage « non ordonné » (variante 1c) divise tout le reste du plan par deux s'il ne dégrade pas.

## 3. Arbre d'expériences (buissons empilés)

```
Mercury Noul train233 (existant, figé : D15 = 32,39)
└─ R0  Mercury Noul — DEV-FULL                           baseline dev, aucune modif de code
   ├─ R1a  Question alignée ESC (PLOT_LINK)              ┐
   ├─ R1b  Mentions balisées dans le texte               │ round 1 : « quel juge ? »
   ├─ R1c  Requête non ordonnée (1 appel / paire)        │ tri sur DEV-SCREEN,
   ├─ R1d  Questions décomposées dans le même appel      │ 2 finalistes sur DEV-FULL
   └─ R1e  2–4 démonstrations récupérées (train)         ┘
        └─ R2  Gagnant R1 + post-traitement événementiel (0 appel)
             ├─ R2a  symétrie + seuil LODO
             ├─ R2b  + chaînes prédites (question « même événement »)
             └─ R2c  + chaînes ECB+ (colonne oracle)
                  └─ R3  Combinaison des 2 meilleurs facteurs R1 (si compatibles)
                       └─ FINAL  gel → CONF-40 (+ D15 en diagnostic)
```

### Phase 0 — hors ligne, 0 appel (≈ 1 h)

1. Rejouer l'analyse de marge sur **D15** (symétrie, agrégation par chaîne, LODO) pour vérifier que l'effet observé sur 13 docs (+10,9 avec chaînes ECB+) tient sur 15. Le diagnostic seulement, sans choisir de règle.
2. **Écrire et figer les règles R2 avant tout appel** : symétrisation = moyenne des deux ordres ; agrégation = max sur les paires de mentions entre deux chaînes ; diffusion à toutes les paires des deux chaînes.
3. Smoke API d'une requête **multi-questions** (prérequis de R1d) sur un document déjà vu (`41_9ecbplus.xml`, 156 paires) : vérifier qu'une requête à plusieurs `noul` renvoie des réponses indépendantes et que `causal` reste identique à la baseline quand les autres questions sont ajoutées. Sinon R1d coûte k appels par paire et passe en dernier.

### R0 — baseline dev (19 028 appels, ou 6 894 si DEV-SCREEN d'abord)

Configuration identique au run train233, seule la liste de documents change. Fournit la référence appariée de tous les nœuds suivants. Commencer par DEV-SCREEN (5,7 h), compléter DEV-FULL en tâche de fond.

### R1 — round « juge » : 5 siblings sous R0, triés sur DEV-SCREEN

| Nœud | Changement unique | Hypothèse mesurable | Appels DEV-SCREEN |
|---|---|---|---:|
| **R1a** question ESC | Définition PLOT_LINK : cause, facilitation, prévention, conséquence narrative, sous-événement ; exclure « marqueur causal seul » et coréférence | ↑ AP, ↑ rappel sur liens implicites/scénario sans ↑ FP | 6 894 |
| **R1b** balisage | `[A]…[/A]`, `[B]…[/B]` dans le document complet, question inchangée | ↓ erreurs d'ancrage (surfaces répétées, marqueur vs événement) | 6 894 |
| **R1c** non ordonné | Une requête par paire non ordonnée, ordre canonique (position dans le texte), prob. copiée aux deux ordres | F1 ≥ R0 − 1 pt **et** coût ÷ 2 ; élimine les 7 % d'incohérences A/B | 3 447 |
| **R1d** décomposition | Mêmes requêtes + questions `noul` : même événement ? A avant B ? lien explicatif/narratif ? connecteur explicite ? ; régression logistique LODO sur DEV | ↑ AUROC/AP via signaux complémentaires (cf. MEFA) ; fournit les chaînes prédites pour R2b | 6 894 (si multi-questions OK) |
| **R1e** démonstrations | 2–4 exemples (moitié pos., moitié nég.) récupérés par similarité de paire dans train \ (D15 ∪ CONF-40), topic différent | ↑ précision et ↑ AP (SERE +7,6, CarICL −8,7 sans récupération) | 6 894 |

**Règle de promotion (fixée maintenant)** : critère principal = **AP** (indépendant du seuil, c'est le goulot identifié) puis F1 à 0,5 ; ΔAP > 0 avec bootstrap documentaire apparié P(Δ > 0) ≥ 0,9. Les deux meilleurs passent sur DEV-FULL. Le gagnant DEV-FULL devient parent de R2. Égalité ⇒ le moins cher (R1c gagne à performance égale).

Ordre de lancement conseillé (coût/valeur) : R1c → R1a → R1b → R1d → R1e. Si R1c tient, les suivants sont lancés en mode non ordonné : moitié moins d'appels.

### R2 — post-traitement événementiel sur le gagnant (0 appel)

- **R2a** symétrisation + seuil LODO sur DEV-FULL. Attendu ≤ +2–4 pts.
- **R2b** agrégation max sur chaînes **prédites** (question « même événement » de R1d, ou appel dédié sinon : ≈ 1 appel par paire non ordonnée, mutualisable). Voie non oracle.
- **R2c** même agrégation avec chaînes **ECB+** (colonne +coref oracle). Attendu ≈ +10 pts (13 docs : +10,9, IC [+6,0 ; +14,5]).

### R3 — combinaison

Empiler le 2ᵉ meilleur facteur R1 sur le gagnant R2 seulement s'ils sont orthogonaux (ex. question ESC + démonstrations). Une seule combinaison, pas de recherche combinatoire.

### FINAL — gel puis CONF-40

1. Geler commit, config, seuil, règles, liste CONF-40 (déjà figée ici).
2. Lancer sur CONF-40 : **Mercury Noul baseline** et **système final** (≈ 8 + 8 h en non ordonné, 16 + 16 h sinon).
3. **DeepSeek historique rejoué** (`run.py`, prompt lean coref, 7 passes, 1 shot) sur CONF-40 et D15 : seul moyen d'avoir un delta apparié avec la baseline 46,6. **Payant** (≈ 4,50 \$ pour 233 docs historiques, donc ≈ 0,8–1 \$ pour 40 + 15 docs) ⇒ demande d'autorisation explicite.
4. Rapporter : brut et +coref oracle, intra/inter, P/R/F1, ΔF1 apparié avec IC bootstrap documentaire, coût en appels. D15 rapporté à part, étiqueté « diagnostic, déjà inspecté ».

## 4. Critères de succès pré-enregistrés

| Niveau | Critère sur CONF-40 |
|---|---|
| Minimal | Système final > Mercury Noul, IC 95 % apparié du ΔF1 > 0 |
| Cible | Système final (brut) ≥ DeepSeek rejoué (brut), apparié |
| Ambitieux | Final +coref oracle ≥ DeepSeek +coref oracle ; viser ≥ 50 F1 |

Plancher de bruit : sd entre runs DeepSeek 7 passes ≈ 0,5 pt sur 233 docs (`notes/pass_variance.md`) ; plus large sur 40 docs. Mercury étant quasi déterministe (1 token), mesurer sa variance sur 300 paires répétées de DEV-SCREEN (dans R0, coût marginal faible via un second passage sans cache).

## 5. Analyses à produire à chaque nœud (hors ligne)

AUROC, AP, F1 à 0,5, F1 LODO, intra/inter, incohérence A/B, rappel par distance et par origine (annoté vs propagé), précision au niveau événement, top-20 FP/FN avec contexte. Même script pour tous les nœuds, rejoué aussi sur D15 en non-régression.

## 6. Risques

- **Quotas gratuits** : un nœud interrompu n'est jamais rapporté comme complet ; il reprend depuis le cache.
- **Multi-questions non supporté** : R1d et R2b coûtent alors k appels/paire ; décaler après R1a/R1c.
- **Fuite** : démonstrations R1e jamais tirées de D15, CONF-40, ni du topic du document évalué.
- **Surajustement dev** : 2 topics seulement ; d'où CONF-40 stratifié sur les 20 topics train.
- **Comparabilité 46,6** : chiffre historique sur 233 docs, prédictions perdues ; seule la réplication appariée de DeepSeek autorise une affirmation.
