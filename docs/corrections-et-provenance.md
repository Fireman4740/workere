# Écarts de sources et valeurs retenues — 12 novembre 2026

Ce registre accompagne la [synthèse courante](synthese-resultats.md) et le [catalogue détaillé](catalogue-experiences.md). Les sources anciennes restent conservées. Un titre, un statut ou une ancienne description ne remplace pas le protocole effectif et les comptes du résultat final.

## 1. Priorité de lecture

Pour une valeur mise en avant : identifier le run et sa configuration, vérifier le panel et le parseur, lire le résumé final, puis appliquer les corrections documentées. Une matrice TP/FP/FN avec recomptage des documents a priorité sur un chiffre isolé dans un paragraphe. Les désaccords restants sont signalés ; ils ne sont pas résolus en choisissant le meilleur score.

## 2. Réponses finales et réflexions : E1/E2/E3/E5

L’[audit structuré du 6 novembre](../SOURCES.md#annexes-locales) spécifie 47 documents CTB, 140 positifs orientés et 4 750 candidats orientés. Il n’extrait les réponses qu’après le dernier `</think>`. Une réponse finale absente donne la prédiction vide prévue par ce replay, sans retirer le document. Les compteurs documentaires et le F1 agrégé ont été recomptés pour cette mise à jour.

| Système | TP | FP | FN | F1 retenu |
|---|---:|---:|---:|---:|
| E1 contrôle |42|26|98|40,38|
| E1 température 0,6 |48|30|92|44,04|
| E1 budget 8k |68|64|72|50,00|
| E2 contrôle |34|30|106|33,33|
| E2 arrêt anticipé |34|34|106|32,69|
| E3 contrôle |36|28|104|35,29|
| E3 intensité 0,5 |34|28|106|33,66|
| E3 intensité 1,0 |24|26|116|25,26|
| E3 aléatoire |32|22|108|32,99|
| E5 contrôle |24|24|116|25,53|
| E5 édition |34|30|106|33,33|
| E5 têtes aléatoires |36|26|104|35,64|

Les anciens journaux contiennent des scores sur d’autres intersections ou avec un parseur lisant des fragments de réflexion. Certaines descriptions ont ensuite été corrigées partiellement, mais donnent encore E1 température 43,64, E3 contrôle34,62 ou un delta E5 arrondi différent. Ces nombres ne sont pas mélangés au replay strict. E2 n’a plus des comptes identiques au contrôle sur les 47 documents : l’égalité d’une ancienne intersection ne démontre pas l’égalité sur le panel entier.

Ce contrôle ne régénère pas les réponses et ne refait pas tous les bootstraps. Les intervalles d’anciens résumés non compatibles avec les nouvelles matrices ne sont pas transposés. La conclusion limitée reste l’absence de bénéfice spécifique démontré de ces interventions, sans preuve générale d’équivalence ou d’absence d’effet.

Journaux sources : [run a5f034d1](../results/inventaire-executions.csv) (`a5f034d1-5e13-4d2f-9b7a-3daf00cafe79`) [run 5ce8a7ea](../results/inventaire-executions.csv) (`5ce8a7ea-342a-40ca-814b-01de968890a9`) [run 865d312c](../results/inventaire-executions.csv) (`865d312c-0ad3-4d0a-a46e-d3964470826f`) [run 25b64a47](../results/inventaire-executions.csv) (`25b64a47-33b5-44b8-bdb4-1b189df73bfb`).

## 3. W5/W6 : titre hérité et panel réel

- W5 conserve le titre « ESC40 », mais son run final porte sur **quatre documents ESC**. Le maximum observé54,17 est exploratoire parmi96 variantes corrélées sur ces quatre documents. Il n’est pas un résultat ESC40 ou ESC233. Journal [run aa041b15](../results/inventaire-executions.csv) (`aa041b15-ebb2-4cd1-8bbf-079aa72ff648`), ligne325, `WSP_COMBO_RESULT`, `complete=true`.
- W6 conserve le titre « MAVEN24 », mais son run final porte sur **quatre documents MAVEN**, dont deux de l’ancien panel et deux de l’extension. Le maximum41,94 n’établit pas un score MAVEN24/710 ni un bénéfice propre du steering. Journal [run 95133280](../results/inventaire-executions.csv) (`95133280-621a-4bb2-87cf-29aef6dca1ae`), ligne505, même marqueur final.

Dans W6, la note de phase et la description du nœud mentionnent18/20 lectures sans intervention et7/20 avec invalides. Le champ `costs.readings` du résumé final local compte au contraire **16/20 et9/20**. La synthèse utilise ces derniers compteurs, en conservant explicitement le désaccord. Les deux sources signalent le même problème qualitatif : de nombreuses sorties invalides, réparties différemment entre conditions. Le contraste base/steering est donc confondu par la validité du format. Aucune amélioration scientifique n’est conclue de cet écart.

Les nouvelles extensions W7/W8 sont des nœuds distincts. Leur soumission et leur nouveau schéma ne leur donnent pas un résultat par anticipation.

## 4. États techniques et résultats scientifiques

- N13 comporte des runs terminés dont le résultat est `BLOCKED_VALIDITY`, avec 63/72,69/72 et68/72 sorties valides. Aucun score scientifique complet n’en est déduit.
- MV1 est marqué `lost`, mais les deux lignes `MV_RESULT` CTB/ESC sont présentes dans son journal. MV2 les rejoue et ajoute MAVEN ; ce n’est pas une réplication indépendante.
- Les tentatives W4–W6 perdues ou interrompues ont sauvegardé des lectures réutilisées dans les runs finaux. Les reprises ne sont pas autant d’expériences indépendantes.
- La reprise CTB seule est encore marquée en annulation dans l’instantané ; aucun score complet validé n’est publié ici. Le compteur de suivi n’est pas un temps GPU actif.
- Les plans FULL CTB183/ESC233/MAVEN710 sont trois expériences sans run : aucune métrique n’est créée à partir de leur titre ou de leur recette prévue.

## 5. Corrections historiques maintenues

J3/RES30 emploie le gold symétrisé corrigé, conformément au [rapport de correction](../SOURCES.md#annexes-locales). Les F1 bruts du journal ne sont pas remis en circulation.

Les anciens88 documents qualifiés de réserve ont été inclus dans H1 ; la [trace d’exposition](../SOURCES.md#annexes-locales) empêche de les annoncer comme une réserve encore intacte à l’échelle du projet.

Les chiffres DeepSeek des corpus complets restent historiques et descriptifs, avec prédictions sources non disponibles pour la nouvelle comparaison. Ils ne reçoivent pas de test apparié construit à partir de seuls scores finaux.
