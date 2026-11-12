# Atlas web des résultats workere

Ouvrir **[index.html](index.html)** directement dans un navigateur. Le fichier contient les données, les styles et le JavaScript : aucun serveur, compte, paquet, CDN ou accès réseau n’est nécessaire.

L’instantané scientifique du **12 novembre 2026 à 12:31 UTC** couvre 117 expériences et 190 exécutions. Le site ajoute aux documents Markdown :

- un catalogue avec recherche, filtres de méthode, corpus et état, et navigation conservant la sélection ;
- 117 explications courtes, 117 schémas individuels détaillés et les protocoles complets ;
- 828 lignes de scores, avec graphiques, intervalles publiés, tableaux exacts, provenance et exports CSV/SVG/JSON ;
- des graphiques d’économies de budget et de validité lorsque le F1 n’est pas le résultat pertinent ;
- les gabarits de prompt et leurs fonctions d’assemblage, relus aux commits de référence, consultables et exportables en texte ;
- une page **Top 10**, un mode présentation et une mise en page pour mobile.

## Lire le Top 10

Le classement par défaut utilise CTB47. Les sélections ESC40/TEST-S40 et MAVEN12 sont séparées. On retient au plus dix méthodes expérimentales éligibles, avec leur meilleur micro-F1 observé sur ce panel. Les variantes de référence recopiées ne deviennent pas de nouvelles méthodes ; MV1 est représenté par sa continuation MV2. Les petits pilotes W1/W2/W5/W6 sont exclus de ces panels.

La sélection de variantes reste **exploratoire** : les budgets, les démonstrations et la supervision peuvent différer. Le rang n’est pas une preuve de supériorité sur une réserve indépendante. ESC et MAVEN peuvent comporter moins de dix méthodes éligibles ; aucun résultat n’est inventé pour compléter dix places.

Les 18 fiches classées (10 CTB, 6 ESC, 2 MAVEN) expliquent **la méthode entière**, y compris les lectures et votes hérités. Chaque fiche précise les entrées, les étapes jusqu’à la liste finale, un exemple fictif, l’intérêt recherché, la plus-value observée face à la bonne référence, les limites, les ressources et le protocole d’évaluation. Les prompts des étapes amont sont consultables sur place. Ouvrir « Comprendre toute la méthode » suffit ; aucune fiche parente n’est nécessaire. Ces explications sont aussi intégrées aux onglets Schéma et Protocole des expériences concernées.

`method-guides.json` contient ces explications et leurs références explicites. Le classement garde ses règles de sélection ; les différences de F1 ne sont pas confondues avec une preuve de progrès. En particulier, la variante W4 classée est comparée au vote historique qu’elle filtre, et non au contrôle `base5` plus faible. L’export JSON emporte le parcours complet, les comparateurs et la banque de prompts.

## Prompts et explications

`prepare_prompts.py` lit les fichiers Git au commit de référence de chaque expérience, sans exécuter le code de recherche. `data.json` contient une banque dédupliquée de 168 textes et fonctions, avec chemins, lignes, commits et empreintes des fichiers. 104 fiches disposent de sources de prompt directement intégrées ; les autres désignent un calcul sans nouvel appel ou un plan sans exécution.

Un **gabarit** montre le texte fixe et les variables. Le document, les mentions et les exemples autorisés dépendent de la requête ; leur corps dynamique n’est pas remplacé par une nouvelle génération dans cet export. Les fonctions d’assemblage documentent cette construction. Un calcul de fusion ou de cache sans nouvel appel renvoie à son lecteur amont. Les plans ne présentent pas de prompt comme effectivement envoyé. Le commit de référence ne représente pas toutes les versions de toutes les reprises.

Les résumés en français et les schémas sont des explications de l’implémentation documentée. Les schémas distinguent entrées, configuration, intervention, réponses, parsing, agrégation et évaluation ; ils ne représentent pas des traces internes observées du modèle.

## Construire le fichier autonome

Python 3, bibliothèque standard uniquement :

```sh
python3 web/build_site.py
python3 web/test_site.py
node --check web/app.js
```

La construction ordinaire lit seulement les fichiers du dossier `web/`. Elle ne nécessite ni les archives originales ni les objets Git des expériences. Le résultat est déterministe.

Pour réimporter **le même instantané revu**, depuis le dépôt de recherche disposant des commits et des archives locales :

```sh
python3 web/prepare_data.py /chemin/vers/les/artefacts/workere
python3 web/build_site.py
python3 web/test_site.py
```

L’import relit les sources de l’instantané fixé dans le script ; il ne découvre pas automatiquement de nouvelles expériences ou exécutions. Une actualisation scientifique exige un nouvel inventaire revu et l’adaptation des correspondances et corrections. Aucun script ne lance d’inférence ni ne publie le site.

## Provenance et contrôles

Les comptes TP/FP/FN sont conservés et le F1 est recompté. Fractions et pourcentages sont convertis explicitement. Les lignes de progression, scores documentaires isolés et sorties WSP partielles sont exclus. Les oracles sont séparés dans l’interface. Les références historiques non appariées ne sont pas mises dans le même graphique que H1.

Les corrections J3 et E1/E2/E3/E5 sont appliquées. P1-Qwen suit également le replay strict 56,49 (74/48/66), au lieu du score historique 56,06 (74/50/66) ; aucun IC historique n’est transféré à ce recomptage. Voir le [registre des corrections](../docs/corrections-et-provenance.md).

`test_site.py` vérifie la couverture, les calculs, les corrections, les sources des prompts, les panels W4/W5/W6, les mesures absentes, les références des 18 fiches complètes et le caractère autonome/reproductible du HTML. `qa_browser.js`, exécuté dans un vrai navigateur via Playwright CLI, vérifie les 585 onglets, les 117 schémas, les trois classements, la navigation, les filtres, les exports et trois largeurs d’écran en mode hors ligne. `qa_methods.js` contrôle les parcours complets, les comparateurs W4/P2, les prompts intégrés et l’export des explications. Le résultat de validation est enregistré dans [validation.json](validation.json).

Les journaux bruts, corps des corpus, poids de modèles et archives de calcul ne sont pas ajoutés à cette branche. Les liens externes de code ne s’ouvrent qu’à la demande.
