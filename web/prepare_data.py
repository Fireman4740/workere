#!/usr/bin/env python3
"""Import the reviewed research snapshot. No inference, network or model calls.

This optional step requires the original local artifacts. Normal HTML builds only
need the committed data.json. Adapters deliberately name result collections;
per-document scores, progress messages and invalidated raw J3/E scores are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPIC = "synthese-globale-2026-10-29/mise-a-jour-2026-11-12"
FAMILIES = [
    ("lecteurs", "Lecteurs et références", "Premiers modèles, lectures documentaires et passage à l’échelle."),
    ("consignes", "Consignes et représentations", "Formulation, démonstrations et représentation des occurrences."),
    ("fusion", "Fusion et calibration", "Symétrie, coréférence, scores combinés et seuils appris."),
    ("harness", "Adaptation du harness", "Choix des transformations et agrégations sans labels de la cible."),
    ("ancrage", "Preuves et ancrage", "Alias, preuves locales, transport et concurrence d’attribution."),
    ("budget", "Budget et réutilisation", "Nombre de lectures, arrêt exact et recombinaisons de caches."),
    ("mecanismes", "Mécanismes internes", "Décodage, sorties forcées, activations, têtes et probes."),
    ("votes", "Votes et diversité", "Échantillonnage, vérification et combinaisons de lecteurs."),
    ("juges", "Juges de paires et graphes", "Contexte, décomposition, réparation et sélection de graphes."),
    ("poids", "Interventions sur les poids", "Éditions de rang un, contrôles aléatoires et confirmation réservée."),
]


def family(n):
    if n >= 110: return "poids"
    if 90 <= n <= 101 or n in [33,34,35,36,38]: return "juges"
    if 79 <= n <= 89 or 102 <= n <= 109: return "votes"
    if 74 <= n <= 78: return "mecanismes"
    if n in [52, 58, 60, 61, 68, 69, 73]: return "budget"
    if n in [41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 55]: return "ancrage"
    if n in [23, 24, 25]: return "harness"
    if n in [14, 15, 18, 26, 27, 28, 29, 30, 31, 32, 37]: return "fusion"
    if n in [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 39, 40, 49, 64]: return "lecteurs"
    return "consignes"


# Each intervention has its own diagram node, rather than a generic copied flow.
INTERVENTIONS = """Span · question oui/non
Mercury · question oui/non
Mercury · trois classes causales
Occurrences A/B explicites
Mercury · ESC233 interrompu
d1 · prototype DEV25 + D15
d1 · référence R0
Luna · lectures documentaires répétées
Une paire non ordonnée par appel
Définition causale ESC
Balisage des mentions A et B
Démonstrations récupérées
Plusieurs questions par appel
Symétrie + seuil hors document
Coréférence prédite + propagation
Symétrie + seuil 0,65 gelés
Luna · confirmation CONF40
Moyenne Luna × d1 · seuil 0,39
Démonstrations avec rationales
Groupes de coréférence explicites
Consignes sémantiques renforcées
Démos structurelles et négatives
Harness d1 · transformations
Harness Luna · agrégations
Harness fusion · agrégations
Accord de deux procédures · brouillon
Juge dans la bande [0,25 ; 0,55]
Accord de coréférence Luna × DeepSeek
Moyenne J2 calibrée · RES30 corrigé
Logistique J2 · RES30 corrigé
Logistique sur deux scores
Logistique + structure
Luna · jugement par paire
Cascade de types → élimination
Proposition des causes par cible
Événements → recopie aux mentions
Ensemble hétérogène du round K
Luna · paire par paire DEV10
Haiku + vote / fusion d1
Qwen documentaire · premier essai
Alias auxiliaires A/R
Planificateur de couverture M0
Preuves sources → transport v1.1
Bras O / P / M / MP
Éditions placebo V0
Concurrence de liaisons argumentales
Diversité M1 · brouillon
Lectures complémentaires vs ordinaires
Qwen xhigh · reprise
Filtre de preuve et d’ancrage
Transport v2 + vérificateur commun
Sous-ensembles exhaustifs 1 / 3 / 7
Démos ordinaires vs contrastives
DeepSeek vs Luna · validité
Décisions lexicales CTB · intégrité
Audit de reproductibilité du cache N3
Deux schedules · réplication N7
Lectures fraîches imbriquées 3 / 7
Retrieval × négatifs · plan factoriel
Six lectures vs vingt triplets
Arrêt par préfixe à décision fixe
Occurrence originale / locale / table
Représentations Qwen · validité
Reprise CTB · résultat non vérifié
Prompt × exécution bornée
Indice et démos · couche 32
Réflexion medium vs désactivée
MBR sur cache N14
Union brute sur cache N14
Dose de démonstrations
Direction opposée / aléatoire
Intervention · couche 40
MBR sur cache N15
Greedy / température / fermeture 8k
Sortie anticipée · replay strict
Direction de réflexion / aléatoire
Probe résiduel supervisé
Édition de têtes / aléatoire
Échantillonnage · cinq lectures CTB
ESC · cinq lectures à 10k
Consigne purpose / enablement
Cinq démos diversifiées
Vérification de R1
Vérification de N1
Budget de réflexion 20k
ESC · budget 24k
Vérification de P3
Vérificateur dense · logprobs
Vérificateur + démos par paire
Contexte local / entier × réflexion
Cartes événementielles × budget
Paires natives · exécution partielle
Paires API · direct / preuves
Rejeu exact des paires
Juge direct · deux orientations
Décomposition du jugement
Ajouts sur négatifs unanimes
Répétition technique du juge
Extension du pilote ESC
Sélection d’un graphe candidat
Édition d’un graphe candidat
Réparation vote · CTB47
Réparation vote · ESC40
Réparation vote · MAVEN12
Cinq vues · continuation interrompue
CTB183 complet · plan
ESC233 complet · plan
MAVEN710 complet · plan
Cinq vues · recombinaison finale
Édition de rang un · screen W1
Édition gelée · réserve W2
Diversité des lecteurs W1 / W2
Édition × vote × vérification CTB47
Édition × vote × vérification ESC4
Édition × vote × vérification MAVEN4
Transfert ESC50 · en cours
Transfert MAVEN50 · en cours""".splitlines()
assert len(INTERVENTIONS) == 117


def load(path):
    return json.loads(path.read_text())


def tidy(text):
    """Repair joined prose/number spacing without changing model/corpus IDs."""
    text=re.sub(r"([a-zà-öø-ÿ]{4,})(?=\d)",r"\1 ",text)
    return re.sub(r"\b(à|de|sur|contre|soit|entre)(?=\d)",r"\1 ",text)


def import_snapshot(root: Path):
    src = root / TOPIC
    inventory = load(src / "inventaire-experiences.json")
    logindex = load(src / "index-journaux.json")
    records = load(src / "resumes-structures.json")["records"]
    runs = {r["id"]: r for r in logindex["runs"]}
    source_files = {}

    def source(path, pointer="", run=None, line=None):
        p = root / path
        if p.exists() and path not in source_files:
            source_files[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        return {"file": path, "pointer": pointer, "run": run, "line": line}

    technical = (HERE.parent / "docs/catalogue-experiences.md").read_text()
    sections = {}
    matches = list(re.finditer(r"^#### (\d+)\. (.+)$", technical, re.M))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else technical.index("\n## 4.", match.end())
        text = technical[match.end():end]
        text = re.split(r'\n(?:<a id=|### [A-H]\. )', text)[0].strip()
        sections[int(match[1])] = text

    experiments = []
    for e in inventory["experiments"]:
        n, ed = e["number"], e["editorial"]
        # The long description contains superseded scores; the reviewed editorial
        # and corrected technical catalogue are used instead.
        state = "mesure"
        if n in [6, 26, 47, 106, 107, 108]: state = "planifie"
        elif n in [116, 117]: state = "en_cours"
        elif n in [13, 40, 43, 54, 55, 63, 64, 65, 92]: state = "incomplet"
        elif n in [41, 42, 45, 46, 51, 56, 61, 66, 68, 69, 70, 71, 72, 73, 77, 98]: state = "diagnostic"
        text = " ".join(str(ed.get(k, "")) for k in ["title", "method", "evaluation"])
        datasets = [ds for ds, pattern in [("CTB", r"CTB"), ("ESC", r"ESC|DEV|CONF|D15|RES30"), ("MAVEN", r"MAVEN")] if re.search(pattern, text, re.I)]
        ex = {"id": e["id"], "number": n, "title": ed["title"], "original_title": e["title"],
              "family": family(n), "datasets": datasets or ["Technique"], "state": state,
              "method": tidy(ed["method"]), "evaluation": tidy(ed["evaluation"]), "outcome": tidy(ed["outcome"]),
              "checks": tidy(ed["checks"]), "technical": sections[n], "sources": ed.get("sources", []),
              "code": e["code_snapshot"], "branch": e["branch"], "archived": e["archived"],
              "runs": [], "rows": [], "diagnostics": [], "notes": [],
              "observations": [],
              "diagram": {"intervention": INTERVENTIONS[n-1], "kind": family(n),
                          "supervised": n in [12, 14, 18, 19, 22, 29, 30, 31, 32, 53, 59, 66, 70, 71, 72, 77, 82, 88, 89]}}
        for rid in e["run_ids"]:
            r = runs[rid]
            ex["runs"].append({k:r.get(k) for k in ["id", "status", "title", "commit", "duration", "updated", "log_available", "log_sha256", "line_count"]})
        experiments.append(ex)
    by_id = {e["id"]: e for e in experiments}
    by_n = {e["number"]: e for e in experiments}
    explanations=load(HERE/"explanations.json")
    assert len(explanations)==117
    for e in experiments:e["simple"]=explanations[e["number"]-1]

    def add(ex, label, score, context, srcinfo, unit="percent", docs=None, note="", oracle=False, delta=None):
        if not isinstance(score, dict): return
        lower = {k.lower(): v for k, v in score.items()}
        counts = lower.get("tp_fp_fn")
        if counts is None and all(k in lower for k in ["tp", "fp", "fn"]):
            counts = [lower[k] for k in ["tp", "fp", "fn"]]
        f1 = next((lower[k] for k in ["f1", "f1_points", "micro_f1", "mean_f1"] if isinstance(lower.get(k), (float, int))), None)
        if f1 is None: return
        factor = 100 if unit == "fraction" else 1
        row = {"variant": label, "context": context, "f1": f1*factor,
               "precision": lower.get("precision", lower.get("p")), "recall": lower.get("recall", lower.get("r")),
               "documents": score.get("documents", docs), "tp": None, "fp": None, "fn": None,
               "delta": None, "ci": None, "resampling": None, "ci_level": None,
               "reference": None, "oracle": oracle or "oracle" in label.lower(),
               "note": note or score.get("scope", ""), "source": srcinfo, "extras": {}}
        for key in ["precision", "recall"]:
            if isinstance(row[key], (int, float)): row[key] *= factor
        if counts and len(counts) == 3:
            row.update(dict(zip(["tp", "fp", "fn"], counts)))
            denominator = 2*counts[0]+counts[1]+counts[2]
            expected = 200*counts[0]/denominator if denominator else 0
            if abs(expected-row["f1"]) > .016:
                raise ValueError(f"F1/count mismatch #{ex['number']} {label}: {expected} vs {row['f1']}")
            # Ratios can be recomputed from the published aggregate counts. A
            # zero denominator stays undefined unless the source defined zero.
            if row["precision"] is None and counts[0]+counts[1]: row["precision"]=100*counts[0]/(counts[0]+counts[1])
            if row["recall"] is None and counts[0]+counts[2]: row["recall"]=100*counts[0]/(counts[0]+counts[2])
        d = delta or score.get("delta")
        if isinstance(d, list) and len(d) == 3:
            row.update(delta=d[0], ci=d[1:], ci_level=95, resampling=score.get("resampling_unit", "non précisée"))
        if isinstance(score.get("delta_points"), (int, float)):
            row.update(delta=score["delta_points"], ci=score.get("ci95"), ci_level=95 if score.get("ci95") else None)
        for k in ["threshold", "readings_per_document", "emitted_tokens_total", "mean_emitted_tokens", "emitted_tokens_ratio_to_single_base"]:
            if isinstance(score.get(k), (int, float)): row["extras"][k] = score[k]
        for k in ["vs_base5", "vs_historical"]:
            d = score.get(k)
            if isinstance(d, dict):
                row["extras"][k] = d
                if k == "vs_base5": row.update(delta=d.get("delta_f1_pp"), ci=d.get("ci95_pp"), ci_level=95, resampling=d.get("resampling_unit"), reference="base5")
        for k in ["quality", "cost", "incremental_cost", "emitted_token_cost", "paired_vs_baselines", "vs_baselines", "vs_qwen7", "vs_cached_n1_p3"]:
            if k in score: row["extras"][k] = score[k]
        ex["rows"].append(row)

    def map_scores(ex, collection, context, s, unit="percent", docs=None, prefix=""):
        if not isinstance(collection, dict): return
        for label, score in collection.items():
            add(ex, prefix+label, score, context, {**s,"pointer": s.get("pointer", "")+"/"+label}, unit, docs)

    # Verify recorded evidence against persisted log lines, and retain references
    # and selected aggregates only. No raw prompts or corpus text are published.
    log_cache = {}
    verified_records = 0
    for r in records:
        if r["marker"] in ["LUNA_BATCH_COMPLETED", "N6_RESPONSE"] or "EVIDENCE" in r["marker"]: continue
        ex = by_id[r["experiment_id"]]; n = ex["number"]; d = r["data"]; marker = r["marker"]
        run = runs[r["run_id"]]
        if run.get("log_available"):
            if run["id"] not in log_cache:
                log_cache[run["id"]] = Path(run["log_path"]).read_text(errors="replace").splitlines()
            line = log_cache[run["id"]][r["line"]-1]
            if marker not in line: raise ValueError(f"Missing evidence marker {r['run_id']}:{r['line']}")
            verified_records += 1
        s = {"file":"journaux/"+r["run_id"]+".log", "pointer":marker, "run":r["run_id"], "line":r["line"]}
        if marker == "WSP_COMBO_RESULT" and d.get("complete"):
            context = {113:"CTB47 · W4 · développement réutilisé",114:"ESC4 · W5 · pilote",115:"MAVEN4 · W6 · pilote confondu"}[n]
            ex["rows"] = []  # final complete record supersedes any repeated complete print
            map_scores(ex, d["results"], context, s, "fraction", {113:47,114:4,115:4}[n])
            ex["diagnostics"] = [{"label":"Coûts et validité des réponses", "data":d.get("costs"), "source":s}, {"label":"Comparaisons préspécifiées", "data":d.get("primary_comparisons"), "source":s}]
        elif marker == "WSP_DIVERSITY_RESULT":
            map_scores(ex, d["conditions"], d["cohort"]+" · CTB20 · cache exploratoire", s, "fraction", 20)
            ex["diagnostics"].append({"label":d["cohort"]+" · coûts et périmètre", "data":{k:d.get(k) for k in ["scope","missing_control","cost_caveat","source_run","source_invalid_answers"]}, "source":s})
            # W1 and W2 single-reader results are the exact same cached records,
            # cited as such; they are not new independent runs.
            target = by_n[110 if d["cohort"] == "W1" else 111]
            for label, score in d["conditions"].items():
                if score.get("readings_per_document") == 1:
                    add(target,label,score,d["cohort"]+" · CTB20 · "+("screen" if d["cohort"]=="W1" else "réserve"),s,"fraction",20,note="Recomptage des sorties W1/W2 importées par W3 ; pas une réplication.")
        elif marker in ["MV_RESULT", "VC_RESULT"]:
            context = d["dataset"]+" · "+str(next(iter(d["policies"].values())).get("documents", "?"))+" docs · développement réutilisé"
            map_scores(ex,d["policies"],context,s)
            for v in d["policies"].values():
                if "baseline" in v:
                    add(ex,"Référence historique",v["baseline"],context,s); break
            for k in ["coverage","cost","cost_note","agreement","baseline_quality"]:
                if k in d: ex["diagnostics"].append({"label":k,"data":d[k],"source":s})
        elif marker in ["AP_RESULT", "QG_RESULT"]:
            collections = [("metrics", d.get("metrics")), ("nouveaux documents uniquement", d.get("new_documents_only"))]
            for sys, coll in d.get("systems", {}).items(): collections.append((sys, coll))
            for sub, coll in collections:
                if not isinstance(coll,dict): continue
                for corpus, score in coll.items():
                    if not isinstance(score,dict): continue
                    context = f"{corpus} · {score.get('documents', '?')} docs · {sub}"
                    add(ex,d.get("method", sub),score,context,s)
                    if "baseline" in score: add(ex,"Référence historique",score["baseline"],context,s)
            for k in ["cost","cost_note","stability","coverage","scope","exact_replay"]:
                if k in d: ex["diagnostics"].append({"label":k,"data":d[k],"source":s})
        elif marker == "RESULT":
            v = r.get("variant", "résultat")
            corpus = v.split("/")[0] if "/" in v else "CTB"
            context = f"{corpus} · {d.get('documents', 47 if n in [88,89] else '?')} docs · même protocole"
            add(ex,v,d,context,s,docs=47 if n in [88,89] else None)
        elif marker == "H2_RESULT" and d["micro"]:
            map_scores(ex,{k:v["binary"] for k,v in d["micro"].items()},f"ESC · harness · {d['documents_scored']} docs",s,"fraction",d["documents_scored"])
            ex["diagnostics"].append({"label":"Couverture du harness", "data":{k:d[k] for k in ["documents_scored","documents_missing","k","budget"]},"source":s})
        elif marker == "ROUND4_RESULT":
            context = "ESC · DEV44 · développement réutilisé"
            for key in ["control_fusion", "new_fusion", "control_luna_binary_majority", "new_luna_binary_majority"]: add(ex,key,d[key],context,s,docs=44)
            primary = d["primary_paired_topic"]
            row = ex["rows"][-3]
            row.update(delta=primary["delta"],ci=primary["ci95"],ci_level=95,resampling="topic",reference="control_fusion")
            ex["diagnostics"].append({"label":"IC et multiplicité", "data":{k:d[k] for k in ["primary_paired_document","primary_paired_topic","secondary_luna_delta"]},"source":s})
        elif marker == "JUDGE_RESULT":
            if n in range(33,39):
                map_scores(ex,d["metrics"],f"ESC · DEV{len(d['documents'])} · round K",s,docs=len(d["documents"]))
            elif n == 27:
                for key in ["frozen_fusion","primary_w_lto_fixed_0.39","secondary_w1_fixed_0.39","secondary_joint_lto"]: add(ex,key,d[key],"ESC · DEV44 · juge J2",s,docs=44)
            elif n == 28:
                ex["diagnostics"].append({"label":"Précision et rappel de coréférence", "data":d["chains"],"source":s})
                if isinstance(d.get("causal_evaluation"),dict):
                    map_scores(ex,d["causal_evaluation"],"ESC · DEV44 · causalité",s,docs=44)
        elif marker == "OFFLINE_FUSION_RESULT":
            for key in ["primary_vs_frozen", "candidate_vs_threshold_only", "threshold_only_vs_frozen"]:
                comparison = d[key]
                for role in ["candidate","control"]: add(ex,key+" / "+role,comparison.get(role),"ESC · DEV44 · "+key,s,docs=44,note="Calibration supervisée hors topic.")
            ex["diagnostics"].append({"label":"Comparaisons de calibration", "data":{k:d[k] for k in ["primary_vs_frozen","average_precision","inference_limit"]},"source":s})
        elif marker in ["M1_RESULT","N1_RESULT"]:
            add(ex,"Référence",d["baseline"],"ESC · DEV20 · ajouts",s,"fraction",20)
            map_scores(ex,d.get("systems",{}),"ESC · DEV20 · ajouts",s,"fraction",20)
            ex["diagnostics"].append({"label":"Contraste H − C et ressources", "data":{k:d[k] for k in ["delta_f1","bootstrap","resources","costs","limitations"] if k in d},"source":s})
        elif marker == "M0_RESULT" and "arms" in d:
            map_scores(ex,d["arms"],"ESC · 2 documents · M0",s,"fraction",2)
            ex["diagnostics"].append({"label":"Coût M0 par bras", "data":d.get("cost_by_arm"),"source":s})
        elif marker in ["N3_RESULT","N7_RESULT","N9_RESULT","N12_RESULT"]:
            for schedule, coll in d["systems"].items():
                if isinstance(coll,dict) and any(k in coll for k in ["F1","tp"]):
                    add(ex,schedule,coll,f"ESC · {d.get('documents','?')} docs",s)
                else: map_scores(ex,coll,f"ESC · {d.get('documents','?')} docs · schedule {schedule}",s,docs=d.get("documents"))
            for key in ["primary_mean_delta_points","topic_bootstrap_CI95_points","contrasts","delta_F1","topic_bootstrap_CI95","token_usage"]:
                if key in d: ex["diagnostics"].append({"label":key,"data":d[key],"source":s})
        elif marker == "N8_RESULT":
            map_scores(ex,d["metrics"]["systems"],"ESC · DEV8 · lectures fraîches",s,docs=8)
            ex["diagnostics"].append({"label":"Comparaison appariée 3 − 7", "data":{k:v for k,v in d["metrics"].items() if k not in ["systems","per_topic_counts"]},"source":s})
        elif marker == "N10_RESULT":
            for schedule, score in d["systems"].items():
                add(ex,schedule+" / 6 lectures",score["full6"],"ESC · N7 · cache",s)
                add(ex,schedule+" / moyenne des 20 triplets",{"f1":score["mean20three_subset_F1_points"]},"ESC · N7 · cache",s,note="Moyenne des micro-F1 des sous-ensembles ; pas F1 des comptes moyens.")
            ex["diagnostics"].append({"label":"Contrastes préspécifiés", "data":d["primary_contrasts"],"source":s})
        elif marker == "N5_RESULT":
            for size, score in d["results"].items():
                add(ex,size+" lecture(s) · moyenne des sous-ensembles",{"f1":score["mean_F1"]},"ESC · CONF40 historique · cache",s,docs=40,note="Moyenne des F1 ; plage des sous-ensembles descriptive, pas IC.")
            ex["diagnostics"].append({"label":"Ablation exhaustive et dispersion", "data":d["results"],"source":s})
        elif marker == "N15_RESULT":
            map_scores(ex,d["systems"],"CTB6 · diagnostic",s,docs=6)
            ex["diagnostics"].append({"label":"Ressources N15", "data":d["resources"],"source":s})
        elif marker == "RESEARCH_CACHED_RESULT":
            map_scores(ex,d["systems"],f"CTB · {len(d['documents'])} docs · cache diagnostique",s,"fraction",len(d["documents"]))
        elif marker in ["N11_RESULT","M0_OFFLINE_SUMMARY","AGENT4_FINAL","N4_RESULT","N6_RESULT","N13_RESULT","N14_RESULT","CTB_ATTRIBUTION_RESULT","ALIAS_REAL_SMOKE_RESULT","N2_FINAL"]:
            # Only aggregate fields, never prompts, raw predictions or per-doc text.
            keep = ["cohorts","validity","integrity","valid","expected","status","coverage","actual_calls","planned_calls","invalid_calls","resources","decision","attrition","provider_gate","new_model_calls","candidate_pairs","planned_requests","model_task_outputs","actual_api_generation_requests","cost","baseline_global","results_by_arm"]
            data = {k:d[k] for k in keep if k in d}
            ex["diagnostics"].append({"label":marker,"data":data,"source":s})
            if marker=="ALIAS_REAL_SMOKE_RESULT" and r["data"]["results_by_arm"]["A"]["TP"]:
                map_scores(ex,d["results_by_arm"],"ESC · smoke alias · 1 document · réponses valides",s,"fraction",1)
            if marker=="CTB_ATTRIBUTION_RESULT":
                for split, block in d["splits"].items():
                    for variant, value in block["variants"].items():
                        add(ex,variant,value["sample"],"CTB · attribution · "+split,s,"fraction",note="Échantillon de paires ciblées ; pas le score du corpus entier.")
            if marker=="N11_RESULT":
                for cohort,value in d["cohorts"].items():
                    ex["observations"].extend([
                        {"label":cohort+" · appels économisés","value":100*value["call_saving_fraction"],"unit":"%","group":"Économies exactes","source":s,"note":f"{value['calls_saved']}/{value['historical_calls']} appels ; zéro décision changée."},
                        {"label":cohort+" · tokens économisés","value":100*value["historical_token_saving_fraction"],"unit":"%","group":"Économies exactes","source":s,"note":"Tokens historiques ; mêmes décisions."}])
            if marker=="N13_RESULT":
                ex["observations"].append({"label":f"Run {r['run_id'][:8]} · {d.get('stage','smoke')}","value":d["valid"],"total":d["expected"],"unit":"réponses","group":"Validité des smokes","source":s,"note":"Réponses valides / attendues ; aucune mesure primaire de F1."})

    histpath = "jev-v1/bilan-complet-experiences/analysis.json"
    hist = load(root/histpath)
    for n, smoke in enumerate(hist["smokes"],1):
        add(by_n[n],smoke["name"],smoke["binary"],"ESC · smoke 41_9 · 1 document",source(histpath,"smokes/"+str(n-1),smoke["run_id"],smoke["evidence_line"]),"fraction",1)
        by_n[n]["diagnostics"].append({"label":"Appels et temps mesurés", "data":smoke["resources"],"source":source(histpath,"smokes/"+str(n-1)+"/resources")})
    for entry in hist["r1"]:
        n = {"R1A":10,"R1B":11,"R1C":9,"R1E":12}.get(entry["variant"])
        if not n: continue
        ex=by_n[n];s=source(histpath,"r1/"+entry["variant"],entry["run_id"])
        c=entry["comparison"]
        add(ex,"R0 référence",{"f1":c["baseline_binary_F1"]},"ESC · DEVSCREEN10",s,"fraction",10)
        add(ex,entry["variant"],entry["binary"],"ESC · DEVSCREEN10",s,"fraction",10)
        ex["rows"][-1].update(delta=c["delta_binary_F1_pp"],ci=c["paired_document_bootstrap"]["delta_binary_F1_CI95_pp"],ci_level=95,resampling="document",reference="R0 référence")
        ex["diagnostics"].append({"label":"Coût et ressources", "data":entry["resources"],"source":s})
    for n in [14,15]:
        for name, v in hist["r2"].items():
            score={**v["binary"],**v["counts"]}
            add(by_n[n],name,score,"ESC · DEV25 · R2",source(histpath,"r2/"+name),"fraction",25,oracle=name.startswith("oracle"))
            q=v["paired_minus_baseline"]
            by_n[n]["rows"][-1].update(delta=q["delta_f1_pp"],ci=q["CI95_pp"],ci_level=95,resampling=q["bootstrap_unit"],reference="baseline_t0.5")

    auditpath="qwen-mecanismes-raisonnement-2026-11-05/data/audit-scores-2026-11-06.json"
    audit=load(root/auditpath)
    for n,prefix in [(74,"E1/"),(75,"E2/"),(76,"E3/"),(78,"E5/")]:
        for name,score in audit["systems"].items():
            if name.startswith(prefix): add(by_n[n],name,score,"CTB47 · replay strict · réponses finales",source(auditpath,"systems/"+name),docs=47,note="Score corrigé ; fragments JSON de réflexion exclus.")
    for n,keys in [(79,["Qwen7","R1/vote5"]),(82,["Qwen7","N1/vote5","R1+N1/vote10"]),(83,["R1/vote5","N2/verify_b8k"]),(84,["N1/vote5","P1/verify_b8k"])]:
        for key in keys: add(by_n[n],key,audit["systems"][key],"CTB47 · développement réutilisé",source(auditpath,"systems/"+key),docs=47)
        by_n[n]["notes"].append("Les valeurs suivent le recomptage strict structuré, y compris P1 = 56,49 ; les anciens résumés peuvent différer.")
    by_n[84]["outcome"]="Le replay strict donne F1 56,49 (TP74/FP48/FN66), soit −3,21 points contre N1 59,70. Le score historique 56,06 incluait deux faux positifs supplémentaires issus du texte complet. Aucun gain confirmé ; l’IC historique n’est pas transposé au score corrigé."
    by_n[104]['outcome']=by_n[104]['outcome'].replace('un seul faux lien','deux faux liens non ordonnés')
    by_n[84]["sources"].append(auditpath)
    p1row=next(r for r in by_n[84]["rows"] if r["variant"]=="P1/verify_b8k")
    p1row.update(delta=audit["systems"]["P1/verify_b8k"]["f1"]-audit["systems"]["N1/vote5"]["f1"],reference="N1/vote5")

    # Fallbacks are reviewed published aggregate scores (never extracted by a
    # loose number regex). They carry the catalogue section as exact provenance.
    fallback={
      5:("ESC · D15 final · diagnostic",[("Mercury",32.39)]),
      7:("ESC · DEV25",[("R0 d1",39.2628205128)]),
      8:("ESC · DEV25",[("d1 R0",39.26),("Luna",50.85)]),
      16:("ESC · CONF40 · historique",[("d1 initial",34.79),("d1 symétrique seuil 0,65",28.29)]),
      17:("ESC · CONF40 · historique",[("d1 initial",34.79),("Luna",36.44)]),
      18:("ESC · TEST-S40 · fusion gelée",[("d1",38.59),("Luna",46.02),("Fusion Luna × d1",48.64)]),
      19:("ESC · TEST-S40 · majorité binaire",[("Luna référence",46.02),("Rationales",44.34)]),
      20:("ESC · TEST-S40 · diagnostic",[("Fusion référence",48.64),("Fusion + groupes",48.88)]),
      29:("ESC · RES30 · gold symétrisé corrigé",[("Fusion référence",36.07),("J3-A corrigé",36.35)]),
      30:("ESC · RES30 · gold symétrisé corrigé",[("Fusion référence",36.07),("J3-B corrigé",35.99)]),
      80:("ESC40 · R2 · 10k",[("Vote Qwen 3",41.81)]),
      81:("CTB47 · R3 · une lecture terminée",[("Première lecture purpose/enablement",29.46)]),
      85:("CTB47 · P3 · 20k",[("Vote P3 5",57.14),("N1 + P3",61.1940298507)]),
      86:("ESC40 · P2 · 24k",[("Vote P2 3",42.3817863398)]),
      87:("CTB47 · vérification P3",[("P3",57.14),("P4",57.14)]),
    }
    # These fallbacks preserve published aggregate precision and do not pretend
    # that unavailable underlying counts are known.
    for n,(context, values) in fallback.items():
        if by_n[n]["rows"]: continue
        for label,value in values:
            add(by_n[n],label,{"f1":value},context,{"file":"docs/catalogue-experiences.md","pointer":f"fiche-{n:02d}","run":by_n[n]["runs"][0]["id"] if by_n[n]["runs"] else None,"line":None},note="Valeur agrégée du catalogue revu ; comptes non disponibles dans cette table.")
    # Final vote counts reread in the persisted R2/P2 logs. The campaign has
    # three passes (40 documents x 3), not five as an old export label implied.
    for n,tp,fp,fn,line in [(80,704,722,1238,715),(86,726,758,1216,797)]:
        row=by_n[n]["rows"][0]
        row.update(tp=tp,fp=fp,fn=fn,f1=200*tp/(2*tp+fp+fn),
                   precision=100*tp/(tp+fp),recall=100*tp/(tp+fn),documents=40,
                   source={"file":"journaux/"+by_n[n]["runs"][0]["id"]+".log",
                           "pointer":"final-vote/TP-FP-FN","run":by_n[n]["runs"][0]["id"],"line":line},
                   note="Trois lectures prévues par article ; comptes finaux relus dans le journal, pas les scores des lectures individuelles.")
    for n,label,delta,ci,reference in [(8,"Luna",11.59,[7.51,15.65],"d1 R0"),(18,"Fusion Luna × d1",10.05,[6.66,13.65],"d1"),(19,"Rationales",-1.67,[-3.85,.85],"Luna référence"),(20,"Fusion + groupes",.24,[-1.25,1.58],"Fusion référence")]:
        row=next(r for r in by_n[n]["rows"] if r["variant"]==label)
        row.update(delta=delta,ci=ci,ci_level=95,reference=reference,resampling="voir protocole source")
    s={"file":"docs/catalogue-experiences.md","pointer":"fiche-77","run":by_n[77]["runs"][0]["id"],"line":None}
    for context, values,docs in [("CTB47 · avant réflexion",[("Probe supervisé",34.88),("Qwen7",58.91)],47),("CTB38 · après réflexion · intersection valide",[("Probe supervisé",51.61),("Qwen7",44.44)],38)]:
        for label,value in values: add(by_n[77],label,{"f1":value},context,s,docs=docs,note="Probe supervisé ; cohorte après réflexion limitée aux 38 documents valides.")
    for n,labels,values in [(70,["demo 0,3","demo 0,6","demo 1,0"],[-5.1,5.4,-4.8]),(71,["hint 0,2","hint opposé","direction aléatoire"],[-3.7,-6.3,-2.1])]:
        # Labels are checked against the detailed source below by the validation
        # report; each contrast has its own valid-document intersection.
        for label,value in zip(labels,values):
            by_n[n]["observations"].append({"label":label,"value":value,"unit":"points de F1","group":"Intersections valides propres à chaque contraste","source":{"file":"docs/catalogue-experiences.md","pointer":f"fiche-{n:02d}"},"note":"Contrastes historiques sur documents communs ; intersections différentes, pas de classement entre variantes. Intervalles numériques non intégrés à cette table."})
    for n in [113,114,115]:
        ex=by_n[n];d=ex["diagnostics"][0]
        for arm, cost in d["data"]["readings"].items():
            total={113:235,114:20,115:20}[n]
            ex["observations"].append({"label":arm+" · lectures valides","value":total-cost["invalid"],"total":total,"unit":"réponses","group":"Validité des nouvelles lectures","source":d["source"],"note":"Le nombre de réponses invalides est lu dans le résumé final du journal."})
    # Import the complete single-reader W1 screen (including both edit doses),
    # and preserve W2's multiplicity-aware comparisons separately from its F1.
    wp="qwen-mecanismes-raisonnement-2026-11-05/weight-space-programming-2026-11-10/data/"
    for n,rel in [(110,"w1-screen/screen_scores.json"),(111,"w2-confirmation/logged_scores.json")]:
        raw=load(root/(wp+rel));ex=by_n[n];s=source(wp+rel,"conditions")
        cached={r["variant"]:r for r in ex["rows"]};ex["rows"]=[]
        for label,score in raw["conditions"].items():
            add(ex,label,score,"W1 · CTB20 · screen" if n==110 else "W2 · CTB20 · réserve",{**s,"pointer":"conditions/"+label},"fraction",20)
            row=ex["rows"][-1]
            if label in cached:row["extras"].update(cached[label]["extras"])
            for k in ["mean_generated_tokens","mean_thinking_tokens","forced_closures","invalid_answers"]:
                if k in score:row["extras"][k]=score[k]
            if "vs_base_f1_pp" in score:
                row.update(delta=score["vs_base_f1_pp"][0],ci=score["vs_base_f1_pp"][1:],ci_level=95,resampling="document",reference="base")
        if n==111:
            contrast=raw["selected_minus_comparators"]["base"]
            row=next(r for r in ex["rows"] if r["variant"]=="cc_0.10")
            row.update(delta=contrast["delta_f1_pp"],ci=contrast["ci95_pp"],ci_level=95,resampling="document",reference="base")
            ex["diagnostics"].append({"label":"Comparaisons corrigées pour la multiplicité (IC99 Bonferroni)","data":{k:raw[k] for k in ["selected_minus_comparators","bootstrap","decision","caveat"]},"source":s})
    # Full-corpus H1 output is a final structured log record.
    for r in records:
        if r["marker"]=="FUSION_RESULT" and by_id[r["experiment_id"]]["number"]==39:
            for corpus, block in r["data"].get("corpora",{}).items():
                for name, score in block.items():
                    if isinstance(score,dict) and name != "historical_unpaired_reference": add(by_n[39],name,score,corpus.upper()+" · H1 corpus complet",{"file":"journaux/"+r["run_id"]+".log","pointer":"FUSION_RESULT/corpora/"+corpus+"/"+name,"run":r["run_id"],"line":r["line"]},docs=block["documents"])

    # Deduplicate retransmitted printouts only within the same experiment/context.
    for ex in experiments:
        seen={}
        for row in ex["rows"]:
            key=(row["context"],row["variant"])
            seen[key]=row
        ex["rows"]=list(seen.values())
        ex["rows"].sort(key=lambda row:(row["context"],row["oracle"],row["variant"]))
        for i,row in enumerate(ex["rows"]): row["id"]=f"{ex['number']}-{i}"
        ex["contexts"]=list(dict.fromkeys(r["context"] for r in ex["rows"]))
        if not ex["rows"]: ex["notes"].append("Aucune série de scores consolidée dans l’export numérique. Le résultat et les limites disponibles sont conservés dans la fiche et le détail technique ; une absence de mesure n’est pas un score nul.")
    data={"schema":"workere-results-web/v1", "project_id":inventory["project_id"], "snapshot":inventory["verified_utc"],
          "families":[dict(id=a,title=b,description=c) for a,b,c in FAMILIES], "experiments":experiments,
          "counts":{"experiments":len(experiments),"runs":len(runs),"logs":logindex["logs_available"],"score_rows":sum(len(e["rows"]) for e in experiments)},
          "run_statuses":{s:sum(r["status"]==s for r in runs.values()) for s in sorted(set(r["status"] for r in runs.values()))},
          "reports":{"synthese":(HERE.parent/"docs/synthese-resultats.md").read_text(),"corrections":(HERE.parent/"docs/corrections-et-provenance.md").read_text()},
          "provenance":{"source_snapshot":TOPIC,"source_sha256":source_files,"verified_log_records":verified_records,"rules":["Scores en pourcentage ; conversion explicite des fractions.","Comptes TP/FP/FN conservés et F1 vérifié à 0,016 point près (arrondi source).","J3 brut exclu ; replay strict E1/E2/E3/E5 prioritaire.","Les sorties WSP partielles sont exclues ; W5 et W6 ont chacun quatre documents.","Pas de courbe d’apprentissage, d’IC ou de coût total inventé.","Instantané documentaire ; aucune nouvelle inférence."]}}
    from prepare_prompts import enrich
    enrich(data)
    (HERE/"data.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({**data["counts"],"experiments_with_scores":sum(bool(e["rows"]) for e in experiments),"verified_log_records":verified_records},ensure_ascii=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifacts",type=Path,help="Original project artifacts root (read only)")
    import_snapshot(p.parse_args().artifacts)
