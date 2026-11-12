/* Self-contained research explorer. No fetch, analytics, CDN or build runtime. */
"use strict";
const DATA = JSON.parse(document.getElementById("research-data").textContent);
const EXP = DATA.experiments;
const BY_ID = new Map(EXP.map(e => [e.id, e]));
const STATES = {mesure:"Résultats mesurés",diagnostic:"Diagnostic / technique",incomplet:"Partiel / validité bloquée",planifie:"Planifié",en_cours:"En cours à l’instantané"};
const RUN_STATES = {done:"Terminé",failed:"Échec",cancelled:"Annulé","cancelled-not-confirmed":"Annulation non confirmée",lost:"Suivi perdu",running:"En cours",cancelling:"Annulation en cours"};
const main = document.getElementById("contenu");
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const norm = value => String(value).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
const fmt = (v,d=2) => v == null || !Number.isFinite(v) ? "—" : new Intl.NumberFormat("fr-FR", {maximumFractionDigits:d,minimumFractionDigits:d}).format(v);
const integer = v => v == null ? "—" : new Intl.NumberFormat("fr-FR",{maximumFractionDigits:0}).format(v);
const sign = v => v == null ? "—" : (v>0?"+":"")+fmt(v);
const chip = e => `<span class="chip ${esc(e.state)}">${esc(STATES[e.state])}</span>`;
const familyName = id => DATA.families.find(f=>f.id===id)?.title || id;
let route = {}, visibleRows = [], visibleSVG = "", activeExperiment = null, lastFocus = null;
let present = false;
let rankedMethods = [];

function parseRoute() {
  const hash = location.hash.slice(1) || "/";
  const [path, query] = hash.split("?");
  return {path, params:new URLSearchParams(query || "")};
}
function url(path, params=new URLSearchParams()) { const q=params.toString(); return "#"+path+(q?"?"+q:""); }
function changed(params, changes) {
  const p=new URLSearchParams(params);
  for(const [k,v] of Object.entries(changes)) { if(v==null || v==="") p.delete(k); else p.set(k,String(v)); }
  return p;
}
function filterParams(params=route.params) {
  const p=new URLSearchParams();
  for(const k of ["q","famille","corpus","statut","scores","ordre"]) if(params.get(k)) p.set(k,params.get(k));
  return p;
}
function linkExperiment(e, params=filterParams()) { return url("/experience/"+e.id, params); }
function expLink(n, label) { const e=EXP.find(x=>x.number===n); return `<a href="${esc(linkExperiment(e,new URLSearchParams()))}">${esc(label||e.title)} ↗</a>`; }
function options(items, selected) { return items.map(([value,label])=>`<option value="${esc(value)}" ${value===selected?"selected":""}>${esc(label)}</option>`).join(""); }
function selectedExperiments(params=route.params) {
  let list=EXP.filter(e=>(!params.get("famille") || e.family===params.get("famille")) && (!params.get("corpus") || e.datasets.includes(params.get("corpus"))) && (!params.get("statut") || e.state===params.get("statut")) && (!params.get("scores") || (params.get("scores")==="oui"?e.rows.length:!e.rows.length)));
  const q=norm(params.get("q")||"").trim();
  if(q) list=list.filter(e=>norm([e.number,e.title,e.original_title,e.id,e.method,e.evaluation,e.outcome,e.diagram.intervention,...e.rows.map(r=>r.variant)].join(" ")).includes(q));
  if(params.get("ordre")==="recent") list.reverse();
  if(params.get("ordre")==="titre") list.sort((a,b)=>a.title.localeCompare(b.title,"fr"));
  return list;
}
function title(eyebrow, heading, subtitle, actions="") {
  return `<div class="page-head"><div><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(heading)}</h1><p>${esc(subtitle)}</p></div>${actions}</div>`;
}
function listRows(list) {
  if(!list.length) return `<div class="empty">Aucune expérience ne correspond à ces filtres.<br><a href="#/experiences">Réinitialiser les filtres</a></div>`;
  return list.map(e=>`<a class="exp-row" href="${esc(linkExperiment(e))}"><span class="exp-number">${String(e.number).padStart(3,"0")}</span><div><h3>${esc(e.title)}</h3><p>${esc(e.method)}</p></div><div class="exp-meta"><div class="chips">${e.datasets.map(d=>`<span class="chip">${esc(d)}</span>`).join("")}${chip(e)}</div><span>${e.rows.length?integer(e.rows.length)+" lignes de résultats":"Fiche & suivi disponibles"}</span></div><span class="arrow" aria-hidden="true">↗</span></a>`).join("");
}

function home() {
  const scoreExperiments=EXP.filter(e=>e.rows.length).length;
  const e18=EXP.find(e=>e.number===18);
  const fusions=e18.rows;
  const homeBars=fusions.map((r,i)=>{
    const y=30+i*45, color=/Fusion/.test(r.variant)?"#007f76":"#a4b3a3";
    return `<g><text x="0" y="${y+11}" fill="#4d6256" font-size="11">${esc(r.variant)}</text><rect x="110" y="${y}" width="${r.f1*3.3}" height="16" rx="3" fill="${color}"/><text x="${120+r.f1*3.3}" y="${y+12}" fill="#294539" font-size="12" font-weight="600">${fmt(r.f1)}</text></g>`;
  }).join("");
  return `<section class="hero"><div><div class="eyebrow">117 expériences · une lecture d’ensemble</div><h1>Comprendre ce qui<br><em>fait progresser</em><br>la détection causale.</h1><p>Des premiers lecteurs aux interventions sur les poids : les méthodes testées, leurs résultats et les limites de chaque découverte, réunis dans un atlas consultable hors ligne.</p><div class="button-row"><a class="button primary" href="#/experiences">Explorer les expériences <span aria-hidden="true">↗</span></a><a class="button" href="#/documents">Lire la synthèse</a></div></div><div class="hero-right"><div class="eyebrow">UN RÉSULTAT DE RÉFÉRENCE</div><h2 class="chart-title">La complémentarité des lecteurs</h2><span class="small">ESC · TEST-S40 · micro-F1 binaire (%)</span><svg viewBox="0 0 355 192" role="img" aria-label="Sur ESC TEST-S40 : d1 38,59 ; Luna 46,02 ; fusion 48,64. Comparaison sur le même protocole.">${homeBars}<line x1="110" y1="164" x2="308" y2="164" stroke="#b8c7b7"/><text x="110" y="181" font-size="9" fill="#617567">0</text><text x="298" y="181" font-size="9" fill="#617567">60</text></svg><p class="chart-note">Fusion − d1 : +10,05 points, IC95 [6,66 ; 13,65].<br>Fusion − Luna : +2,62 points, IC95 [−1,37 ; 7,69].</p>${expLink(18,"Voir le protocole et les sources")}</div></section>
  <section class="stats" aria-label="Couverture de l’instantané"><div class="stat"><strong>117</strong><span>expériences documentées</span></div><div class="stat"><strong>190</strong><span>exécutions, reprises comprises</span></div><div class="stat"><strong>${scoreExperiments}</strong><span>fiches avec séries chiffrées</span></div><div class="stat"><strong>10</strong><span>familles de méthodes</span></div></section>
  <div class="section-top"><h2>Trois découvertes à retenir</h2><a href="#/documents">Parcours de présentation →</a></div>
  <section class="insights"><article class="card"><div class="insight-index">01<span>COMPLÉMENTARITÉ</span></div><h3>Deux signaux peuvent mieux se compléter.</h3><p>La fusion historique gagne sur d1 dans TEST-S40. Son avantage sur Luna reste incertain. Le corpus et le protocole donnent son sens au gain.</p>${expLink(18,"Explorer la fusion")}</article><article class="card"><div class="insight-index">02<span>BUDGET & VALIDITÉ</span></div><h3>Une sortie complète change la lecture des scores.</h3><p>Le replay strict E1 corrige les scores lorsque la réflexion était tronquée. La fermeture à 8k atteint F1 50,00 sur CTB47 ; l’amélioration n’est pas confirmée.</p>${expLink(74,"Comprendre le replay strict")}</article><article class="card"><div class="insight-index">03<span>CONTRÔLES</span></div><h3>Le contrôle aléatoire compte autant que le gain.</h3><p>W2 ne confirme pas un gain propre à la direction éditée. Les pilotes de transfert W5 et W6 portent chacun sur quatre documents, avec une confusion de validité dans W6.</p>${expLink(111,"Examiner la confirmation réservée")}</article></section>
  <div class="section-top"><h2>Parcourir la recherche</h2><a href="#/methodes">Les 10 familles →</a></div>
  <section class="home-path">${[["lecteurs","01 / Références","Choisir les lecteurs"],["fusion","02 / Combinaisons","Fusionner les signaux"],["budget","03 / Ressources","Mesurer le budget"],["juges","04 / Vérification","Réparer les décisions"],["poids","05 / Interventions","Modifier le modèle"]].map(([id,n,t])=>`<a href="${url('/experiences',new URLSearchParams({famille:id}))}"><small>${n}</small>${t}<b>→</b></a>`).join("")}</section>
  <div class="section-top"><h2>Comment lire les résultats</h2></div><div class="panel"><div class="protocol-grid"><div><h3>La tâche testée</h3><p class="small muted">Les mentions d’événements et les candidats sont fournis par les corpus. Les systèmes prédisent les relations causales entre ces mentions. Le micro-F1 binaire ne mesure pas une extraction d’événements de bout en bout.</p></div><div><h3>Un instantané traçable</h3><p class="small muted">Les 190 exécutions incluent échecs et reprises. ${DATA.counts.logs} journaux étaient disponibles lors de la revue. W7/W8 sont indiqués en cours au 12 novembre, sans résultat final anticipé. Aucune inférence n’est lancée par ce site.</p></div></div></div>`;
}

function catalogue() {
  const p=route.params,list=selectedExperiments();
  return title("Catalogue des expériences", "Explorer les 117 expériences", "Filtrer les méthodes, les corpus et l’état des preuves. Chaque fiche conserve son protocole et ses sources.", `<a class="button" href="#/methodes">Parcourir les méthodes →</a>`)+`
  <form id="filters"><div class="catalogue-toolbar"><label class="field search-field">Rechercher<input name="q" type="search" value="${esc(p.get('q')||'')}" placeholder="Nom, méthode, résultat, numéro…" autocomplete="off"></label><label class="field">Méthode<select name="famille">${options([["","Toutes les méthodes"],...DATA.families.map(f=>[f.id,f.title])],p.get('famille')||'')}</select></label><label class="field">Corpus<select name="corpus">${options([["","Tous les corpus"],["CTB","CTB"],["ESC","ESC"],["MAVEN","MAVEN"],["Technique","Technique"]],p.get('corpus')||'')}</select></label><label class="field">État de l’étude<select name="statut">${options([["","Tous les états"],...Object.entries(STATES)],p.get('statut')||'')}</select></label></div>
  <div class="catalogue-meta"><span id="catalogue-count">${list.length} expérience${list.length>1?'s':''} <span class="sep">/</span> 117 <a href="#/experiences" style="margin-left:12px">Réinitialiser</a></span><label class="field">Séries chiffrées<select name="scores">${options([["","Toutes les fiches"],["oui","Avec séries chiffrées"],["non","Sans série consolidée"]],p.get('scores')||'')}</select></label><label class="field">Trier<select name="ordre">${options([["","Ordre des fiches"],["recent","Ordre inverse"],["titre","Titre A → Z"]],p.get('ordre')||'')}</select></label></div></form><div id="catalogue-list" class="experiment-list">${listRows(list)}</div>`;
}
function methods() {
  return title("Panorama des méthodes", "Dix façons d’intervenir sur la décision", "Des changements de consigne aux modifications de poids : chaque famille rassemble les variantes, contrôles et évaluations associés.")+`<div class="family-grid">${DATA.families.map((f,i)=>{const list=EXP.filter(e=>e.family===f.id);return `<a class="family-card" href="${url('/experiences',new URLSearchParams({famille:f.id}))}"><span class="family-number">${String(i+1).padStart(2,'0')} / MÉTHODE</span><span class="family-count">${list.length} expériences ↗</span><h3>${esc(f.title)}</h3><p>${esc(f.description)}</p><p style="margin-top:17px;color:#007f76">${list.filter(e=>e.rows.length).length} fiches avec résultats chiffrés</p></a>`}).join('')}</div><div class="callout" style="margin-top:24px">Les expériences d’une même famille peuvent utiliser des corpus, budgets et ensembles d’évaluation différents. La navigation par méthode ne constitue pas un classement global de F1.</div>`;
}

function rankingCandidates(corpus) {
  const allowed={CTB:[74,75,76,77,78,79,81,82,83,84,85,87,88,89,102,109,113],ESC:[18,19,80,86,103,109],MAVEN:[104,109]}[corpus];
  const candidateFilters={74:r=>!/control/.test(r.variant),75:r=>/early_exit/.test(r.variant),76:r=>/refl_/.test(r.variant),77:r=>/Probe/.test(r.variant),78:r=>/heads_edit/.test(r.variant),79:r=>r.variant==='R1/vote5',82:r=>r.variant==='N1/vote5',83:r=>r.variant==='N2/verify_b8k',84:r=>r.variant==='P1/verify_b8k',87:r=>r.variant==='P4'};
  const generic=r=>!r.oracle && !/^(?:historical|Référence historique|A_vote|Qwen7|Qwen7 référence|P3|Luna référence|Fusion référence)$/.test(r.variant);
  const found=[];
  for(const n of allowed){
    const e=EXP[n-1];
    const rows=e.rows.filter(r=>{
      const scope=corpus==='CTB'?(/^CTB47|^CTB · 47/.test(r.context) && (!r.documents||r.documents===47)):corpus==='ESC'?(/^ESC40|^ESC · TEST-S40|^ESC · 40/.test(r.context) && (!r.documents||r.documents===40)):(/^MAVEN · 12/.test(r.context)&&r.documents===12);
      return scope && generic(r) && (!candidateFilters[n]||candidateFilters[n](r));
    }).sort((a,b)=>b.f1-a.f1||a.variant.localeCompare(b.variant));
    if(rows.length)found.push({experiment:e,row:rows[0],variants:rows.length});
  }
  return found.sort((a,b)=>b.row.f1-a.row.f1||a.experiment.number-b.experiment.number).slice(0,10);
}
function topTen() {
  const corpus=['CTB','ESC','MAVEN'].includes(route.params.get('corpus'))?route.params.get('corpus'):'CTB';
  rankedMethods=rankingCandidates(corpus);
  const label={CTB:'CTB47',ESC:'ESC40 / TEST-S40',MAVEN:'MAVEN12'}[corpus];
  const graph=plot(rankedMethods.map(x=>({...x.row,variant:x.row.variant+' · '+x.experiment.title})), 'f1',{title:'Top des méthodes · '+label,id:'top-'+corpus},label+' · présence binaire des liens');
  return title('Comparaison des méthodes', 'Top 10 des méthodes testées', 'Les meilleurs scores observés dans un même périmètre. Une entrée par méthode expérimentale, avec accès au protocole, au prompt et aux limites.')+`
  <div class="callout"><strong>Classement exploratoire par micro-F1.</strong> Le classement retient la meilleure variante disponible de chaque méthode éligible sur le panel sélectionné. Les budgets et la supervision peuvent différer ; la sélection de variantes sur un jeu déjà exposé ne démontre pas une supériorité en généralisation. Les petits pilotes W1/W2/W5/W6 et les résultats incomplets sont exclus de ces panels.</div>
  <section class="panel"><form id="ranking-controls"><label class="field" style="max-width:430px">Panel comparable<select name="corpus">${options([['CTB','CTB47 · 47 documents'],['ESC','ESC40 / TEST-S40 · 40 documents'],['MAVEN','MAVEN12 · 12 documents']],corpus)}</select></label></form><h2 class="chart-title" style="margin-top:23px">${rankedMethods.length} méthodes · ${label}</h2><p class="chart-note">Au plus dix méthodes : les panels ESC et MAVEN possèdent moins de dix méthodes éligibles dans l’export. Les variantes de référence recopiées et la continuation MV1 déjà reprise par MV2 sont exclues.</p><div class="chart-container">${graph}</div><button class="quiet" data-action="export-ranking">↓ Classement, variantes et sources JSON</button></section>
  <div class="ranking-grid">${rankedMethods.map(({experiment:e,row:r,variants},i)=>`<article class="panel rank-card"><div class="rank-head"><span class="rank-number">${String(i+1).padStart(2,'0')}</span><strong>${fmt(r.f1)}<small>F1 %</small></strong></div><h2>${esc(e.title)}</h2><p class="plain-summary">${esc(e.simple)}</p><p class="small muted">Variante retenue : <strong>${esc(r.variant)}</strong><br>${variants} variante${variants>1?'s':''} éligible${variants>1?'s':''} observée${variants>1?'s':''} pour cette méthode.${e.diagram.supervised?' Supervision / calibration ou démonstrations autorisées : voir le protocole.':''}</p><div class="button-row"><a class="button primary" href="${linkExperiment(e,new URLSearchParams())}">Résultats →</a><a class="button" href="${url('/experience/'+e.id,new URLSearchParams({onglet:'schema'}))}">Méthode</a><a class="button" href="${url('/experience/'+e.id,new URLSearchParams({onglet:'prompts'}))}">Prompt</a></div><p class="chart-note">${esc(e.checks)}</p></article>`).join('')}</div>`;
}

function textLines(text, limit=32, max=3) {
  const words=String(text).split(/\s+/), lines=[]; let line="";
  for(const word of words) { if((line+" "+word).trim().length>limit && line) {lines.push(line); line=word;} else line+=(line?" ":"")+word; }
  if(line) lines.push(line);
  if(lines.length>max) { lines.length=max; lines[max-1]=lines[max-1].slice(0,limit-1)+"…"; }
  return lines;
}
function node(x,y,w,h,label,sub,accent=false) {
  const ls=textLines(label,29,3),top=y+h/2-((ls.length-1)*15)/2-(sub?10:0);
  return `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${accent?'#e3f1e9':'#f7f8f3'}" stroke="${accent?'#007f76':'#cedacc'}" stroke-width="${accent?1.6:1}"/>${ls.map((l,i)=>`<text class="node-label" x="${x+w/2}" y="${top+i*15}" text-anchor="middle" font-size="12" font-weight="${accent?600:500}" fill="${accent?'#006b60':'#30493c'}">${esc(l)}</text>`).join('')}${sub?`<text class="node-label" x="${x+w/2}" y="${y+h-13}" text-anchor="middle" font-size="9" fill="#708073">${esc(sub)}</text>`:''}</g>`;
}
function diagram(e) {
  const kind=e.diagram.kind, cached=Boolean(e.prompt_upstream), planned=e.state==='planifie';
  const prep={lecteurs:'Repérer les deux mentions',consignes:'Préparer la présentation',fusion:'Charger les signaux à combiner',harness:'Construire les transformations',ancrage:'Repérer les preuves et les mentions',budget:'Fixer les lectures ou sous-ensembles',mecanismes:'Charger le modèle et le contrôle',votes:'Fixer les lectures et exemples',juges:'Construire paires ou graphes',poids:'Charger le modèle et la direction'}[kind];
  const aggregate={lecteurs:'Appliquer seuil ou vote',consignes:'Appliquer la règle de décision',fusion:'Combiner et appliquer le seuil',harness:'Choisir et agréger les vues',ancrage:'Admettre les ajouts vérifiés',budget:'Comparer les allocations',mecanismes:'Extraire les relations finales',votes:'Voter, filtrer ou réparer',juges:'Appliquer accord / réparation',poids:'Voter et vérifier les liens'}[kind];
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" class="pipeline" style="font-family:Arial,sans-serif" viewBox="0 0 1030 555" role="img" aria-labelledby="diagram-title diagram-desc"><title id="diagram-title">${esc(e.title)} — méthode ${planned?'prévue':'testée'}</title><desc id="diagram-desc">${esc(e.simple)} ${esc(e.method)}. Le gold de la cible n’entre que dans l’évaluation. Schéma explicatif de l’implémentation documentée.</desc><defs><marker id="arrow-tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#8ba08e"/></marker></defs><rect width="1030" height="555" fill="white" rx="10"/>`;
  const edge=(x1,y1,x2,y2,dash=false)=>`<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="#8ba08e" stroke-width="1.4" ${dash?'stroke-dasharray="5 5"':''} marker-end="url(#arrow-tip)"/>`;
  svg+=`<text x="16" y="34" font-size="12" fill="#597260">01 / Construire la décision</text>`;
  svg+=node(550,14,260,68,kind==='poids'?'Édition / contrôle aléatoire':e.diagram.supervised?'Exemples / calibration autorisés':'Configuration de la condition','Paramètres propres à l’essai');
  svg+=edge(680,82,680,120,true);
  svg+=node(15,125,210,96,cached?'Sorties et données enregistrées':'Document + mentions candidates',e.datasets.join(' / '));
  svg+=edge(225,173,265,173)+node(266,125,244,96,prep,'Périmètre décrit dans le protocole');
  svg+=edge(510,173,545,173)+node(546,125,266,96,e.diagram.intervention,'Changement testé · fiche '+e.number,true);
  svg+=edge(812,173,845,173)+node(846,125,169,96,cached?'Signaux disponibles':'Réponses du modèle',cached?'Pas de nouvelle génération':'IDs et liens proposés');
  svg+=`<path d="M 930 221 L 930 252 L 120 252 L 120 340" fill="none" stroke="#8ba08e" stroke-width="1.4" marker-end="url(#arrow-tip)"/>`;
  svg+=`<text x="16" y="305" font-size="12" fill="#597260">02 / Contrôler et évaluer</text>`;
  svg+=node(15,345,210,96,'Vérifier format et identifiants','Traitement des invalides selon contrat');
  svg+=edge(225,393,265,393)+node(266,345,244,96,aggregate,'Règle propre à chaque condition');
  svg+=edge(510,393,545,393)+node(546,345,266,96,planned?'Évaluation prévue':'Comparer aux annotations','Même ensemble évalué pour le contraste');
  svg+=edge(812,393,845,393)+node(846,345,169,96,e.number===61?'Appels / tokens économisés':e.rows.length?'TP / FP / FN → F1':'Couverture / validité',planned?'Aucun résultat attribué':'Mesures réellement disponibles');
  svg+=node(550,467,260,57,'Gold de la cible','Utilisé uniquement pour l’évaluation');
  svg+=edge(680,467,680,446,true);
  const lines=textLines(e.evaluation,72,3);lines.forEach((l,i)=>svg+=`<text x="16" y="${478+i*16}" font-size="10" fill="#697e70">${esc(l)}</text>`);
  svg+=`<rect x="850" y="29" width="9" height="9" fill="#007f76" rx="2"/><text x="866" y="37" font-size="10" fill="#587361">Changement testé</text></svg>`;
  return svg;
}

function currentRows(e) {
  const p=route.params;
  let context=p.get("cohorte");
  if(!e.contexts.includes(context)) context=e.contexts[0];
  let rows=e.rows.filter(r=>r.context===context);
  const oracle=p.get("oracle")==="oui";
  rows=rows.filter(r=>r.oracle===oracle);
  const q=norm(p.get("variante")||""); if(q) rows=rows.filter(r=>norm(r.variant).includes(q));
  if(p.get("tri")==="f1") rows.sort((a,b)=>b.f1-a.f1);
  return {rows,context,oracle};
}
function valueFor(row,metric) {
  if(metric==="tokens") return row.extras.emitted_tokens_total ?? row.extras.cost?.think_tokens ?? row.extras.incremental_cost?.think_tokens ?? null;
  return row[metric];
}
const METRIC_LABELS={f1:"Micro-F1 (%)",precision:"Précision (%)",recall:"Rappel (%)",delta:"ΔF1 (points)",tokens:"Tokens émis / réflexion (voir source)"};
function plot(rows,metric,e,context) {
  const actual=rows.filter(r=>typeof valueFor(r,metric)==="number" && Number.isFinite(valueFor(r,metric)));
  if(!actual.length) return `<div class="empty">Cette mesure n’est pas disponible pour les variantes affichées. Consulter les diagnostics ou choisir une autre mesure.</div>`;
  const labelWidth=310, chartWidth=510, width=960, step=38, top=58, height=top+actual.length*step+48;
  let min=metric==="delta"?Math.min(0,...actual.flatMap(r=>r.ci?[r.ci[0],r.delta]:[r.delta])):0;
  let max=metric==="delta"?Math.max(0,...actual.flatMap(r=>r.ci?[r.ci[1],r.delta]:[r.delta])):metric==="tokens"?Math.max(...actual.map(r=>valueFor(r,metric))):100;
  if(max===min) {min-=1;max+=1;}
  if(metric==="delta") {const pad=(max-min)*.08;min-=pad;max+=pad;}
  const x=v=>labelWidth+(v-min)/(max-min)*chartWidth;
  let s=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="chart-svg-title chart-svg-desc"><title id="chart-svg-title">${esc(e.title)} — ${esc(METRIC_LABELS[metric])}</title><desc id="chart-svg-desc">${esc(context)}. ${actual.length} variantes. Les valeurs exactes et leurs sources figurent dans le tableau associé. Cliquer une marque pour sa provenance.</desc><rect width="${width}" height="${height}" fill="white"/><text x="0" y="17" font-family="sans-serif" font-size="12" fill="#182c36">${esc(textLines(context,120,1)[0])}</text><text x="${labelWidth}" y="40" font-family="sans-serif" font-size="10" fill="#617567">${esc(METRIC_LABELS[metric])}</text>`;
  for(let i=0;i<=4;i++) {const v=min+(max-min)*i/4,at=x(v);s+=`<line x1="${at}" y1="48" x2="${at}" y2="${height-35}" stroke="#e3e8df"/><text x="${at}" y="${height-15}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#718171">${metric==='tokens'?integer(v):fmt(v,metric==='delta'?1:0)}</text>`;}
  if(metric==="delta") s+=`<line x1="${x(0)}" y1="48" x2="${x(0)}" y2="${height-35}" stroke="#879a8a" stroke-dasharray="4 4"/>`;
  actual.forEach((r,i)=>{
    const y=top+i*step+14,v=valueFor(r,metric),color=r.oracle?"#7861a5":/baseline|référence|control|base5$|^base$|historical/i.test(r.variant)?"#a0aea5":"#007f76";
    const short=textLines(r.variant,43,1)[0];
    s+=`<g class="chart-point" tabindex="0" role="button" data-evidence="${r.id}" aria-label="${esc(r.variant)}, ${esc(METRIC_LABELS[metric])} ${esc(fmt(v))}. Afficher la source."><title>${esc(r.variant)} : ${metric==='delta'?sign(v):fmt(v)}${r.ci&&metric==='delta'?' · IC'+(r.ci_level||'?')+' ['+r.ci.map(x=>fmt(x)).join(' ; ')+']':''}</title><text x="0" y="${y+4}" font-family="sans-serif" font-size="11" fill="#385244">${esc(short)}</text>`;
    if(metric==="delta") {if(r.ci) s+=`<line x1="${x(r.ci[0])}" y1="${y}" x2="${x(r.ci[1])}" y2="${y}" stroke="${color}" stroke-width="2"/><line x1="${x(r.ci[0])}" y1="${y-4}" x2="${x(r.ci[0])}" y2="${y+4}" stroke="${color}"/><line x1="${x(r.ci[1])}" y1="${y-4}" x2="${x(r.ci[1])}" y2="${y+4}" stroke="${color}"/>`;s+=`<circle cx="${x(v)}" cy="${y}" r="5" fill="${color}"/>`;
    } else s+=`<rect x="${x(0)}" y="${y-8}" width="${Math.max(0,x(v)-x(0))}" height="16" rx="3" fill="${color}"/>`;
    s+=`<text x="${labelWidth+chartWidth+25}" y="${y+4}" font-family="sans-serif" font-size="11" fill="#274d3b">${metric==='delta'?sign(v):metric==='tokens'?integer(v):fmt(v)}</text></g>`;
  });
  s+=`<text x="0" y="${height-15}" font-family="sans-serif" font-size="9" fill="#718171">workere · instantané 12 novembre 2026 · ${esc(e.id.slice(0,8))}</text></svg>`;
  visibleSVG=s; return s;
}
function resultTable(rows) {
  return `<div class="table-wrap"><table><caption class="skip">Résultats exacts des variantes affichées, scores en pourcentage</caption><thead><tr><th>Variante</th><th>F1 %</th><th>P %</th><th>R %</th><th>TP / FP / FN</th><th>ΔF1 / IC</th><th>Source</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.variant)}${r.oracle?' <span class="chip diagnostic">Oracle</span>':''}</td><td class="number">${fmt(r.f1)}</td><td class="number">${fmt(r.precision)}</td><td class="number">${fmt(r.recall)}</td><td class="number">${integer(r.tp)} / ${integer(r.fp)} / ${integer(r.fn)}</td><td class="number">${sign(r.delta)}${r.ci?`<br><span class="small muted">IC${r.ci_level||'?'} [${fmt(r.ci[0])} ; ${fmt(r.ci[1])}]</span>`:''}</td><td><button class="table-button" data-evidence="${r.id}">Valeur & source ↗</button></td></tr>`).join('')}</tbody></table></div>`;
}
function results(e) {
  visibleSVG=""; visibleRows=[];
  if(!e.rows.length) return `${observationCharts(e)}<section class="panel"><h2>État des mesures</h2><p class="muted small">${esc(e.notes[0])}</p>${coverage(e)}<div class="button-row" style="margin-top:20px"><a class="button" href="${url(route.path,changed(route.params,{onglet:'protocole'}))}">Lire le détail technique →</a></div></section>${diagnostics(e)}`;
  const {rows,context,oracle}=currentRows(e), p=route.params;
  const perPage=[12,24,1000].includes(Number(p.get('taille')))?Number(p.get('taille')):12;
  const pages=Math.max(1,Math.ceil(rows.length/perPage));
  const page=Math.min(pages,Math.max(1,Number(p.get('page'))||1));
  const displayed=rows.slice((page-1)*perPage,page*perPage); visibleRows=rows;
  const metric=Object.hasOwn(METRIC_LABELS,p.get('mesure'))?p.get('mesure'):'f1';
  return `<div class="callout"><strong>Comparaison au sein d’un seul périmètre.</strong> ${esc(context)}. Les variantes affichées peuvent avoir des budgets différents. Les références de chaque ΔF1 sont précisées dans « Valeur & source ». ${oracle?'Les variantes oracle utilisent des informations gold et constituent des diagnostics.':''}</div>
  <section class="panel"><form id="result-controls" class="chart-tools"><label class="field context-field">Ensemble d’évaluation<select name="cohorte">${options(e.contexts.map(c=>[c,c]),context)}</select></label><label class="field">Mesure<select name="mesure">${options(Object.entries(METRIC_LABELS),metric)}</select></label><label class="field">Variantes par page<select name="taille">${options([['12','12 variantes'],['24','24 variantes'],['1000','Toutes les variantes']],String(perPage))}</select></label><label class="field context-field">Rechercher une variante<input type="search" name="variante" value="${esc(p.get('variante')||'')}" placeholder="base, repair, vote…"></label><label class="field">Ordre des variantes<select name="tri">${options([['','Nom de la variante'],['f1','F1 décroissant (exploratoire)']],p.get('tri')||'')}</select></label><label class="field">Type de mesure<select name="oracle">${options([['non','Systèmes évalués'],['oui','Oracle (diagnostic séparé)']],oracle?'oui':'non')}</select></label></form>
  <div class="chart-heading"><div><h2 class="chart-title">${e.number===52&&metric==='f1'?'Moyenne des micro-F1 des sous-ensembles (%)':esc(METRIC_LABELS[metric])}</h2><p>${rows.length} variantes dans cette sélection · page ${page} / ${pages}</p></div><div class="legend"><span><i></i>Variante testée</span><span><i class="gray"></i>Référence</span></div></div>
  <div class="chart-container" id="result-chart">${plot(displayed,metric,e,context)}</div><div class="chart-actions"><button class="quiet" data-action="export-svg" ${visibleSVG?'':'disabled'}>↓ Graphique SVG</button><button class="quiet" data-action="export-csv" ${rows.length?'':'disabled'}>↓ Tableau CSV (${rows.length})</button><button class="quiet" data-action="export-selection">↓ Données & sources JSON</button></div>
  <p class="chart-note">${metric==='delta'?'Les segments représentent uniquement les IC publiés. Leur niveau et leur unité de rééchantillonnage sont disponibles dans la provenance.':'Les barres commencent à zéro. Les deux décimales sont un format d’affichage ; la source conserve sa précision d’origine.'} Une case « — » indique une valeur absente. Le F1 nul reste affiché « 0,00 ».</p>
  ${displayed.length?resultTable(displayed):'<div class="empty">Aucune variante de ce type pour ce périmètre. Modifier les filtres de variantes ou revenir aux systèmes évalués.</div>'}
  <div class="pagination"><span>${rows.length?`${(page-1)*perPage+1}–${Math.min(page*perPage,rows.length)} sur ${rows.length}`:'0 résultat'} · même sélection pour le graphique et le tableau</span><div><button class="quiet" data-page="${page-1}" ${page<=1?'disabled':''}>← Précédent</button><button class="quiet" data-page="${page+1}" ${page>=pages?'disabled':''}>Suivant →</button></div></div></section>
  ${e.number>=113&&e.number<=115?'<div class="callout">Les nombreuses combinaisons ont été explorées sur ce même jeu. Le meilleur score de la table ne constitue pas une confirmation indépendante. Les coûts historiques et les vérifications s’ajoutent aux nouvelles lectures.</div>':''}${observationCharts(e)}${diagnostics(e)}`;
}

function observationCharts(e) {
  if(!e.observations?.length)return '';
  return [...new Set(e.observations.map(o=>o.group))].map(group=>{
    const rows=e.observations.filter(o=>o.group===group),unit=rows[0].unit;
    const min=Math.min(0,...rows.map(o=>o.value)),max=unit==='%'?100:Math.max(0,...rows.map(o=>o.total??o.value));
    const width=940,height=80+rows.length*40,x=v=>340+(v-min)/(max-min||1)*475;
    let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(group)}. ${esc(rows.map(o=>o.label+': '+fmt(o.value)+(o.total?' sur '+o.total:'')+' '+o.unit).join('; '))}"><rect width="${width}" height="${height}" fill="white"/><text x="340" y="20" font-size="10" fill="#617567">${esc(unit)}</text><line x1="${x(0)}" y1="30" x2="${x(0)}" y2="${height-20}" stroke="#aab7a6"/>`;
    rows.forEach((o,i)=>{const y=48+i*40;svg+=`<g><title>${esc(o.note)}</title><text x="0" y="${y+4}" font-size="11" fill="#385244">${esc(textLines(o.label,46,1)[0])}</text>${o.total?`<rect x="${x(0)}" y="${y-8}" width="${x(o.total)-x(0)}" height="16" fill="#e9ece4" rx="3"/>`:''}<rect x="${Math.min(x(0),x(o.value))}" y="${y-8}" width="${Math.abs(x(o.value)-x(0))}" height="16" fill="${o.value<0?'#af7752':'#007f76'}" rx="3"/><text x="840" y="${y+4}" font-size="11" fill="#274d3b">${fmt(o.value,o.total?0:2)}${o.total?' / '+integer(o.total):''}</text></g>`;});svg+='</svg>';
    return `<section class="panel"><h2 class="chart-title">${esc(group)}</h2><div class="chart-container">${svg}</div><div class="table-wrap"><table><thead><tr><th>Observation</th><th>Valeur</th><th>Portée & source</th></tr></thead><tbody>${rows.map(o=>`<tr><td>${esc(o.label)}</td><td class="number">${fmt(o.value,o.total?0:2)}${o.total?' / '+integer(o.total):''} ${esc(o.unit)}</td><td>${esc(o.note)}<br><span class="source-path">${sourceText(o.source)}</span></td></tr>`).join('')}</tbody></table></div><div class="button-row" style="margin-top:14px"><button class="quiet" data-action="export-observations">↓ Observations & sources JSON</button></div></section>`;
  }).join('');
}

function coverage(e) {
  const available=e.runs.filter(r=>r.log_available).length;
  return `<div class="technical-coverage"><div class="card coverage-card"><h3>Exécutions documentées</h3><strong>${e.runs.length}</strong><p>${e.runs.length?'Les reprises et échecs restent dans l’historique.':'Aucune exécution enregistrée à l’instantané.'}</p></div><div class="card coverage-card"><h3>Journaux locaux disponibles</h3><strong>${available} / ${e.runs.length}</strong><div class="coverage-bar"><span style="width:${e.runs.length?100*available/e.runs.length:0}%"></span></div><p>Disponibilité des preuves techniques, sans score anticipé.</p></div></div>`;
}
function diagnostics(e) {
  if(!e.diagnostics.length) return '';
  return `<section class="panel"><h2>Diagnostics, budget & validité</h2><p class="muted small">Agrégats enregistrés dans les sources. Les unités et les périmètres de coût restent ceux du protocole ; un coût incrémental n’est pas un coût total de déploiement.</p>${e.diagnostics.map(d=>`<details><summary>${esc(d.label)}</summary>${structuredTable(d.data)}<p class="source-path">${sourceText(d.source)}</p></details>`).join('')}</section>`;
}
function structuredTable(data) {
  if(data==null) return '<p class="small muted">Non renseigné dans cette sortie.</p>';
  if(typeof data!=='object') return `<p class="small">${esc(data)}</p>`;
  return `<pre class="json-data">${esc(JSON.stringify(data,null,2))}</pre>`;
}
function sourceText(s) {return esc((s?.file||'Source non renseignée')+(s?.pointer?' → '+s.pointer:'')+(s?.line?' · ligne '+s.line:'')+(s?.run?' · run '+s.run:''));}

function promptSection(e,compact=false) {
  const refs=e.prompt_refs||[];
  const templates=refs.filter(r=>DATA.prompt_bank[r.id].kind==='template');
  const builders=refs.filter(r=>DATA.prompt_bank[r.id].kind==='builder');
  return `<section class="panel"><div class="eyebrow">PROMPT & CONSTRUCTION DE LA REQUÊTE</div><h2>${e.state==='planifie'?'Sources de préparation des prompts':'Le texte donné au modèle'}</h2><p class="small muted">${esc(e.prompt_status)}</p>${e.prompt_upstream?`<p class="small">${expLink(e.prompt_upstream,'Voir le lecteur amont et son prompt')}</p>`:''}<p class="small muted">Un essai peut utiliser plusieurs conditions, documents et exemples. Les gabarits ci-dessous montrent le texte fixe et les variables ; les fonctions montrent leur assemblage exact. Une variable comme <code>{doc_text}</code> est remplacée par le document et les démonstrations autorisées à l’exécution.</p>${compact?`<a class="button" href="${url(route.path,changed(route.params,{onglet:'prompts'}))}">Afficher les ${templates.length} gabarits et ${builders.length} fonctions →</a>`:[...templates,...builders].map((r,i)=>{const p=DATA.prompt_bank[r.id];return `<details class="prompt-block" ${i===0?'open':''}><summary><span class="chip">${p.kind==='template'?'Texte / gabarit':'Assemblage en code'}</span> ${esc(p.label)} <span class="source-path">· ${esc(r.path)}</span></summary><p class="source-path">Commit ${esc(r.commit)} · ${r.line?'ligne '+r.line:''} · SHA-256 ${esc(r.blob_sha256)}</p><div class="button-row"><button class="quiet" data-prompt-download="${p.id}">↓ Texte exact</button></div><pre class="prompt-text">${esc(p.content)}</pre><a class="small" target="_blank" rel="noopener noreferrer" href="https://github.com/Fireman4740/workere/blob/${r.commit}/${r.path.split('/').map(encodeURIComponent).join('/')}#L${r.line||1}">Voir la source au commit ↗</a></details>`}).join('')}${!refs.length?'<p class="small muted">Aucun gabarit supplémentaire retrouvé dans les sources ciblées. Aucun prompt n’a été inventé pour remplir cette fiche.</p>':''}</section>`;
}

function sources(e) {
  const unique=new Map(); e.rows.forEach(r=>unique.set(JSON.stringify(r.source),r.source)); e.diagnostics.forEach(d=>unique.set(JSON.stringify(d.source),d.source));
  return `<section class="panel"><h2>Sources de la fiche</h2><p class="small muted">Les valeurs numériques, le protocole et les corrections sont intégrés à ce fichier HTML. Les chemins ci-dessous identifient les archives locales d’origine ; ils ne sont pas des liens vers des fichiers publiés. Les liens de code ouvrent GitHub à votre demande.</p><ul class="source-list">${e.sources.map(s=>`<li><span class="source-path">${esc(s)}</span></li>`).join('')}</ul><h3 style="margin-top:25px">Snapshot du code</h3><p class="source-path">${esc(e.branch)}<br>Commit : ${esc(e.code.commit||e.code.reference)}<br>Disponibilité vérifiée : ${e.code.git_show_available?'oui':'non'}</p><ul class="source-list">${(e.code.changed_files||[]).map(f=>`<li><a href="https://github.com/Fireman4740/workere/blob/${encodeURIComponent(e.code.commit||e.code.reference)}/${f.split('/').map(encodeURIComponent).join('/')}" target="_blank" rel="noopener noreferrer">${esc(f)} ↗</a></li>`).join('')}</ul></section>
  <section class="panel"><h2>Historique des exécutions (${e.runs.length})</h2><p class="small muted">États enregistrés à l’instantané du 12 novembre 2026. Une exécution terminée ou un statut de suivi perdu n’indique pas à lui seul la qualité scientifique du résultat.</p>${e.runs.length?`<div class="table-wrap"><table><thead><tr><th>Run</th><th>État</th><th>Durée enregistrée</th><th>Journal disponible</th><th>Commit</th></tr></thead><tbody>${e.runs.map(r=>`<tr><td><button class="table-button run-link" data-run="${r.id}">${r.id.slice(0,8)} ↗</button></td><td class="run-state">${esc(RUN_STATES[r.status]||r.status)}</td><td>${esc(r.duration??'—')}</td><td>${r.log_available?'Oui':'Non'}</td><td class="source-path">${esc((r.commit||'—').slice(0,12))}</td></tr>`).join('')}</tbody></table></div>`:'<p class="small muted">Nœud préparatoire sans exécution.</p>'}</section>
  <section class="panel"><h3>Références numériques intégrées</h3>${[...unique.values()].map(s=>`<p class="source-path">${sourceText(s)}</p>`).join('')||'<p class="small muted">Aucune série consolidée pour cette fiche.</p>'}</section>`;
}
function detail(e) {
  activeExperiment=e;
  const tab=['resultats','schema','protocole','prompts','sources'].includes(route.params.get('onglet'))?route.params.get('onglet'):'resultats';
  const list=selectedExperiments(filterParams()); let i=list.findIndex(x=>x.id===e.id);
  const previous=list[i-1],next=list[i+1];
  let content='';
  if(tab==='resultats') content=results(e);
  if(tab==='schema') content=`<section class="panel"><div class="chart-heading"><div><h2>${e.state==='planifie'?'La méthode prévue, sans exécution':'La solution implémentée et testée'}</h2><p>Schéma explicatif propre à la fiche ${String(e.number).padStart(3,'0')}</p></div><button class="quiet" data-action="export-diagram">↓ Schéma SVG</button></div><div class="pipeline-wrap">${diagram(e)}</div><p class="diagram-caption">Le nœud vert indique l’intervention de cet essai. Les flèches en pointillés montrent des entrées de configuration ou d’évaluation. Ce schéma résume l’implémentation documentée ; il ne représente pas une trace interne observée du modèle. ${e.state==='planifie'?'Cette fiche décrit un plan, sans résultat expérimental propre.':''}</p><div class="diagram-steps"><div><h3>1. Intervention</h3><p>${esc(e.method)}</p></div><div><h3>2. Évaluation</h3><p>${esc(e.evaluation)}</p></div><div><h3>3. Contrôles et validité</h3><p>${esc(e.checks)}</p></div><div><h3>4. Observation</h3><p>${esc(e.outcome)}</p></div></div></section>`;
  if(tab==='protocole') content=`<section class="panel"><div class="eyebrow">EN CLAIR</div><h2>Ce qui a été fait</h2><p class="plain-summary">${esc(e.simple)}</p><ol class="protocol-steps"><li><strong>Préparer les entrées.</strong> ${esc(e.evaluation)}</li><li><strong>Appliquer le changement testé.</strong> ${esc(e.method)}</li><li><strong>Construire la demande au modèle.</strong> ${e.prompt_upstream?'Cette étape réutilise des réponses enregistrées ; les prompts appartiennent aux lecteurs amont.':'Les gabarits, les champs variables et leurs fonctions d’assemblage sont présentés dans l’onglet Prompts.'}</li><li><strong>Contrôler les sorties et la comparaison.</strong> ${esc(e.checks)}</li><li><strong>Lire le résultat dans son périmètre.</strong> ${esc(e.outcome)}</li></ol></section><div class="protocol-grid"><section class="panel"><h3>Intervention testée</h3><p>${esc(e.method)}</p></section><section class="panel"><h3>Périmètre d’évaluation</h3><p>${esc(e.evaluation)}</p></section></div>${promptSection(e,true)}<section class="panel"><h3>Contrôles & limites</h3><p class="small">${esc(e.checks)}</p>${e.notes.map(n=>`<p class="small muted">${esc(n)}</p>`).join('')}</section><section class="panel"><h2>Détail technique complet</h2><div class="prose">${markdown(e.technical)}</div></section>`;
  if(tab==='prompts') content=promptSection(e);
  if(tab==='sources') content=sources(e);
  return `<a class="back-link" href="${url('/experiences',filterParams())}">← Revenir aux ${list.length} expériences de la sélection</a><section class="exp-head"><div class="chips"><span class="eyebrow" style="margin:0 8px 0 0">EXPÉRIENCE ${String(e.number).padStart(3,'0')}</span><a class="chip" href="${url('/experiences',new URLSearchParams({famille:e.family}))}">${esc(familyName(e.family))}</a>${e.datasets.map(d=>`<span class="chip">${esc(d)}</span>`).join('')}${chip(e)}</div><h1>${esc(e.title)}</h1><p class="intro">${esc(e.simple)}</p><p class="small muted">Intervention : ${esc(e.method)}</p></section><section class="lead-result ${e.state}"><div class="eyebrow">RÉSULTAT & PORTÉE</div><p>${esc(e.outcome)}</p></section><nav class="tabs" aria-label="Sections de l’expérience">${[['resultats','Graphiques & tableaux'],['schema','Schéma de la méthode'],['protocole','Protocole & détail'],['prompts','Prompts'],['sources','Sources & exécutions']].map(([id,label])=>`<a class="${id===tab?'active':''}" ${id===tab?'aria-current="page"':''} href="${url(route.path,changed(route.params,{onglet:id,page:null}))}">${label}</a>`).join('')}</nav>${content}<nav class="prev-next" aria-label="Expériences de la sélection"><div>${previous?`<a href="${linkExperiment(previous)}"><small>← PRÉCÉDENTE DANS LA SÉLECTION</small>${esc(previous.title)}</a>`:''}</div><div>${next?`<a href="${linkExperiment(next)}"><small>SUIVANTE DANS LA SÉLECTION →</small>${esc(next.title)}</a>`:''}</div></nav>`;
}

// Small, safe Markdown reader for the repository's prose/tables. Source HTML is
// escaped; unavailable archive links retain their human-readable source labels.
function inline(text) {
  const tokens=[];
  text=String(text).replace(/`([^`]+)`/g,(_,x)=>{tokens.push('<code>'+esc(x)+'</code>');return '\u0001'+(tokens.length-1)+'\u0001';});
  text=esc(text);
  text=text.replace(/\[([^\]]+)\]\(([^\s)]+)\)/g,(_,label,raw)=>{
    const target=raw.replace(/&amp;/g,'&'); let href='';
    if(/^https?:\/\//.test(target)) href=esc(target);
    else if(/synthese|recapitulatif/.test(target)) href='#/documents';
    else if(/corrections|ecarts/.test(target)) href='#/documents?onglet=corrections';
    else if(/#fiche-\d+/.test(target)) {const n=Number(target.match(/#fiche-(\d+)/)[1]),e=EXP.find(x=>x.number===n);if(e) href=linkExperiment(e,new URLSearchParams());}
    else if(/catalogue|detail-technique/.test(target)) href='#/experiences';
    else if(/inventaire-experiences/.test(target)) href='#/experiences';
    return href?`<a href="${href}" ${/^https?:/.test(target)?'target="_blank" rel="noopener noreferrer"':''}>${label}</a>`:`<span title="Référence de l’archive source">${label}</span>`;
  });
  text=text.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\*([^*]+)\*/g,'<em>$1</em>');
  return text.replace(/\u0001(\d+)\u0001/g,(_,i)=>tokens[Number(i)]);
}
function markdown(text) {
  const lines=String(text).split('\n'); let out='',paragraph=[],inCode=false,code=[],list=null;
  const flush=()=>{if(paragraph.length){out+='<p>'+inline(paragraph.join(' '))+'</p>';paragraph=[];}if(list){out+='</'+list+'>';list=null;}};
  for(let i=0;i<lines.length;i++) {
    const l=lines[i];
    if(/^```/.test(l)){flush();if(inCode){out+='<pre><code>'+esc(code.join('\n'))+'</code></pre>';code=[];}inCode=!inCode;continue;}
    if(inCode){code.push(l);continue;}
    if(/^<a id=/.test(l))continue;
    if(!l.trim()){flush();continue;}
    const h=l.match(/^(#{1,5})\s+(.+)/);if(h){flush();out+='<h'+Math.min(4,h[1].length)+'>'+inline(h[2])+'</h'+Math.min(4,h[1].length)+'>';continue;}
    if(l.trim().startsWith('|') && lines[i+1]?.match(/^\s*\|?[\s:|-]+\|/)){
      flush();const cells=x=>x.trim().replace(/^\||\|$/g,'').split('|').map(s=>s.trim());
      out+='<table><thead><tr>'+cells(l).map(c=>'<th>'+inline(c)+'</th>').join('')+'</tr></thead><tbody>';i+=2;
      while(i<lines.length && lines[i].trim().startsWith('|')){out+='<tr>'+cells(lines[i]).map(c=>'<td>'+inline(c)+'</td>').join('')+'</tr>';i++;}i--;out+='</tbody></table>';continue;
    }
    const item=l.match(/^\s*(?:[-*]|\d+\.)\s+(.+)/);if(item){if(paragraph.length){out+='<p>'+inline(paragraph.join(' '))+'</p>';paragraph=[];}const type=/^\s*\d/.test(l)?'ol':'ul';if(list!==type){if(list)out+='</'+list+'>';out+='<'+type+'>';list=type;}out+='<li>'+inline(item[1])+'</li>';continue;}
    if(/^---+$/.test(l.trim())){flush();out+='<hr>';continue;}
    if(l.startsWith('> ')){flush();out+='<blockquote>'+inline(l.slice(2))+'</blockquote>';continue;}
    paragraph.push(l);
  }
  flush();return out;
}
function documents() {
  const tab=['synthese','corrections','lecture'].includes(route.params.get('onglet'))?route.params.get('onglet'):'synthese';
  let content='';
  if(tab==='lecture') content=`<section class="panel prose"><h2>Contrat de lecture</h2><p>Les mentions et les candidats sont fournis par le corpus. Le score principal est le micro-F1 binaire des relations ; il additionne les comptes avant la division.</p><p><code>F1 (%) = 200 × TP / (2 × TP + FP + FN)</code></p><p>Les scores sont affichés en pourcentage et les écarts de F1 en points. Les corpus, cohortes et protocoles distincts restent dans des sélections séparées. Les moyennes de F1 des sous-ensembles portent une note explicite.</p><h3>Incertitude et validité</h3><p>Un IC provient d’un rééchantillonnage enregistré dans la source. Son unité peut être le document ou le topic. Les intervalles simultanés, les diagnostics post hoc et les contrôles oracle ne doivent pas être lus comme une confirmation indépendante.</p><p>Les résultats issus de caches déjà exposés sont exploratoires. Les anciennes cohortes de confirmation ont parfois été réutilisées ; la fiche précise leur rôle à chaque étape. Les expériences terminées avec des réponses invalides gardent leur limite technique.</p><h3>Traçabilité de l’export</h3><ul>${DATA.provenance.rules.map(x=>'<li>'+esc(x)+'</li>').join('')}</ul><p>${DATA.provenance.verified_log_records} enregistrements structurés ont été relus dans les journaux locaux lors de l’import. Les références, comptes, corrections et empreintes de sources sont inclus dans les données exportables.</p><div class="button-row"><button class="quiet" data-action="export-all">↓ Instantané complet JSON</button></div><h3>Construire et actualiser</h3><p>Le fichier HTML fonctionne directement depuis le disque. Le générateur Python fourni reconstruit le site depuis les données normalisées ; l’import optionnel relit les archives locales du projet. Voir les instructions du dossier web dans la branche documentaire.</p></section>`;
  else content=`<section class="panel"><div class="button-row" style="margin-bottom:25px"><button class="quiet" data-action="print">Imprimer / enregistrer en PDF</button><button class="quiet" data-action="export-report" data-report="${tab}">↓ Document Markdown</button></div><article class="prose">${markdown(DATA.reports[tab])}</article></section>`;
  return title("Documents & preuves", "Présenter les résultats, garder les nuances", "La synthèse de présentation, les corrections scientifiques et le contrat de lecture accompagnent les fiches interactives.")+`<nav class="tabs doc-tabs" aria-label="Documents">${[['synthese','Synthèse & présentation'],['corrections','Corrections & provenance'],['lecture','Guide de lecture & export']].map(([id,label])=>`<a class="${tab===id?'active':''}" href="#/documents?onglet=${id}">${label}</a>`).join('')}</nav>${content}`;
}

function showEvidence(row) {
  lastFocus=document.activeElement;
  const run=activeExperiment?.runs.find(r=>r.id===row.source.run);
  document.getElementById('evidence-body').innerHTML=`<div class="eyebrow">${esc(row.context)}</div><h3>${esc(row.variant)}</h3><p class="small">F1 : <strong>${fmt(row.f1)} %</strong> · précision : ${fmt(row.precision)} % · rappel : ${fmt(row.recall)} %<br>TP / FP / FN : ${integer(row.tp)} / ${integer(row.fp)} / ${integer(row.fn)}</p>${row.delta!=null?`<p class="small">ΔF1 : ${sign(row.delta)} points${row.ci?` · IC${row.ci_level||'?'} [${row.ci.map(x=>fmt(x)).join(' ; ')}]`:''}<br>Référence : ${esc(row.reference||'voir le champ du marqueur source')} · unité de rééchantillonnage : ${esc(row.resampling||'non précisée dans cette table')}</p>`:''}<p class="small muted">${esc(row.note)}</p><p class="source-path">${sourceText(row.source)}</p>${run?`<p class="source-path">SHA-256 du journal : ${esc(run.log_sha256||'non disponible')}</p>`:''}<details open><summary>Enregistrement numérique normalisé</summary><pre class="json-data">${esc(JSON.stringify(row,null,2))}</pre></details><p class="small muted">Les chemins identifient les archives de la revue. Le fichier HTML intègre les valeurs sélectionnées, sans journal brut ni contenu de corpus.</p>`;
  document.getElementById('evidence').showModal();
}
function showRun(id) {
  const run=activeExperiment?.runs.find(r=>r.id===id);if(!run)return;
  lastFocus=document.activeElement;
  document.getElementById('evidence-body').innerHTML=`<h3>Exécution ${esc(id.slice(0,8))}</h3><pre class="json-data">${esc(JSON.stringify(run,null,2))}</pre><p class="small muted">État documentaire au 12 novembre. La présence d’un journal et le statut d’une exécution ne remplacent pas la lecture des résultats.</p>`;
  document.getElementById('evidence').showModal();
}
function download(name,text,type) {
  const blob=new Blob([text],{type}),href=URL.createObjectURL(blob),a=document.createElement('a');a.href=href;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(href),1000);
  toast('Export préparé : '+name);
}
function toast(message){const t=document.getElementById('toast');t.textContent=message;t.style.display='block';clearTimeout(toast.timer);toast.timer=setTimeout(()=>t.style.display='none',3500);}
function exportCSV() {
  const columns=['experiment_number','experiment_id','context','variant','documents','f1_percent','precision_percent','recall_percent','tp','fp','fn','delta_f1_pp','ci_level','ci_low_pp','ci_high_pp','reference','resampling','oracle','note','source_file','source_pointer','run_id','line','extras_json'];
  const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
  const rows=visibleRows.map(r=>[activeExperiment.number,activeExperiment.id,r.context,r.variant,r.documents,r.f1,r.precision,r.recall,r.tp,r.fp,r.fn,r.delta,r.ci_level,r.ci?.[0],r.ci?.[1],r.reference,r.resampling,r.oracle,r.note,r.source.file,r.source.pointer,r.source.run,r.source.line,JSON.stringify(r.extras)]);
  download('workere-'+activeExperiment.number+'-selection.csv','\uFEFF'+[columns,...rows].map(r=>r.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8');
}
function updateControls(form, resetPage=false) {
  const p=new URLSearchParams(route.params);
  for(const [key,value] of new FormData(form).entries()) {if(value)p.set(key,value);else p.delete(key);}
  if(resetPage)p.delete('page');
  location.hash=url(route.path,p).slice(1);
}
function render() {
  const oldRoute=route;
  route=parseRoute();activeExperiment=null;visibleRows=[];visibleSVG='';
  let nav='home',crumb="Vue d’ensemble",html='';
  if(route.path==='/') html=home();
  else if(route.path==='/experiences'){html=catalogue();nav='experiences';crumb='Expériences';}
  else if(route.path==='/methodes'){html=methods();nav='methodes';crumb='Méthodes testées';}
  else if(route.path==='/top-10'){html=topTen();nav='top';crumb='Top 10 des méthodes';}
  else if(route.path==='/documents'){html=documents();nav='documents';crumb='Synthèse & sources';}
  else if(route.path.startsWith('/experience/')) {const e=BY_ID.get(route.path.split('/')[2]);if(e){html=detail(e);nav='experiences';crumb='Expériences / '+String(e.number).padStart(3,'0');}}
  if(!html) html=title('Navigation','Cette fiche est introuvable','Revenir au catalogue pour sélectionner une expérience.')+'<a class="button primary" href="#/experiences">Ouvrir le catalogue</a>';
  main.innerHTML=html;
  document.getElementById('breadcrumb').textContent='Résultats / '+crumb;
  document.title=(activeExperiment?activeExperiment.title:crumb)+' · workere';
  document.querySelectorAll('[data-nav]').forEach(a=>{const active=a.dataset.nav===nav;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  // Keep chart position during a variant or page change, reset for actual routes.
  if(oldRoute.path!==route.path || oldRoute.params?.get('onglet')!==route.params.get('onglet')) window.scrollTo(0,0);
  document.getElementById('filters')?.addEventListener('submit',ev=>{ev.preventDefault();updateControls(ev.currentTarget);});
  document.getElementById('ranking-controls')?.addEventListener('submit',ev=>ev.preventDefault());
  document.getElementById('ranking-controls')?.addEventListener('change',ev=>updateControls(ev.currentTarget));
  document.getElementById('filters')?.addEventListener('change',ev=>updateControls(ev.currentTarget));
  document.getElementById('result-controls')?.addEventListener('submit',ev=>{ev.preventDefault();updateControls(ev.currentTarget,true);});
  document.getElementById('result-controls')?.addEventListener('change',ev=>updateControls(ev.currentTarget,true));
  for(const form of document.querySelectorAll('#filters,#result-controls')) {
    form.querySelectorAll('input[type="search"]').forEach(input=>input.addEventListener('input',()=>{
      // Search updates the list without destroying the focused input.
      if(form.id==='filters') {
        const p=new URLSearchParams(route.params); if(input.value)p.set('q',input.value);else p.delete('q');route.params=p;
        history.replaceState(null,'',url(route.path,p));const list=selectedExperiments(p);
        document.getElementById('catalogue-list').innerHTML=listRows(list);document.getElementById('catalogue-count').innerHTML=`${list.length} expériences <span class="sep">/</span> 117 <a href="#/experiences" style="margin-left:12px">Réinitialiser</a>`;
      }
    }));
  }
}
main.addEventListener('click',event=>{
  const el=event.target.closest('[data-evidence],[data-action],[data-page],[data-run],[data-prompt-download]');if(!el)return;
  if(el.dataset.promptDownload){const p=DATA.prompt_bank[el.dataset.promptDownload];download('workere-prompt-'+p.id+'.txt',p.content,'text/plain;charset=utf-8');return;}
  if(el.dataset.evidence) {const row=activeExperiment?.rows.find(r=>r.id===el.dataset.evidence)||rankedMethods.find(x=>x.row.id===el.dataset.evidence)?.row;if(row)showEvidence(row);return;}
  if(el.dataset.run){showRun(el.dataset.run);return;}
  if(el.dataset.page){location.hash=url(route.path,changed(route.params,{page:el.dataset.page})).slice(1);return;}
  switch(el.dataset.action) {
    case 'export-svg':if(visibleSVG)download('workere-'+activeExperiment.number+'-graphique.svg',visibleSVG,'image/svg+xml');break;
    case 'export-diagram':download('workere-'+activeExperiment.number+'-methode.svg',diagram(activeExperiment),'image/svg+xml');break;
    case 'export-csv':exportCSV();break;
    case 'export-selection':download('workere-'+activeExperiment.number+'-selection.json',JSON.stringify({snapshot:DATA.snapshot,experiment:activeExperiment.id,method:activeExperiment.method,evaluation:activeExperiment.evaluation,checks:activeExperiment.checks,rows:visibleRows},null,2),'application/json');break;
    case 'export-observations':download('workere-'+activeExperiment.number+'-observations.json',JSON.stringify({experiment:activeExperiment.id,snapshot:DATA.snapshot,observations:activeExperiment.observations},null,2),'application/json');break;
    case 'export-ranking':download('workere-top-10-'+(route.params.get('corpus')||'CTB')+'.json',JSON.stringify({snapshot:DATA.snapshot,criterion:'Meilleur micro-F1 observé par méthode sur le même panel ; sélection exploratoire',ranking:rankedMethods.map((x,i)=>({rank:i+1,experiment:x.experiment.id,title:x.experiment.title,explanation:x.experiment.simple,eligible_variants:x.variants,row:x.row}))},null,2),'application/json');break;
    case 'export-all':download('workere-instantane-2026-11-12.json',JSON.stringify(DATA,null,2),'application/json');break;
    case 'export-report':download('workere-'+el.dataset.report+'.md',DATA.reports[el.dataset.report],'text/markdown;charset=utf-8');break;
    case 'print':window.print();break;
  }
});
main.addEventListener('keydown',event=>{if((event.key==='Enter'||event.key===' ')&&event.target.matches('g[data-evidence]')){event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
function setPresentation(value){present=value;document.body.classList.toggle('present',present);document.getElementById('exit-presentation').hidden=!present;document.getElementById('presentation').setAttribute('aria-pressed',String(present));}
document.getElementById('presentation').addEventListener('click',()=>setPresentation(!present));
document.getElementById('exit-presentation').addEventListener('click',()=>setPresentation(false));
document.addEventListener('keydown',ev=>{if(ev.key==='Escape'&&present)setPresentation(false);});
document.getElementById('close-evidence').addEventListener('click',()=>document.getElementById('evidence').close());
document.getElementById('evidence').addEventListener('close',()=>lastFocus?.focus());
window.addEventListener('hashchange',render);
document.querySelector('.skip').addEventListener('click',ev=>{ev.preventDefault();main.focus();main.scrollIntoView();});
render();
