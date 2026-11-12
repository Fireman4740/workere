/* Run with playwright-cli run-code (file contents as the argument).
 * A real browser is required; this is not an @playwright/test suite. */
async (page) => {
  const errors=[],requests=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('request',r=>requests.push(r.url()));
  await page.context().setOffline(true);
  await page.reload();
  const base=page.url().split('#')[0];
  const experiments=await page.evaluate(()=>DATA.experiments.map(e=>({id:e.id,n:e.number,title:e.title,rows:e.rows.length})));
  const assert=(value,message)=>{if(!value)throw new Error(message);};
  const labels={resultats:'Graphiques & tableaux',schema:'Schéma de la méthode',protocole:'Protocole & détail',prompts:'Prompts',sources:'Sources & exécutions'};
  const go=async(e,tab='resultats',query='')=>{
    await page.evaluate(h=>location.hash=h,'#/experience/'+e.id+'?onglet='+tab+query);
    await page.waitForFunction(({id,label})=>activeExperiment?.id===id && document.querySelector('.tabs a.active')?.textContent===label,{id:e.id,label:labels[tab]});
  };
  const diagramTexts=new Set();let tabs=0;
  for(const e of experiments){
    for(const tab of ['resultats','schema','protocole','prompts','sources']){
      await go(e,tab);tabs++;
      assert((await page.locator('.exp-head h1').textContent())===e.title,'Wrong heading '+e.n);
      if(tab!=='prompts')assert(!/\b(?:NaN|undefined)\b/.test(await page.locator('main').innerText()),'Invalid rendered value '+e.n+' '+tab);
      if(tab==='schema'){
        assert(await page.locator('svg.pipeline').count()===1,'Missing diagram '+e.n);
        diagramTexts.add(await page.locator('svg.pipeline').textContent());
      }
      if(tab==='protocole')assert((await page.locator('.prose').innerText()).length>400,'Missing technical detail '+e.n);
    }
  }
  assert(diagramTexts.size===117,'Diagrams not distinct');
  await page.evaluate(()=>location.hash='#/top-10');
  await page.locator('#ranking-controls').waitFor();
  assert(await page.locator('.rank-card').count()===10,'Top 10 CTB missing');
  assert((await page.locator('.rank-card').first().innerText()).includes('61,19'),'Top CTB winner wrong');
  await page.locator('#ranking-controls select').selectOption('ESC');
  await page.waitForFunction(()=>route.params.get('corpus')==='ESC');
  assert(await page.locator('.rank-card').count()>=4,'ESC ranking missing');
  assert((await page.locator('.rank-card').first().innerText()).includes('48,64'),'ESC top winner wrong');
  await page.locator('#ranking-controls select').selectOption('MAVEN');
  await page.waitForFunction(()=>route.params.get('corpus')==='MAVEN');
  assert(await page.locator('.rank-card').count()===2,'MAVEN ranking missing');
  const w4=experiments.find(e=>e.n===113),w6=experiments.find(e=>e.n===115);
  await go(w4,'resultats','&taille=1000');
  await page.locator('select[name=taille]').selectOption('1000');
  await page.waitForFunction(()=>document.querySelector('.table-wrap tbody')?.children.length===96);
  assert(await page.locator('.table-wrap').first().locator('tbody tr').count()===96,'W4 matrix incomplete');
  await page.locator('select[name=mesure]').selectOption('delta');
  await page.waitForFunction(()=>document.querySelector('.chart-title')?.textContent==='ΔF1 (points)');
  assert(await page.locator('#result-chart circle').count()===96,'W4 interval plot incomplete');
  await page.locator('[data-evidence]').first().press('Enter');
  await page.locator('#evidence[open]').waitFor();
  assert((await page.locator('#evidence-body').innerText()).includes('WSP_COMBO_RESULT'),'Source inspector missing');
  await page.locator('#close-evidence').click();
  assert(await page.locator('#evidence[open]').count()===0,'Dialog did not close');
  const [csv]=await Promise.all([page.waitForEvent('download'),page.locator('[data-action=export-csv]').click()]);
  await csv.saveAs('output/playwright/w4-selection.csv');
  const [svg]=await Promise.all([page.waitForEvent('download'),page.locator('[data-action=export-svg]').click()]);
  await svg.saveAs('output/playwright/w4-graphique.svg');
  await go(w4,'schema');
  const [diagram]=await Promise.all([page.waitForEvent('download'),page.locator('[data-action=export-diagram]').click()]);
  await diagram.saveAs('output/playwright/w4-methode.svg');
  await go(experiments.find(e=>e.n===8),'prompts');
  assert(await page.locator('.prompt-text').count()>0,'Luna prompt absent');
  const [prompt]=await Promise.all([page.waitForEvent('download'),page.locator('[data-prompt-download]').first().click()]);
  await prompt.saveAs('output/playwright/prompt-luna.txt');
  await go(w4,'schema');
  await page.setViewportSize({width:1440,height:1000});
  await page.screenshot({path:'output/playwright/methode-w4.png',fullPage:true});
  await go(w6);
  assert((await page.locator('main').innerText()).includes('0,00'),'Zero score absent');
  assert((await page.locator('main').innerText()).includes('4 / 20'),'W6 invalid-output coverage absent');
  await page.evaluate(()=>location.hash='#/experiences?famille=poids');
  await page.locator('#filters').waitFor();
  assert(await page.locator('.exp-row').count()===8,'Family filter wrong');
  await page.locator('input[name=q]').fill('édition');
  const count=await page.locator('.exp-row').count();assert(count>0 && count<8,'Accent-insensitive search failed');
  await page.locator('input[name=q]').fill('W4');
  assert(await page.locator('.exp-row').count()===1,'W4 search failed');
  await page.locator('.exp-row').click();
  await page.locator('.exp-head').waitFor();
  await page.locator('.back-link').click();
  await page.locator('#filters').waitFor();
  assert(await page.locator('.exp-row').count()===1,'Filters not retained on return');
  const screens=[];
  for(const width of [390,768,1440]){
    await page.setViewportSize({width,height:900});
    await page.evaluate(()=>location.hash='#/');
    await page.locator('.hero').waitFor();
    assert(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),'Home horizontal overflow '+width);
    await page.screenshot({path:'output/playwright/accueil-'+width+'.png',fullPage:true});
    await go(w4);
    assert(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),'Detail horizontal overflow '+width);
    if(width===390)await page.screenshot({path:'output/playwright/resultats-mobile.png',fullPage:true});
    screens.push(width);
  }
  await page.locator('#presentation').click();
  assert(await page.locator('body.present').count()===1,'Presentation mode failed');
  await page.locator('#exit-presentation').click();
  assert(await page.locator('body.present').count()===0,'Exit presentation failed');
  await page.evaluate(()=>location.hash='#/documents?onglet=corrections');
  await page.locator('.prose').waitFor();
  assert((await page.locator('.prose').innerText()).includes('56,49'),'P1 correction absent');
  assert(errors.length===0,'Browser errors: '+errors.join('; '));
  assert(requests.filter(u=>/^https?:/.test(u)).length===0,'Unexpected network request');
  await page.goto(base+'#/');
  return {experiments:experiments.length,detail_tabs_checked:tabs,distinct_diagrams:diagramTexts.size,rankings:['CTB47 top 10','ESC40','MAVEN12'],w4_variants:96,exports:['CSV','graph SVG','method SVG','prompt text'],viewports:screens,offline:true,external_requests:0,browser_errors:errors};
}
