/* Targeted browser checks for self-contained method guides; playwright-cli run-code. */
async (page) => {
  const errors=[],requests=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('request',r=>requests.push(r.url()));
  await page.context().setOffline(true);
  await page.reload();
  const assert=(x,m)=>{if(!x)throw new Error(m);};
  const go=async hash=>{
    await page.evaluate(h=>location.hash=h,hash);
    await page.waitForFunction(h=>location.hash===h && route.path===h.slice(1).split('?')[0],hash);
  };
  const checked=[];
  for(const corpus of ['CTB','ESC','MAVEN']) {
    await go('#/top-10?corpus='+corpus);
    const count={CTB:10,ESC:6,MAVEN:2}[corpus];
    assert(await page.locator('.complete-method').count()===count,'Guide coverage '+corpus);
    await page.locator('.method-disclosure').evaluateAll(nodes=>nodes.forEach(n=>n.open=true));
    const comparisons=await page.evaluate(c=>rankedMethods.map(x=>({n:x.experiment.number,
      guide:methodGuide(x.experiment,x.row,c),comparisons:guideComparisons(x.experiment,x.row,c)})),corpus);
    for(const x of comparisons) {
      assert(x.guide.inputs&&x.guide.output&&x.guide.steps.length>=4,'Incomplete chain '+x.n);
      assert(x.comparisons.length>0,'Missing comparator '+x.n);
      assert(await page.locator(`[data-method="${x.n}"] .prompt-block`).count()>0,'No inline prompts '+x.n);
      if(x.n===113)assert(Math.abs(x.comparisons[0].delta+1.6520451179)<1e-7,'Wrong W4 starting point');
      if(corpus==='ESC'&&x.n===86)assert(Math.abs(x.comparisons[0].delta-.5765606865)<1e-7,'P2 count precision');
      checked.push(corpus+'/'+x.n);
    }
    assert(!/\b(?:NaN|undefined)\b/.test(await page.locator('main').innerText()),'Broken guide values '+corpus);
    for(const width of [390,768,1440]) {
      await page.setViewportSize({width,height:1000});
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Page overflow '+corpus+'/'+width);
    }
  }
  await go('#/top-10?corpus=CTB');
  const first=page.locator('.rank-card').first();
  await first.locator('.guide-prompts > summary').click();
  const prompt=first.locator('.prompt-block').first();
  await prompt.locator('summary').click();
  assert((await prompt.locator('pre').textContent()).length>50,'Empty prompt text');
  await page.setViewportSize({width:390,height:1000});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Expanded prompt overflow');
  const [textDownload]=await Promise.all([page.waitForEvent('download'),prompt.locator('[data-prompt-download]').click()]);
  await textDownload.saveAs('output/playwright/method-guide-prompt.txt');
  const [rankingDownload]=await Promise.all([page.waitForEvent('download'),page.locator('[data-action=export-ranking]').click()]);
  await rankingDownload.saveAs('output/playwright/method-guides-ctb.json');
  const frozen=await page.evaluate(()=>EXP.filter(e=>[85,103,109,113].includes(e.number)).map(e=>({id:e.id,n:e.number})));
  for(const e of frozen) {
    for(const tab of ['schema','protocole']) {
      await go('#/experience/'+e.id+'?onglet='+tab);
      assert(await page.locator('.complete-method').count()===(e.n===109?3:1),'Detail guide missing '+e.n+'/'+tab);
    }
  }
  assert(errors.length===0,'Browser errors '+errors.join(';'));
  const external=requests.filter(x=>/^https?:/.test(x));
  assert(external.length===0,'External requests');
  return {guides_checked:checked,viewports:[390,768,1440],detail_tabs:8,
    inline_prompts:true,ranking_and_prompt_exports:true,offline:true,external_requests:external.length,errors};
}
