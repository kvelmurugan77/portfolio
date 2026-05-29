import {S} from './state.js';import {$,fmt} from './utils.js';import {drawMap} from './map.js';

let _focusTurbines=false;

// Call refresh(true) after layout import to zoom map to turbines
export function refresh(focusTurbines=false){
  _focusTurbines=focusTurbines;
  readProject();
  $('layoutCount').textContent=S.turbines.length;
  $('capacityMW').textContent=fmt(S.turbines.length*S.project.ratedKW/1000,1)+' MW';
  if(S.windClimate){
    $('windSummary').innerHTML=`<div class=card><b>${fmt(S.windClimate.mean)}</b><br>Imported mast climate @ ${S.windClimate.height}m</div><div class=card><b>${S.windClimate.sectors.length}</b><br>Sectors</div>`;
    $('sectorTable').innerHTML='<tr><th>Sector</th><th>Freq %</th><th>A</th><th>k</th></tr>'+S.windClimate.sectors.map(s=>`<tr><td>${fmt(s.dir,0)}</td><td>${fmt(s.freq*100,2)}</td><td>${fmt(s.A)}</td><td>${fmt(s.k)}</td></tr>`).join('');
  }else if(S.gwa){
    $('windSummary').innerHTML=`<div class=card><b>${fmt(S.gwa.climate.mean)}</b><br>GWA WS @ ${S.project.hubHeight}m</div><div class=card><b>${fmt(S.gwa.climate.powerDensity,0)}</b><br>Power density W/m²</div><div class=card><b>${fmt(S.gwa.climate.roughness)}</b><br>GWA roughness</div>`;
    $('sectorTable').innerHTML='<tr><th>Sector</th><th>Freq %</th><th>A</th><th>k</th></tr>'+S.gwa.climate.sectors.map((d,i)=>`<tr><td>${d}</td><td>${fmt(S.gwa.climate.freq[i]*100,2)}</td><td>${fmt(S.gwa.climate.A[i])}</td><td>${fmt(S.gwa.climate.k[i])}</td></tr>`).join('');
  }
  if(S.results){
    $('aepSummary').innerHTML=`<div class=card><b>${fmt(S.results.netGWh,1)}</b><br>Net GWh/y</div><div class=card><b>${S.results.climateSource||'—'}</b><br>Wind climate source</div><div class=card><b>${fmt(S.results.wakeLoss,1)}%</b><br>Wake loss</div><div class=card><b>${fmt(S.results.cf,1)}%</b><br>Capacity factor</div>`;
    $('turbineTable').innerHTML='<tr><th>WTG</th><th>Lat</th><th>Lon</th><th>WS</th><th>Gross</th><th>Wake</th><th>Net</th><th>CF</th></tr>'+S.results.per.map(t=>`<tr><td>${t.name}</td><td>${fmt(t.lat,5)}</td><td>${fmt(t.lon,5)}</td><td>${fmt(t.meanWS)}</td><td>${fmt(t.grossGWh)}</td><td>${fmt(t.wakeLoss,1)}%</td><td>${fmt(t.netGWh)}</td><td>${fmt(t.cf,1)}%</td></tr>`).join('');
  }
  if(S.ts){
    $('tsSummary').innerHTML=`<div class=card><b>${fmt(S.ts.netGWh,1)}</b><br>Farm Net GWh</div><div class=card><b>${fmt(S.ts.wakeLoss,1)}%</b><br>Wake loss</div><div class=card><b>${fmt(S.ts.cf,1)}%</b><br>Farm CF</div><div class=card><b>${S.ts.source||'—'}</b><br>TS source</div>`;
    if($('tsTable'))$('tsTable').innerHTML='<tr><th>WTG</th><th>Mean WS</th><th>Gross GWh</th><th>Wake %</th><th>Net GWh</th><th>CF %</th></tr>'+S.ts.perTurbine.map(t=>`<tr><td>${t.name}</td><td>${fmt(t.meanWS)}</td><td>${fmt(t.grossGWh)}</td><td>${fmt(t.wakeLoss,1)}</td><td>${fmt(t.netGWh)}</td><td>${fmt(t.cf,1)}%</td></tr>`).join('');
  }
  drawMap(_focusTurbines);
  // Reset focus flag after one draw
  _focusTurbines=false;
}

export function readProject(){Object.assign(S.project,{name:$('projectName').value,lat:+$('siteLat').value,lon:+$('siteLon').value,hubHeight:+$('hubHeight').value,rotorD:+$('rotorD').value,ratedKW:+$('ratedKW').value,lossPct:+$('lossPct').value,wakeK:+$('wakeK').value,z0:+$('z0').value,mastHeight:+$('mastHeight').value,wakeModel:$('wakeModel').value,wakeCombination:$('wakeCombination').value,z0ref:+$('z0ref').value})}

export function setupTabs(){document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.tabbody').forEach(e=>e.classList.remove('active'));b.classList.add('active');$('tab-'+b.dataset.tab).classList.add('active');setTimeout(()=>drawMap(),80)})}
