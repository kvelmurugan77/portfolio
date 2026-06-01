import {S,log} from './state.js';
function norm(h){return String(h||'').trim().toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'')}
const num=v=>{const x=Number(v);return Number.isFinite(x)?x:null};
export function parsePowerCtCurve(text){
  const lines=text.trim().split(/\r?\n/).filter(Boolean);if(!lines.length)return null;
  const sep=(lines[0].match(/;/g)||[]).length>(lines[0].match(/,/g)||[]).length?';':',';
  const h=lines[0].split(sep).map(norm);const pts=[];
  for(const l of lines.slice(1)){
    const c=l.split(sep).map(x=>x.trim());const r={};h.forEach((k,i)=>r[k]=c[i]);
    const ws=num(r.ws??r.wind_speed??r.speed??r.v);const p=num(r.power??r.power_kw??r.kw??r.p);const ct=num(r.ct??r.thrust_coefficient);
    if(ws!=null&&p!=null)pts.push({ws,p,ct:ct??null});
  }
  pts.sort((a,b)=>a.ws-b.ws);S.powerCurve={pts};
  let maxP=Math.max(...pts.map(x=>x.p));
  // Bug #7 fix: Auto-detect power units (kW, MW, or W)
  let unitLabel='kW';
  if(maxP>0&&maxP<500){unitLabel='MW';pts.forEach(p=>p.p*=1000);maxP*=1000;log('Power values appear to be in MW — auto-converted to kW');}
  else if(maxP>500000){unitLabel='W';pts.forEach(p=>p.p/=1000);maxP/=1000;log('Power values appear to be in W — auto-converted to kW');}
  if(maxP>0)S.project.ratedKW=maxP;
  log(`Imported power/CT curve: ${pts.length} points, rated=${maxP.toFixed(0)} kW (${unitLabel} detected)`);return S.powerCurve;
}
export function interpCurve(ws){
  const pc=S.powerCurve?.pts;if(!pc?.length)return null;
  if(ws<pc[0].ws||ws>pc[pc.length-1].ws)return{power:0,ct:0};
  for(let i=0;i<pc.length-1;i++)if(ws>=pc[i].ws&&ws<pc[i+1].ws){const t=(ws-pc[i].ws)/(pc[i+1].ws-pc[i].ws||1);const p=pc[i].p+t*(pc[i+1].p-pc[i].p);let ct=0;if(pc[i].ct!=null&&pc[i+1].ct!=null)ct=pc[i].ct+t*(pc[i+1].ct-pc[i].ct);else ct=null;return{power:p,ct}}
  return{power:0,ct:0};
}
