import {S,log} from './state.js';
import {gamma} from './utils.js';
function norm(h){return String(h||'').trim().toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'')}
function rows(text){const lines=text.trim().split(/\r?\n/).filter(Boolean);if(!lines.length)return[];const sep=(lines[0].match(/;/g)||[]).length>(lines[0].match(/,/g)||[]).length?';':',';const h=lines[0].split(sep).map(norm);return lines.slice(1).map(l=>{const c=l.split(sep).map(x=>x.trim());const r={};h.forEach((k,i)=>r[k]=c[i]);return r})}
const num=v=>{const x=Number(String(v??'').replace('%',''));return Number.isFinite(x)?x:null};
export function parseMastClimate(text,opts={}){
  const rs=rows(text),secs=[];
  for(const r of rs){
    const dir=num(r.sector??r.dir??r.direction??r.sector_deg);const freq=num(r.freq??r.frequency??r.frequency_pct??r.freq_pct);
    const A=num(r.a??r.weibull_a??r.scale??r.owc_a);const k=num(r.k??r.weibull_k??r.shape??r.owc_k);
    if(dir==null||freq==null||A==null||k==null)continue;
    secs.push({dir:((dir%360)+360)%360,freq:freq>1?freq/100:freq,A,k});
  }
  const fs=secs.reduce((s,x)=>s+x.freq,0)||1;secs.forEach(s=>s.freq/=fs);secs.sort((a,b)=>a.dir-b.dir);
  const climate={source:'LT corrected mast climate',height:opts.height||S.project.mastHeight||100,z0:opts.z0||S.project.z0||0.03,sectors:secs,mean:secs.reduce((a,s)=>a+s.freq*s.A*gamma(1+1/s.k),0)};
  S.windClimate=climate;log(`Imported mast climate: ${secs.length} sectors, mean=${climate.mean.toFixed(2)} m/s @ ${climate.height}m`);return climate;
}
export function activeClimateAtHub(){
  // If mast climate exists, vertically extrapolate sector A to hub at mast roughness.
  if(S.windClimate?.sectors?.length){
    const c=S.windClimate,p=S.project;const z0=Math.max(0.0002,c.z0||p.z0||0.03);const ratio=Math.log(p.hubHeight/z0)/Math.log((c.height||p.mastHeight||100)/z0);
    return {source:c.source,height:p.hubHeight,sectors:c.sectors.map(s=>({dir:s.dir,freq:s.freq,A:s.A*ratio,k:s.k})),mean:c.sectors.reduce((a,s)=>a+s.freq*s.A*ratio*gamma(1+1/s.k),0)};
  }
  if(S.gwa?.climate){const c=S.gwa.climate;return{source:'GWA point climate',height:c.height,sectors:c.sectors.map((d,i)=>({dir:d,freq:c.freq[i],A:c.A[i],k:c.k[i]})),mean:c.mean}}
  return null;
}
