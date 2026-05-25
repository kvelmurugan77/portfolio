import {S,log} from './state.js';import {rad} from './utils.js';
function norm(h){return String(h||'').trim().toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'')}
const num=v=>{const x=Number(v);return Number.isFinite(x)?x:null};
export function parseObstacles(text){
  const lines=text.trim().split(/\r?\n/).filter(Boolean);if(!lines.length)return[];const sep=(lines[0].match(/;/g)||[]).length>(lines[0].match(/,/g)||[]).length?';':',';const h=lines[0].split(sep).map(norm);const obs=[];
  for(const l of lines.slice(1)){const c=l.split(sep).map(x=>x.trim());const r={};h.forEach((k,i)=>r[k]=c[i]);const lat=num(r.lat??r.latitude),lon=num(r.lon??r.longitude),height=num(r.height??r.h),width=num(r.width??r.length??r.w),porosity=num(r.porosity??r.p)??0.3;if(lat!=null&&lon!=null&&height!=null)obs.push({lat,lon,height,width:width||50,porosity:Math.max(0,Math.min(1,porosity))})}
  S.obstacles=obs;log(`Imported ${obs.length} shelter obstacles`);return obs;
}
export function shelterFactor(t,dir){
  if(!S.obstacles?.length)return 1;const lat=t.lat,lon=t.lon,mx=111320*Math.cos(rad(lat))||1,my=111320;const th=rad(dir);const ux=Math.sin(th),uy=Math.cos(th);let total=0;
  for(const o of S.obstacles){const dx=(o.lon-lon)*mx,dy=(o.lat-lat)*my;const up=dx*ux+dy*uy; // obstacle upstream in FROM direction
    if(up<=0||up>35*o.height)continue;const cross=Math.abs(dx*Math.cos(th)-dy*Math.sin(th));if(cross>Math.max(o.width,3*o.height))continue;
    const xH=up/o.height;const hub=S.project.hubHeight||100;const heightFactor=Math.min(1,Math.pow(o.height/Math.max(1,hub),0.55));const core=Math.exp(-Math.pow((xH-8)/11,2));const lateral=Math.exp(-Math.pow(cross/Math.max(1,o.width*0.7+2*o.height),2));const def=(1-o.porosity)*0.28*heightFactor*core*lateral;total+=def*def}
  return Math.max(0.75,1-Math.sqrt(total));
}
