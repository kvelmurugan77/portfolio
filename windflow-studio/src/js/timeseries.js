import {S,log} from './state.js';import {logLaw} from './utils.js';import {siteRatio} from './flow.js';import {wakeRun,power} from './wake.js';import {activeTimeSeries} from './mastTimeSeries.js';
function hoursBetween(t1,t2){const a=new Date(t1),b=new Date(t2);if(isNaN(a)||isNaN(b))return 1;const h=(b-a)/36e5;return h>0&&h<48?h:1}
export function runTimeSeries(step=1){
  const src=activeTimeSeries();if(!src)throw Error('Import mast time series or download ERA5 first');if(!S.turbines.length)throw Error('No turbines');
  const p=S.project,N=S.turbines.length,loss=1-p.lossPct/100;const recs=src.records||src.sp?.map((ws,i)=>({time:src.time?.[i]||String(i),ws,wd:src.dir?.[i]||0}))||[];
  const fromH=src.height||(src.height==='10m'?10:100)||p.mastHeight||100;
  const per=Array.from({length:N},(_,i)=>({id:S.turbines[i].id,name:S.turbines[i].name,grossKWh:0,wakeKWh:0,netKWh:0,wsSum:0,n:0,monthly:new Array(12).fill(0)}));
  const farmMonthly=new Array(12).fill(0),farmRows=[],detailRows=[];let gross=0,wake=0,net=0,processed=0;
  for(let i=0;i<recs.length;i+=step){const r=recs[i];if(r.ws==null||r.ws<0.1)continue;const dtH=step*Math.max(1,hoursBetween(recs[i]?.time,recs[Math.min(i+step,recs.length-1)]?.time));const dir=r.wd??0;const wsHub=logLaw(r.ws,fromH,p.hubHeight,p.z0);const free=S.turbines.map(t=>wsHub*siteRatio(t,dir));const wsp=wakeRun(free,dir);let gFarm=0,nFarm=0,wFarm=0;const mo=r.time?new Date(r.time).getUTCMonth():0;
    for(let t=0;t<N;t++){const g=power(free[t])*dtH,w=power(wsp[t])*dtH,n=w*loss;per[t].grossKWh+=g;per[t].wakeKWh+=w;per[t].netKWh+=n;per[t].wsSum+=free[t];per[t].n++;if(mo>=0&&mo<12)per[t].monthly[mo]+=n;gFarm+=g;wFarm+=w;nFarm+=n;if(detailRows.length<250000)detailRows.push([r.time,S.turbines[t].name,free[t].toFixed(3),wsp[t].toFixed(3),dir.toFixed(1),(g/dtH).toFixed(1),(n/dtH).toFixed(1),dtH.toFixed(3)])}
    gross+=gFarm;wake+=wFarm;net+=nFarm;if(mo>=0&&mo<12)farmMonthly[mo]+=nFarm;if(farmRows.length<100000)farmRows.push([r.time,wsHub.toFixed(3),dir.toFixed(1),(gFarm/dtH).toFixed(1),(nFarm/dtH).toFixed(1),dtH.toFixed(3)]);processed++}
  per.forEach(t=>{t.meanWS=t.wsSum/Math.max(1,t.n);t.grossGWh=t.grossKWh/1e6;t.netGWh=t.netKWh/1e6;t.wakeLoss=t.grossKWh?(t.grossKWh-t.wakeKWh)/t.grossKWh*100:0;t.cf=t.netKWh/(p.ratedKW*8760)*100});
  S.ts={source:src.source||'ERA5',records:processed,grossGWh:gross/1e6,netGWh:net/1e6,wakeLoss:gross?(gross-wake)/gross*100:0,cf:net/(N*p.ratedKW*8760)*100,monthly:farmMonthly,perTurbine:per,farmRows,detailRows};
  log(`Time series (${S.ts.source}): ${processed.toLocaleString()} steps, farm net ${S.ts.netGWh.toFixed(2)} GWh`);return S.ts}
