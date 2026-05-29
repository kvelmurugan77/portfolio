import {S,log} from './state.js';import {weibullPdf,rad} from './utils.js';import {activeClimateAtHub} from './mastClimate.js';import {terrainAt} from './terrain.js';import {effectiveZ0Fetch,roughnessChangeRatio} from './roughness.js';import {wakeRun,power} from './wake.js';import {shelterFactor} from './shelter.js';
export function terrainSpeedup(lat,lon,dir,z0){
  if(!S.terrain)return 1;const z=terrainAt(lat,lon);if(z==null)return 1;
  // Advanced clean-mode orography: multi-scale slope + curvature response.
  // This remains understandable (no FFT Option C) but captures ridge/valley effects better than slope-only.
  const scales=[250,500,1000,2000,4000,8000],weights=[.10,.18,.25,.23,.16,.08];
  const th=rad(dir),mx=111320*Math.cos(rad(lat))||1,my=111320;let resp=0,wsum=0;
  for(let i=0;i<scales.length;i++){
    const d=scales[i],dLat=Math.cos(th)*d/my,dLon=Math.sin(th)*d/mx;
    const zu=terrainAt(lat+dLat,lon+dLon),zd=terrainAt(lat-dLat,lon-dLon);
    const zl=terrainAt(lat-dLon*my/mx,lon+dLat*mx/my),zr=terrainAt(lat+dLon*my/mx,lon-dLat*mx/my);
    if(zu==null||zd==null)continue;
    const slope=(z-zu)/d*0.60 + (zd-zu)/(2*d)*0.25;
    const curvAlong=(zd-2*z+zu)/(d*d);
    const curvCross=(zl!=null&&zr!=null)?(zl-2*z+zr)/(d*d):0;
    const ridge= -0.35*d*(curvAlong+0.5*curvCross); // positive on crests, negative in valleys
    resp+=weights[i]*(slope+ridge);wsum+=weights[i];
  }
  if(!wsum)return 1;
  const amp=1.70;return Math.max(.70,Math.min(1.42,Math.exp(Math.max(-.28,Math.min(.28,amp*resp/wsum)))));
}
export function siteRatio(t,dir){
  const p=S.project;
  const z0eff=effectiveZ0Fetch(t.lat,t.lon,dir,p.z0);
  const rough=roughnessChangeRatio(t.lat,t.lon,dir,p.hubHeight,p.z0,p.z0);
  const oro=terrainSpeedup(t.lat,t.lon,dir,z0eff);
  const sec=Math.round((((dir%360)+360)%360)/30)%12;
  const cal=S.calibration?.sectorSR?.[sec]??1; // optional validation/calibration multiplier
  return Math.max(.50,Math.min(1.70,rough*oro*shelterFactor(t,dir)*cal));
}
export function estimateRIX(){if(!S.terrain)return 0;const T=S.terrain,{grid,ny,nx}=T;const dx=((T.lon1-T.lon0)/(nx-1))*111320*Math.cos(rad((T.lat0+T.lat1)/2))||1,dy=((T.lat1-T.lat0)/(ny-1))*111320||1;let steep=0,total=0;for(let i=1;i<ny-1;i++)for(let j=1;j<nx-1;j++){const dzdx=(grid[i][j+1]-grid[i][j-1])/(2*dx),dzdy=(grid[i+1][j]-grid[i-1][j])/(2*dy),s=Math.sqrt(dzdx*dzdx+dzdy*dzdy);if(s>0.30)steep++;total++}return total?100*steep/total:0}

export function buildGwcReport(climate){
  const p=S.project,zref=p.z0ref||0.05,zmast=p.z0||0.03;
  return climate.sectors.map(sec=>{
    const toRef=Math.log(p.hubHeight/zref)/Math.log(p.hubHeight/zmast);
    return{dir:sec.dir,freq:sec.freq,A_OWC:sec.A,k:sec.k,A_GWC:sec.A*toRef,z0ref:zref};
  });
}

export function runAEP(){
  const climate=activeClimateAtHub();
  if(!climate)throw Error('Import LT mast climate or download GWA first');
  if(!S.turbines.length)throw Error('Load/generate layout first');
  const p=S.project,N=S.turbines.length;
  const gross=new Array(N).fill(0),wake=new Array(N).fill(0),mean=new Array(N).fill(0),probSum=new Array(N).fill(0);
  for(const secObj of climate.sectors){
    const dir=secObj.dir,A=secObj.A,k=secObj.k,freq=secObj.freq;
    for(let v=.5;v<=32;v+=.5){
      const prob=freq*weibullPdf(v,A,k)*.5;if(prob<1e-9)continue;
      const free=S.turbines.map(t=>v*siteRatio(t,dir));
      const wsp=wakeRun(free,dir);
      for(let i=0;i<N;i++){gross[i]+=power(free[i])*prob*8760;wake[i]+=power(wsp[i])*prob*8760;mean[i]+=free[i]*prob;probSum[i]+=prob}
    }
  }
  const loss=1-p.lossPct/100;
  const per=S.turbines.map((t,i)=>({id:t.id,name:t.name,lat:t.lat,lon:t.lon,meanWS:mean[i]/(probSum[i]||1),grossGWh:gross[i]/1e6,wakeLoss:gross[i]?(gross[i]-wake[i])/gross[i]*100:0,netGWh:wake[i]*loss/1e6,cf:wake[i]*loss/(p.ratedKW*8760)*100}));
  const grossGWh=per.reduce((s,t)=>s+t.grossGWh,0),netGWh=per.reduce((s,t)=>s+t.netGWh,0),wakeLoss=grossGWh?(grossGWh-per.reduce((s,t)=>s+t.netGWh,0)/loss)/grossGWh*100:0;
  const rix=estimateRIX();
  S.results={per,grossGWh,netGWh,wakeLoss,cf:netGWh*1000/(N*p.ratedKW/1000*8760)*100,capacityMW:N*p.ratedKW/1000,rix,climateSource:climate.source,climateMean:climate.mean,gwc:buildGwcReport(climate)};
  if(rix>5)log(`RIX ${rix.toFixed(1)}%: complex terrain; validate against mast/LiDAR or CFD/WAsP.`,'w');
  log(`AEP complete (${climate.source}): Net ${netGWh.toFixed(1)} GWh, wake ${wakeLoss.toFixed(1)}%, RIX ${rix.toFixed(1)}%`);
  return S.results;
}
