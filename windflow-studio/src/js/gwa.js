import {S,log} from './state.js';import {gamma,powerDensity} from './utils.js';

// List of CORS proxies to try (in order of preference)
const CORS_PROXIES = [
  url => `https://corsproxy.io/?${encodeURIComponent(url)}`,
  url => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
  url => `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(url)}`,
];

export async function downloadGWA(lat,lon,height,z0){
  const url=`https://globalwindatlas.info/api/gwa/custom/Lib/?lat=${lat}&long=${lon}`;
  let txt='', lastErr=null;

  // 1) Try direct fetch first (works in some environments / local dev)
  try{
    const r=await fetch(url,{headers:{'Accept':'text/plain'}});
    if(r.ok){txt=await r.text();}
  }catch(e){lastErr=e;}

  // 2) If direct fetch failed or returned non-GWA content, try CORS proxies
  if(!txt||!txt.includes('Generalized Wind Climate')){
    txt='';lastErr=null;
    for(const mkProxy of CORS_PROXIES){
      try{
        const proxyUrl=mkProxy(url);
        log(`Trying CORS proxy for GWA…`,'i');
        const r=await fetch(proxyUrl);
        if(!r.ok){lastErr=Error('Proxy HTTP '+r.status);continue;}
        const candidate=await r.text();
        if(candidate&&candidate.includes('Generalized Wind Climate')){
          txt=candidate;break;
        }
        lastErr=Error('Proxy returned non-GWA content');
      }catch(e){
        lastErr=e;continue;
      }
    }
  }

  // 3) If still no valid data, throw with helpful message
  if(!txt||!txt.includes('Generalized Wind Climate')){
    throw Error(
      'GWA download blocked by CORS. Use "Import GWA .lib" with a manually downloaded Global Wind Atlas Lib/GWC file. ' +
      'To download: visit https://globalwindatlas.info, search your site, and export the .lib file. ' +
      'Details: '+(lastErr?.message||'download failed')
    );
  }

  return setGWAFromText(txt,lat,lon,height,z0);
}

export function setGWAFromText(txt,lat=S.project.lat,lon=S.project.lon,height=S.project.hubHeight,z0=S.project.z0){
  const gwc=parseGwc(txt);
  const climate=interpGwc(gwc,height,z0);
  S.gwa={raw:gwc,climate,lat,lon,height,z0};
  log(`GWA climate loaded: ${climate.mean.toFixed(2)} m/s @ ${height}m, z0=${climate.roughness}`);
  return S.gwa;
}

function parseNums(line){return line.trim().split(/\s+/).map(Number).filter(Number.isFinite)}

export function parseGwc(txt){
  const lines=txt.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
  const countLine=lines.find(l=>/^\d+\s+\d+\s+\d+/.test(l));
  if(!countLine)throw Error('Invalid GWA/GWC file: missing dimension line');
  const idx=lines.indexOf(countLine);
  const [nr,nh,ns]=parseNums(countLine).map(x=>Math.round(x));
  const rough=parseNums(lines[idx+1]).slice(0,nr);
  const height=parseNums(lines[idx+2]).slice(0,nh);
  const sector=Array.from({length:ns},(_,i)=>i*360/ns);
  if(rough.length!==nr||height.length!==nh)throw Error('Invalid GWA/GWC file: roughness/height dimensions do not match');
  const A={},k={},freq={};
  let p=idx+3;
  for(const r of rough){
    const f=parseNums(lines[p++]).slice(0,ns);freq[r]=f;
    A[r]={};k[r]={};
    for(const h of height){
      A[r][h]=parseNums(lines[p++]).slice(0,ns);
      k[r][h]=parseNums(lines[p++]).slice(0,ns);
      if(A[r][h].length!==ns||k[r][h].length!==ns)throw Error('Invalid GWA/GWC file: A/k sector count mismatch');
    }
  }
  return{rough,height,sector,A,k,freq};
}

export function interpGwc(gwc,h,z0){
  // Bug #8 fix: Interpolate between two bracketing roughness classes instead of snapping to nearest
  const hs=gwc.height;
  const roughs=gwc.rough.sort((a,b)=>a-b);
  
  // Find bracketing roughness classes
  let rLo=roughs[0],rHi=roughs[roughs.length-1];
  for(let i=0;i<roughs.length-1;i++){
    if(roughs[i]<=z0&&roughs[i+1]>=z0){rLo=roughs[i];rHi=roughs[i+1];break;}
    if(roughs[i]>z0){rLo=roughs[Math.max(0,i-1)];rHi=roughs[i];break;}
  }
  if(z0>=roughs[roughs.length-1]){rLo=roughs[roughs.length-1];rHi=rLo;}
  if(z0<=roughs[0]){rLo=roughs[0];rHi=rLo;}
  
  // Interpolation weight in log-roughness space
  const rW=(rLo===rHi)?0:(Math.log(Math.max(z0,0.0001))-Math.log(Math.max(rLo,0.0001)))/(Math.log(Math.max(rHi,0.0001))-Math.log(Math.max(rLo,0.0001)));
  const effectiveR=rW<0.5?rLo:rHi; // For height interpolation, use the closer roughness class
  
  let lo=Math.max(...hs.filter(x=>x<=h)),hi=Math.min(...hs.filter(x=>x>=h));
  if(!Number.isFinite(lo))lo=hs[0];if(!Number.isFinite(hi))hi=hs[hs.length-1];
  const wA=lo===hi?0:(Math.log(h)-Math.log(lo))/(Math.log(hi)-Math.log(lo));
  // Bug #9 fix: Use linear interpolation for k (not log-height)
  const wK=lo===hi?0:(h-lo)/(hi-lo);
  
  const A=[],k=[];
  for(let i=0;i<gwc.sector.length;i++){
    // Interpolate A/k at each roughness class and height, then blend roughness classes
    const aLo=(1-wA)*gwc.A[rLo][lo][i]+wA*gwc.A[rLo][hi][i];
    const aHi=(1-wA)*gwc.A[rHi][lo][i]+wA*gwc.A[rHi][hi][i];
    A[i]=(1-rW)*aLo+rW*aHi;
    const kLo=(1-wK)*gwc.k[rLo][lo][i]+wK*gwc.k[rLo][hi][i];
    const kHi=(1-wK)*gwc.k[rHi][lo][i]+wK*gwc.k[rHi][hi][i];
    k[i]=(1-rW)*kLo+rW*kHi;
  }
  const roughness=z0; // Use the actual site roughness (not snapped class)
  let f=gwc.freq[rLo].map((fi,i)=>(1-rW)*fi+rW*(gwc.freq[rHi]?.[i]||fi));
  f=f.map(x=>x/100);
  const fs=f.reduce((a,b)=>a+b,0)||1;f=f.map(x=>x/fs);
  const mean=f.reduce((s,fi,i)=>s+fi*A[i]*gamma(1+1/k[i]),0);
  const pd=f.reduce((s,fi,i)=>s+fi*powerDensity(A[i],k[i]),0);
  return{roughness,height:h,A,k,freq:f,mean,powerDensity:pd,sectors:gwc.sector};
}
