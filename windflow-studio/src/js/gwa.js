import {S,log} from './state.js';import {gamma,powerDensity} from './utils.js';

export async function downloadGWA(lat,lon,height,z0){
  const url=`https://globalwindatlas.info/api/gwa/custom/Lib/?lat=${lat}&long=${lon}`;
  let txt='', lastErr=null;
  // Direct fetch works in some environments; browsers may block it due to missing CORS.
  try{const r=await fetch(url,{headers:{Referer:'https://globalwindatlas.info'}});if(!r.ok)throw Error('GWA HTTP '+r.status);txt=await r.text();}
  catch(e){lastErr=e;}
  if(!txt||!txt.includes('Generalized Wind Climate')){
    throw Error('GWA direct download is blocked by browser/CORS. Use "Import GWA .lib" with a Global Wind Atlas Lib/GWC file. Details: '+(lastErr?.message||'download failed'));
  }
  return setGWAFromText(txt,lat,lon,height,z0);
}
export function setGWAFromText(txt,lat=S.project.lat,lon=S.project.lon,height=S.project.hubHeight,z0=S.project.z0){
  const gwc=parseGwc(txt);const climate=interpGwc(gwc,height,z0);S.gwa={raw:gwc,climate,lat,lon,height,z0};
  log(`GWA climate loaded: ${climate.mean.toFixed(2)} m/s @ ${height}m, z0=${climate.roughness}`);return S.gwa;
}
function parseNums(line){return line.trim().split(/\s+/).map(Number).filter(Number.isFinite)}
export function parseGwc(txt){
  const lines=txt.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
  const countLine=lines.find(l=>/^\d+\s+\d+\s+\d+/.test(l));
  if(!countLine)throw Error('Invalid GWA/GWC file: missing dimension line');
  const idx=lines.indexOf(countLine);const [nr,nh,ns]=parseNums(countLine).map(x=>Math.round(x));
  const rough=parseNums(lines[idx+1]).slice(0,nr);const height=parseNums(lines[idx+2]).slice(0,nh);const sector=Array.from({length:ns},(_,i)=>i*360/ns);
  if(rough.length!==nr||height.length!==nh)throw Error('Invalid GWA/GWC file: roughness/height dimensions do not match');
  const A={},k={},freq={};let p=idx+3;
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
  const r=gwc.rough.reduce((a,b)=>Math.abs(b-z0)<Math.abs(a-z0)?b:a,gwc.rough[0]);
  const hs=gwc.height;let lo=Math.max(...hs.filter(x=>x<=h)),hi=Math.min(...hs.filter(x=>x>=h));
  if(!Number.isFinite(lo))lo=hs[0];if(!Number.isFinite(hi))hi=hs[hs.length-1];
  const w=lo===hi?0:(Math.log(h)-Math.log(lo))/(Math.log(hi)-Math.log(lo));
  const A=[],k=[];for(let i=0;i<gwc.sector.length;i++){A[i]=(1-w)*gwc.A[r][lo][i]+w*gwc.A[r][hi][i];k[i]=(1-w)*gwc.k[r][lo][i]+w*gwc.k[r][hi][i]}
  let f=gwc.freq[r].map(x=>x/100);const fs=f.reduce((a,b)=>a+b,0)||1;f=f.map(x=>x/fs);
  const mean=f.reduce((s,fi,i)=>s+fi*A[i]*gamma(1+1/k[i]),0);const pd=f.reduce((s,fi,i)=>s+fi*powerDensity(A[i],k[i]),0);
  return{roughness:r,height:h,A,k,freq:f,mean,powerDensity:pd,sectors:gwc.sector};
}
export function setGWAFromTimeSeries(sp,dir,height=S.project.hubHeight,z0=S.project.z0,fromHeight=100){
  const sectors=Array.from({length:12},()=>[]);const ratio=Math.log(height/Math.max(.0002,z0))/Math.log(fromHeight/Math.max(.0002,z0));
  for(let i=0;i<sp.length;i++){const v=sp[i];if(v==null||v<.1)continue;const d=dir[i]??0;const s=Math.floor((((d%360)+360)%360+15)/30)%12;sectors[s].push(v*ratio)}
  const total=sectors.reduce((a,b)=>a+b.length,0)||1;const A=[],k=[],freq=[],secDirs=[];
  for(let s=0;s<12;s++){const arr=sectors[s];freq[s]=arr.length/total;secDirs[s]=s*30;if(arr.length<3){A[s]=7;k[s]=2;continue}const mean=arr.reduce((a,b)=>a+b,0)/arr.length;const sd=Math.sqrt(arr.reduce((a,b)=>a+(b-mean)*(b-mean),0)/arr.length);const cv=Math.max(.05,sd/Math.max(.1,mean));k[s]=Math.max(1.1,Math.min(5,Math.pow(cv,-1.086)));A[s]=mean/gamma(1+1/k[s])}
  const mean=freq.reduce((a,f,i)=>a+f*A[i]*gamma(1+1/k[i]),0);const pd=freq.reduce((a,f,i)=>a+f*powerDensity(A[i],k[i]),0);S.gwa={raw:null,lat:S.project.lat,lon:S.project.lon,height,z0,climate:{roughness:z0,height,A,k,freq,mean,powerDensity:pd,sectors:secDirs},fallback:true};log(`Fallback wind climate from ERA5/time-series: ${mean.toFixed(2)} m/s @ ${height}m`,'w');return S.gwa}
