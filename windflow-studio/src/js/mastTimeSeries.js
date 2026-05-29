import {S,log} from './state.js';
function norm(h){return String(h||'').trim().toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'')}
function splitLine(line,sep){return line.split(sep).map(x=>x.trim().replace(/^"|"$/g,''))}
const num=v=>{if(v==null||v==='')return null;const x=Number(String(v).replace(',','.'));return Number.isFinite(x)?x:null}
function detectSep(line){const seps=[',',';','\t'];return seps.sort((a,b)=>(line.split(b).length-line.split(a).length))[0]}
function parseDate(v,date2){
  if(!v)return null;let s=String(v).trim();if(date2)s+=' '+String(date2).trim();
  // Windographer often exports dd/mm/yyyy hh:mm or yyyy-mm-dd hh:mm
  let d=new Date(s.replace(/\//g,'-'));
  if(!isNaN(d))return d.toISOString().slice(0,16);
  const m=s.match(/(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\s+(\d{1,2}):(\d{2})/);
  if(m){let[,dd,mm,yy,hh,mi]=m;if(+dd<=12&&+mm>12){[dd,mm]=[mm,dd]}if(yy.length===2)yy='20'+yy;d=new Date(Date.UTC(+yy,+mm-1,+dd,+hh,+mi));if(!isNaN(d))return d.toISOString().slice(0,16)}
  return s;
}
export function parseMastTimeSeries(text,opts={}){
  const raw=text.split(/\r?\n/).filter(l=>l.trim()&&!l.trim().startsWith('#'));
  if(!raw.length)throw Error('Empty mast time-series file');
  // Find header row containing speed/direction/date keywords
  let hi=0;for(let i=0;i<Math.min(30,raw.length);i++){const l=raw[i].toLowerCase();if((l.includes('speed')||l.includes('ws'))&&(l.includes('dir')||l.includes('direction'))){hi=i;break}}
  const sep=detectSep(raw[hi]);const headers=splitLine(raw[hi],sep).map(norm);const rows=[];
  const idx=(names)=>{for(const n of names){const j=headers.findIndex(h=>h===n||h.includes(n));if(j>=0)return j}return -1}
  const iTime=idx(['timestamp','date_time','datetime','time_stamp']);const iDate=idx(['date']);const iClock=idx(['time']);
  const iWS=idx(['wind_speed','speed','ws','velocity','m_s']);const iWD=idx(['wind_direction','direction','dir','wd']);
  const iTemp=idx(['temperature','temp']);const iPres=idx(['pressure','press','barometric']);
  if(iWS<0||iWD<0)throw Error('Could not find wind speed and direction columns');
  for(let r=hi+1;r<raw.length;r++){
    const c=splitLine(raw[r],sep);if(c.length<Math.max(iWS,iWD)+1)continue;
    const ws=num(c[iWS]),wd=num(c[iWD]);if(ws==null||wd==null)continue;
    const time=iTime>=0?parseDate(c[iTime]):parseDate(c[iDate],iClock>=0?c[iClock]:null);
    rows.push({time:time||String(rows.length),ws,wd:((wd%360)+360)%360,temp:iTemp>=0?num(c[iTemp]):null,pres:iPres>=0?num(c[iPres]):null});
  }
  if(rows.length<10)throw Error('Too few valid mast time-series records');
  const mean=rows.reduce((s,r)=>s+r.ws,0)/rows.length;
  S.mastTS={source:opts.source||'Imported mast/Windographer time series',records:rows,height:opts.height||S.project.mastHeight||100,mean};
  log(`Imported mast time series: ${rows.length.toLocaleString()} records, mean=${mean.toFixed(2)} m/s @ ${S.mastTS.height}m`);
  return S.mastTS;
}
export function activeTimeSeries(){return S.mastTS||S.era5||null}
