/*
 * YawScout streaming SCADA engine
 *
 * Parses File objects in bounded chunks inside a Web Worker. It deliberately
 * returns aggregates rather than raw SCADA rows so multi-million-row studies
 * do not require the main browser window to hold the full export in memory.
 */
'use strict';

const CHUNK_BYTES = 4 * 1024 * 1024;
const PROGRESS_EVERY = 50000;

function finite(x){ return typeof x === 'number' && Number.isFinite(x); }
function key(v){ return String(v ?? '').toLowerCase().replace(/[^a-z0-9]/g,''); }
function parseNumber(raw){
  if(raw === null || raw === undefined) return NaN;
  let s = String(raw).trim().replace(/\u00a0/g,'').replace(/\s/g,'');
  if(!s) return NaN;
  s = s.replace(/[^0-9,+.\-Ee]/g,'');
  const nc=(s.match(/,/g)||[]).length, nd=(s.match(/\./g)||[]).length;
  if(nc && nd){
    if(s.lastIndexOf(',') > s.lastIndexOf('.')) s=s.replace(/\./g,'').replace(',','.');
    else s=s.replace(/,/g,'');
  } else if(nc > 1) s=s.replace(/,/g,'');
  else if(nd > 1) s=s.replace(/\./g,'');
  else if(nc) s=s.replace(',','.');
  return Number(s);
}
function parseDate(raw, mode){
  if(raw === null || raw === undefined) return null;
  const s=String(raw).trim();
  if(!s) return null;
  const serial=parseNumber(raw);
  if(/^\d{4,5}(?:[.,]\d+)?$/.test(s) && finite(serial) && serial>20000 && serial<80000){
    const d=new Date(Date.UTC(1899,11,30)+serial*86400000);
    return isNaN(d) ? null : d;
  }
  let m=s.match(/^(\d{4})[-/]?(\d{2})[-/]?(\d{2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?/);
  if(m){
    const d=new Date(Date.UTC(+m[1],+m[2]-1,+m[3],+(m[4]||0),+(m[5]||0),+(m[6]||0)));
    return isNaN(d) ? null : d;
  }
  m=s.match(/^(\d{1,2})[\/.-](\d{1,2})[\/.-](\d{2,4})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?/);
  if(!m){ const d=new Date(s); return isNaN(d) ? null : d; }
  let a=+m[1],b=+m[2],y=+m[3];
  if(y<100) y+=y<70?2000:1900;
  const dayfirst=mode==='dayfirst';
  const d=new Date(Date.UTC(y,(dayfirst?b:a)-1,dayfirst?a:b,+(m[4]||0),+(m[5]||0),+(m[6]||0)));
  return isNaN(d) ? null : d;
}
function diffDeg(a,b){ return ((a-b+540)%360)-180; }
function median(a){
  const v=a.filter(finite).sort((x,y)=>x-y);
  if(!v.length) return NaN;
  const m=Math.floor(v.length/2);
  return v.length%2 ? v[m] : (v[m-1]+v[m])/2;
}
function detectDelimiter(text){
  const first=(text.split(/\r?\n/).find(x=>x.trim())||'');
  let best=',',count=-1;
  [',',';','\t','|'].forEach(d=>{
    let q=false,n=1;
    for(let i=0;i<first.length;i++){
      if(first[i]==='"') q=!q;
      else if(!q && first[i]===d) n++;
    }
    if(n>count){best=d;count=n;}
  });
  return best;
}

/* State-machine CSV parser that preserves quoted fields across Blob chunks. */
function csvStreamParser(delimiter,onRow){
  let row=[],field='',quoted=false,skipLF=false;
  function endRow(){ row.push(field); if(row.some(x=>String(x).trim()!=='')) onRow(row); row=[];field=''; }
  return {
    push(text){
      for(let i=0;i<text.length;i++){
        const c=text[i];
        if(skipLF){ skipLF=false; if(c==='\n') continue; }
        if(quoted){
          if(c==='"'){
            if(text[i+1]==='"'){ field+='"'; i++; }
            else quoted=false;
          } else field+=c;
          continue;
        }
        if(c==='"'){ quoted=true; continue; }
        if(c===delimiter){ row.push(field);field='';continue; }
        if(c==='\r'){ endRow();skipLF=true;continue; }
        if(c==='\n'){ endRow();continue; }
        field+=c;
      }
    },
    finish(){ if(field!=='' || row.length) endRow(); }
  };
}

async function iterateFile(file, delimiterOpt, headerRow, onHeader, onData){
  let delim=delimiterOpt, header=null, index=0;
  const firstText=await file.slice(0,Math.min(file.size,1024*1024)).text();
  if(delim==='auto') delim=detectDelimiter(firstText);
  const decoder=new TextDecoder('utf-8');
  const parser=csvStreamParser(delim,row=>{
    if(index===headerRow){
      header=row.map(x=>String(x).replace(/^\uFEFF/,'').trim());
      onHeader(header,delim);
    } else if(index>headerRow && header){
      onData(row,header);
    }
    index++;
  });
  for(let offset=0;offset<file.size;offset+=CHUNK_BYTES){
    const buf=await file.slice(offset,Math.min(file.size,offset+CHUNK_BYTES)).arrayBuffer();
    parser.push(decoder.decode(buf,{stream:offset+CHUNK_BYTES<file.size}));
  }
  parser.push(decoder.decode());
  parser.finish();
}

function indexMap(header){
  const out={}; header.forEach((h,i)=>out[String(h).trim()]=i); return out;
}
function makeFileContext(header,fileName,opts){
  const idx=indexMap(header);
  const getIndex=name=>name && Object.prototype.hasOwnProperty.call(idx,name) ? idx[name] : -1;
  if(opts.layout==='wide'){
    const layouts=(opts.wideLayouts||[]).map(l=>{
      const cols={}; Object.keys(l.columns||{}).forEach(k=>cols[k]=getIndex(l.columns[k]));
      return {id:l.id,cols};
    }).filter(l=>['windSpeed','power','windDir','nacelleDir'].every(k=>l.cols[k]>=0));
    return {wide:true,layouts,timestamp:getIndex(opts.map.timestamp),fileName};
  }
  const cols={}; Object.keys(opts.map||{}).forEach(k=>cols[k]=getIndex(opts.map[k]));
  return {wide:false,cols,fileName,sourceWTG:(opts.sourceIds||{})[fileName]||''};
}
function makeRecords(row,ctx,opts){
  const read=i=>i>=0 ? row[i] : '';
  if(ctx.wide){
    if(ctx.timestamp<0) return [];
    return ctx.layouts.map(l=>({
      timestamp:read(ctx.timestamp), turbine:l.id,
      windSpeed:read(l.cols.windSpeed), power:read(l.cols.power),
      windDir:read(l.cols.windDir), nacelleDir:read(l.cols.nacelleDir),
      status:read(l.cols.status), curtailment:read(l.cols.curtailment), pitch:read(l.cols.pitch)
    }));
  }
  const c=ctx.cols;
  return [{
    timestamp:read(c.timestamp), turbine:read(c.turbine)||ctx.sourceWTG,
    windSpeed:read(c.windSpeed), power:read(c.power), windDir:read(c.windDir), nacelleDir:read(c.nacelleDir),
    status:read(c.status), curtailment:read(c.curtailment), pitch:read(c.pitch)
  }];
}
function isFlagged(v){
  const x=String(v??'').trim().toLowerCase();
  return x!=='' && !['0','0.0','false','no','none','off','normal','nan','null','-'].includes(x);
}
function normalise(rec,opts){
  const c=opts.cfg, u=opts.units;
  const ts=parseDate(rec.timestamp,opts.dateMode), t=String(rec.turbine||'').trim();
  const ws=parseNumber(rec.windSpeed)*u.wf, p=parseNumber(rec.power)*u.pf;
  const wd=parseNumber(rec.windDir)*u.df, nd=parseNumber(rec.nacelleDir)*u.df;
  if(!ts||!t||!finite(ws)||!finite(p)||!finite(wd)||!finite(nd)) return {reason:'bad'};
  if(ws<c.wsMin || ws>c.wsMax || p<Math.max(.02*c.rated,10)) return {reason:'ws'};
  if(c.statuses.length && !c.statuses.includes(String(rec.status||'').trim().toLowerCase())) return {reason:'status'};
  if(opts.hasCurtail && isFlagged(rec.curtailment)) return {reason:'curtail'};
  const pitch=parseNumber(rec.pitch);
  if(opts.hasPitch && finite(pitch) && Math.abs(pitch)>c.maxPitch) return {reason:'pitch'};
  return {record:{ts,t,ws,p,wd:((wd%360)+360)%360,nd:((nd%360)+360)%360,offset:diffDeg(nd,wd)}};
}
function bin(x,w=.5){ return Math.floor(x/w); }
function plusBin(map,b,value){ const x=map[b]||(map[b]={sum:0,n:0});x.sum+=value;x.n++; }
function getTurbine(map,t){
  return map[t]||(map[t]={n:0,sin:0,cos:0,offsetBins:{},sectorBins:{},months:{},biasSum:0,biasSq:0,biasN:0});
}
function compactBins(map){ return Object.keys(map).map(k=>({key:+k,sum:map[k].sum,n:map[k].n})); }

async function run(opts){
  const passOne={turbines:{},fleetBins:{},reasons:{bad:0,ws:0,status:0,curtail:0,pitch:0},accepted:0,read:0};
  const total=opts.files.reduce((s,f)=>s+f.size,0);
  let processedBytes=0,lastProgress=0;
  function progress(phase,rows){
    if(rows-lastProgress>=PROGRESS_EVERY){
      lastProgress=rows;
      self.postMessage({type:'progress',phase,rows,bytes:processedBytes,total});
    }
  }
  async function processPass(phase,consume){
    processedBytes=0;lastProgress=0;
    for(const file of opts.files){
      let ctx=null;
      await iterateFile(file,opts.delimiter,opts.headerRow,(header)=>{ctx=makeFileContext(header,file.name,opts);},row=>{
        if(!ctx) return;
        const records=makeRecords(row,ctx,opts);
        records.forEach(rec=>consume(rec));
      });
      processedBytes+=file.size;
      self.postMessage({type:'progress',phase,rows:phase===1?passOne.read:0,bytes:processedBytes,total});
    }
  }
  await processPass(1,rec=>{
    passOne.read++;
    const n=normalise(rec,opts);
    if(!n.record){passOne.reasons[n.reason]=(passOne.reasons[n.reason]||0)+1;progress(1,passOne.read);return;}
    const r=n.record, t=getTurbine(passOne.turbines,r.t);
    t.n++;plusBin(t.ownBins||(t.ownBins={}),bin(r.ws),r.p);plusBin(passOne.fleetBins,bin(r.ws),r.p);
    passOne.accepted++;progress(1,passOne.read);
  });
  const passTwo={turbines:{},accepted:0,reasons:{bad:0,ws:0,status:0,curtail:0,pitch:0},read:0,timeOrderWarnings:0};
  let currentKey=null,currentGroup=[];
  function flushGroup(){
    if(currentGroup.length<3)return;
    currentGroup.forEach(r=>{
      const others=currentGroup.filter(x=>x.t!==r.t).map(x=>x.ws);
      if(others.length<2)return;
      const d=r.ws-median(others), t=getTurbine(passTwo.turbines,r.t);
      if(finite(d)){t.biasSum+=d;t.biasSq+=d*d;t.biasN++;}
    });
  }
  await processPass(2,rec=>{
    passTwo.read++;
    const n=normalise(rec,opts);
    if(!n.record){passTwo.reasons[n.reason]=(passTwo.reasons[n.reason]||0)+1;progress(2,passTwo.read);return;}
    const r=n.record, baseT=passOne.turbines[r.t], own=baseT&&baseT.ownBins[bin(r.ws)], fleet=passOne.fleetBins[bin(r.ws)];
    if(!own||!fleet||own.n===0||fleet.n===0){progress(2,passTwo.read);return;}
    const t=getTurbine(passTwo.turbines,r.t);t.n++;t.sin+=Math.sin(r.offset*Math.PI/180);t.cos+=Math.cos(r.offset*Math.PI/180);
    const month=r.ts.getUTCFullYear()+'-'+String(r.ts.getUTCMonth()+1).padStart(2,'0');
    const m=t.months[month]||(t.months[month]={sin:0,cos:0,n:0});m.sin+=Math.sin(r.offset*Math.PI/180);m.cos+=Math.cos(r.offset*Math.PI/180);m.n++;
    if(Math.abs(r.offset)<=30 && own.sum/own.n>=.03*opts.cfg.rated) plusBin(t.offsetBins,Math.floor((r.offset+30)/2),r.p/(own.sum/own.n));
    if(fleet.sum/fleet.n>=.03*opts.cfg.rated) plusBin(t.sectorBins,Math.floor(r.wd/30),r.p/(fleet.sum/fleet.n));
    const tsKey=r.ts.getTime();
    if(currentKey===null) currentKey=tsKey;
    if(tsKey!==currentKey){
      if(tsKey<currentKey) passTwo.timeOrderWarnings++;
      flushGroup();currentGroup=[];currentKey=tsKey;
    }
    currentGroup.push(r);
    passTwo.accepted++;progress(2,passTwo.read);
  });
  flushGroup();
  const turbines=Object.keys(passTwo.turbines).map(id=>{
    const t=passTwo.turbines[id],r=Math.sqrt(t.sin*t.sin+t.cos*t.cos)/Math.max(1,t.n);
    let staticOffset=Math.atan2(t.sin,t.cos)*180/Math.PI;staticOffset=((staticOffset+540)%360)-180;
    const offsetBins=compactBins(t.offsetBins).map(x=>({x:-29+x.key*2,y:x.sum/x.n,n:x.n}));
    const sectors=compactBins(t.sectorBins).map(x=>({x:x.key*30+15,y:x.sum/x.n,n:x.n}));
    const months=Object.keys(t.months).map(m=>{const v=t.months[m],R=Math.sqrt(v.sin*v.sin+v.cos*v.cos)/v.n;let mean=Math.atan2(v.sin,v.cos)*180/Math.PI;mean=((mean+540)%360)-180;return {m,mean,n:v.n,R};});
    const wsBias=t.biasN?{mean:t.biasSum/t.biasN,sd:Math.sqrt(Math.max(0,t.biasSq/t.biasN-(t.biasSum/t.biasN)**2)),n:t.biasN}:{mean:NaN,sd:NaN,n:0};
    const sd=Math.sqrt(-2*Math.log(Math.max(r,1e-12)))*180/Math.PI;
    return {turbine:id,n:t.n,staticOffset,offsetSd:sd,resultant:r,offsetBins,sectors,months,wsBias};
  });
  return {turbines,meta:{accepted:passTwo.accepted,read:passTwo.read,reasons:passTwo.reasons,timeOrderWarnings:passTwo.timeOrderWarnings,method:'streaming aggregates (two-pass, bounded-memory)'}};
}

self.onmessage=async e=>{
  if(!e.data||e.data.type!=='run')return;
  try{self.postMessage({type:'progress',phase:0,rows:0,bytes:0,total:e.data.options.files.reduce((s,f)=>s+f.size,0)});const result=await run(e.data.options);self.postMessage({type:'done',result});}
  catch(err){self.postMessage({type:'error',message:err&&err.message?err.message:String(err),stack:err&&err.stack?err.stack:''});}
};
