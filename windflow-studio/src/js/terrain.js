import {S,log} from './state.js';import {rad} from './utils.js';
export function autoCI(range){if(range<50)return 5;if(range<200)return 10;if(range<500)return 20;return 50}

export async function downloadTerrain(lat,lon,radKm,ng,ciSel='auto',progress=()=>{}){
  const key=[lat.toFixed(5),lon.toFixed(5),radKm,ng,ciSel].join('|');
  if(S.terrainCache.has(key)){
    const v=S.terrainCache.get(key);
    S.terrain=v.terrain;S.contours=v.contours;
    log('Terrain loaded from cache');return v.terrain;
  }

  const dLat=radKm/111.32,dLon=radKm/(111.32*Math.cos(rad(lat)));
  const lat0=lat-dLat,lat1=lat+dLat,lon0=lon-dLon,lon1=lon+dLon;
  const lats=[],lons=[];
  for(let i=0;i<ng;i++)for(let j=0;j<ng;j++){
    lats.push(+(lat0+i*(lat1-lat0)/(ng-1)).toFixed(6));
    lons.push(+(lon0+j*(lon1-lon0)/(ng-1)).toFixed(6));
  }

  // Chunk into batches of 80 points (well within Open-Meteo limits per request)
  const BATCH=80;
  const chunks=[];
  for(let i=0;i<lats.length;i+=BATCH)
    chunks.push({i,la:lats.slice(i,i+BATCH),lo:lons.slice(i,i+BATCH)});

  const elev=new Array(lats.length);
  let next=0,done=0;

  async function fetchChunk(ch){
    const url=`https://api.open-meteo.com/v1/elevation?latitude=${ch.la.join(',')}&longitude=${ch.lo.join(',')}`;
    // Retry with exponential backoff; handle 429 rate-limit gracefully
    for(let attempt=0;attempt<5;attempt++){
      try{
        const r=await fetch(url);
        if(r.status===429){
          // Rate limited — wait longer and retry
          const wait=2000*Math.pow(2,attempt);
          log(`Terrain rate-limited (429), retrying in ${wait/1000}s…`,'w');
          await new Promise(res=>setTimeout(res,wait));
          continue;
        }
        if(!r.ok)throw Error('HTTP '+r.status);
        const j=await r.json();
        return j.elevation||[];
      }catch(e){
        if(attempt>=4)throw e;
        const wait=1000*(attempt+1);
        log(`Terrain fetch error, retry ${attempt+1} in ${wait/1000}s: ${e.message}`,'w');
        await new Promise(res=>setTimeout(res,wait));
      }
    }
    throw Error('Terrain fetch failed after retries');
  }

  // Use only 2 concurrent workers (was 6) to avoid hitting rate limits
  const CONCURRENCY=2;
  // Add a minimum delay between consecutive requests per worker
  const MIN_DELAY=400; // ms between requests

  async function worker(){
    while(next<chunks.length){
      const ch=chunks[next++];
      const t0=Date.now();
      const arr=await fetchChunk(ch);
      arr.forEach((v,k)=>elev[ch.i+k]=v);
      done++;
      progress(10+45*done/chunks.length,`Elevation ${done}/${chunks.length}`);
      // Enforce minimum delay between requests
      const elapsed=Date.now()-t0;
      if(elapsed<MIN_DELAY)await new Promise(r=>setTimeout(r,MIN_DELAY-elapsed));
    }
  }

  await Promise.all(Array.from({length:Math.min(CONCURRENCY,chunks.length)},worker));

  const grid=[];
  for(let i=0;i<ng;i++){grid[i]=[];for(let j=0;j<ng;j++){
    const v=elev[i*ng+j];grid[i][j]=v==null||isNaN(v)?0:v;
  }}

  const vals=elev.filter(v=>v!=null&&!isNaN(v));
  const minE=Math.min(...vals),maxE=Math.max(...vals),meanE=vals.reduce((a,b)=>a+b,0)/vals.length;
  let ci=ciSel==='auto'?autoCI(maxE-minE):Number(ciSel)||10;
  if((maxE-minE)/ci>35)ci=Math.ceil(((maxE-minE)/35)/5)*5;

  progress(70,'Generating contours');

  const levels=[];for(let z=Math.ceil(minE/ci)*ci;z<=maxE;z+=ci)levels.push(z);
  const contours=marchingSquares(grid,lat0,lat1,lon0,lon1,ng,ng,levels);

  S.terrain={grid,lat0,lat1,lon0,lon1,ny:ng,nx:ng,minE,maxE,meanE,ci,levels};
  S.contours=levels.map(z=>({z,segs:contours[z]||[]}));
  S.terrainCache.set(key,{terrain:S.terrain,contours:S.contours});
  log(`Terrain ${ng}\u00d7${ng}, ${levels.length} contours, ${minE.toFixed(0)}-${maxE.toFixed(0)}m`);
  return S.terrain;
}

export function terrainAt(lat,lon){
  const T=S.terrain;if(!T)return null;
  const{lat0,lat1,lon0,lon1,ny,nx,grid}=T;
  if(lat<lat0||lat>lat1||lon<lon0||lon>lon1)return null;
  const y=(lat-lat0)/(lat1-lat0)*(ny-1),x=(lon-lon0)/(lon1-lon0)*(nx-1);
  const i=Math.max(0,Math.min(ny-2,Math.floor(y))),j=Math.max(0,Math.min(nx-2,Math.floor(x)));
  const fy=y-i,fx=x-j;
  return grid[i][j]*(1-fx)*(1-fy)+grid[i][j+1]*fx*(1-fy)+grid[i+1][j]*(1-fx)*fy+grid[i+1][j+1]*fx*fy;
}

function marchingSquares(grid,lat0,lat1,lon0,lon1,ny,nx,levels){
  const dLat=(lat1-lat0)/(ny-1),dLon=(lon1-lon0)/(nx-1),res={};
  for(const z of levels){
    const segs=[];
    for(let i=0;i<ny-1;i++)for(let j=0;j<nx-1;j++){
      const v=[grid[i][j],grid[i][j+1],grid[i+1][j+1],grid[i+1][j]],
            b=v.map(x=>x>=z);
      let pts=[];
      function ip(a,bv,pa,pb){const t=(z-a)/((bv-a)||1);return{lat:pa.lat+t*(pb.lat-pa.lat),lon:pa.lon+t*(pb.lon-pa.lon)}}
      const p=[{lat:lat0+i*dLat,lon:lon0+j*dLon},{lat:lat0+i*dLat,lon:lon0+(j+1)*dLon},{lat:lat0+(i+1)*dLat,lon:lon0+(j+1)*dLon},{lat:lat0+(i+1)*dLat,lon:lon0+j*dLon}];
      for(let e=0;e<4;e++){const e2=(e+1)%4;if(b[e]!==b[e2])pts.push(ip(v[e],v[e2],p[e],p[e2]))}
      if(pts.length===2)segs.push(pts);
      else if(pts.length===4){segs.push([pts[0],pts[1]],[pts[2],pts[3]])}
    }
    res[z]=segs;
  }
  return res;
}
