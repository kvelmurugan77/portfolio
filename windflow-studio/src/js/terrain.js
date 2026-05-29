import {S,log} from './state.js';import {rad} from './utils.js';
export function autoCI(range){if(range<50)return 5;if(range<200)return 10;if(range<500)return 20;return 50}

export async function downloadTerrain(lat,lon,radKm,ng,ciSel='auto',progress=()=>{}){
  const key=[lat.toFixed(5),lon.toFixed(5),radKm,ng,ciSel].join('|');
  if(S.terrainCache.has(key)){
    const v=S.terrainCache.get(key);
    S.terrain=v.terrain;S.contours=v.contours;
    log('Terrain loaded from cache');return v.terrain;
  }

  // Cap grid size to prevent API overload (max 60x60 = 3600 points)
  ng=Math.min(ng,60);
  if(ng>40)log(`Grid capped at ${ng}\u00d7${ng} for stability`,'w');

  const dLat=radKm/111.32,dLon=radKm/(111.32*Math.cos(rad(lat)));
  const lat0=lat-dLat,lat1=lat+dLat,lon0=lon-dLon,lon1=lon+dLon;

  // Generate grid points
  const lats=[],lons=[];
  for(let i=0;i<ng;i++)for(let j=0;j<ng;j++){
    lats.push(+(lat0+i*(lat1-lat0)/(ng-1)).toFixed(6));
    lons.push(+(lon0+j*(lon1-lon0)/(ng-1)).toFixed(6));
  }

  const totalPoints=lats.length;
  log(`Terrain: requesting ${totalPoints} elevation points in ${ng}\u00d7${ng} grid, radius ${radKm}km`);

  // Dynamic batch size: smaller batches for large grids to avoid URL length limits
  const BATCH=totalPoints>2000?30:totalPoints>1000?40:50;
  const chunks=[];
  for(let i=0;i<lats.length;i+=BATCH)
    chunks.push({i,la:lats.slice(i,i+BATCH),lo:lons.slice(i,i+BATCH)});

  log(`Terrain: ${chunks.length} API chunks (${BATCH} points each)`);

  const elev=new Array(lats.length).fill(null);
  let done=0,failedChunks=[];

  // SEQUENTIAL fetching — one request at a time with adaptive delay
  // This is much more reliable than parallel for Open-Meteo's rate limits
  const BASE_DELAY=totalPoints>1500?1200:totalPoints>800?800:500;

  progress(5,`Fetching ${chunks.length} chunks…`);

  for(const ch of chunks){
    const t0=Date.now();
    let success=false;
    for(let attempt=0;attempt<8;attempt++){
      try{
        const url=`https://api.open-meteo.com/v1/elevation?latitude=${ch.la.join(',')}&longitude=${ch.lo.join(',')}`;
        const r=await fetch(url);
        if(r.status===429){
          // Rate limited — exponential backoff with jitter, up to 60s
          const base=3000*Math.pow(2,Math.min(attempt,5));
          const jitter=Math.random()*2000;
          const wait=Math.min(base+jitter,60000);
          log(`Terrain rate-limited (429), waiting ${(wait/1000).toFixed(1)}s before retry ${attempt+1}…`,'w');
          await new Promise(res=>setTimeout(res,wait));
          continue;
        }
        if(r.status===503||r.status===504){
          const wait=8000*(attempt+1);
          log(`Terrain server busy (${r.status}), waiting ${wait/1000}s…`,'w');
          await new Promise(res=>setTimeout(res,wait));
          continue;
        }
        if(!r.ok)throw Error('HTTP '+r.status);
        const j=await r.json();
        const result=j.elevation||[];
        if(result.length!==ch.la.length){
          log(`Terrain chunk returned ${result.length}/${ch.la.length} elevations, retrying…`,'w');
          if(attempt<7){await new Promise(res=>setTimeout(res,3000));continue;}
        }
        result.forEach((v,k)=>{if(ch.i+k<elev.length)elev[ch.i+k]=v;});
        success=true;
        break;
      }catch(e){
        if(attempt>=7){
          log(`Terrain chunk failed after 8 retries: ${e.message}`,'e');
          break;
        }
        const wait=2000*(attempt+1)+Math.random()*1000;
        log(`Terrain fetch error, retry ${attempt+1}/8 in ${(wait/1000).toFixed(1)}s: ${e.message}`,'w');
        await new Promise(res=>setTimeout(res,wait));
      }
    }

    if(!success){
      failedChunks.push(ch);
    }

    done++;
    const pct=5+55*done/chunks.length;
    progress(pct,`Elevation ${done}/${chunks.length}${failedChunks.length?' ('+failedChunks.length+' failed)':''}`);

    // Adaptive delay between requests
    const elapsed=Date.now()-t0;
    const delay=Math.max(BASE_DELAY, BASE_DELAY-elapsed+200);
    if(done<chunks.length)await new Promise(r=>setTimeout(r,delay));
  }

  // Retry failed chunks with longer delays
  if(failedChunks.length>0 && failedChunks.length<=chunks.length*0.4){
    log(`Retrying ${failedChunks.length} failed terrain chunks with longer delays…`,'w');
    for(const ch of failedChunks){
      try{
        await new Promise(r=>setTimeout(r,5000));
        const url=`https://api.open-meteo.com/v1/elevation?latitude=${ch.la.join(',')}&longitude=${ch.lo.join(',')}`;
        for(let attempt=0;attempt<5;attempt++){
          const r=await fetch(url);
          if(r.status===429){await new Promise(res=>setTimeout(res,10000*(attempt+1)));continue;}
          if(!r.ok)throw Error('HTTP '+r.status);
          const j=await r.json();
          const result=j.elevation||[];
          result.forEach((v,k)=>{if(ch.i+k<elev.length)elev[ch.i+k]=v;});
          break;
        }
      }catch(e){
        log(`Terrain retry failed: ${e.message}`,'e');
      }
    }
  }

  // Check we got enough data
  const validCount=elev.filter(v=>v!=null).length;
  const coverage=validCount/totalPoints;
  if(coverage<0.5)throw Error(`Terrain download incomplete: only ${validCount}/${totalPoints} points (${(coverage*100).toFixed(0)}%). Try a smaller grid or radius.`);

  // Fill any remaining nulls with nearest-neighbor interpolation
  for(let idx=0;idx<elev.length;idx++){
    if(elev[idx]==null){
      const row=Math.floor(idx/ng),col=idx%ng;
      let best=null,bestDist=Infinity;
      for(let dr=-3;dr<=3;dr++)for(let dc=-3;dc<=3;dc++){
        const nr=row+dr,nc=col+dc;
        if(nr>=0&&nr<ng&&nc>=0&&nc<ng){
          const ni=nr*ng+nc;
          if(elev[ni]!=null){
            const d=dr*dr+dc*dc;
            if(d<bestDist){bestDist=d;best=elev[ni];}
          }
        }
      }
      elev[idx]=best!=null?best:0;
    }
  }

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
  log(`Terrain ${ng}\u00d7${ng}, ${levels.length} contours, ${minE.toFixed(0)}-${maxE.toFixed(0)}m, coverage ${(coverage*100).toFixed(0)}%`);
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
