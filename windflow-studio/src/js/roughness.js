import {S,log} from './state.js';import {rad} from './utils.js';

export async function downloadRoughness(lat,lon,radKm){
  // Determine bounding box: use terrain if available, otherwise use project coords + radius
  let lat0,lon0,lat1,lon1;
  if(S.terrain){
    const T=S.terrain;lat0=T.lat0;lon0=T.lon0;lat1=T.lat1;lon1=T.lon1;
  } else {
    // Fallback: use project coordinates with a default or provided radius
    const rKm=radKm||8;
    const dLat=rKm/111.32,dLon=rKm/(111.32*Math.cos(rad(lat||S.project.lat)));
    lat0=(lat||S.project.lat)-dLat;lat1=(lat||S.project.lat)+dLat;
    lon0=(lon||S.project.lon)-dLon;lon1=(lon||S.project.lon)+dLon;
  }

  const q=`[out:json][timeout:60][bbox:${lat0},${lon0},${lat1},${lon1}];
(way["landuse"];way["natural"~"wood|water|scrub|grassland|wetland|sand|bare_rock"];relation["landuse"]["type"="multipolygon"];);
out geom;`;

  // Retry with backoff for Overpass API (can also rate-limit)
  let d;
  for(let attempt=0;attempt<4;attempt++){
    try{
      const r=await fetch('https://overpass-api.de/api/interpreter',{
        method:'POST',
        body:'data='+encodeURIComponent(q)
      });
      if(r.status===429){
        const wait=3000*(attempt+1);
        log(`Overpass rate-limited (429), retrying in ${wait/1000}s…`,'w');
        await new Promise(res=>setTimeout(res,wait));
        continue;
      }
      if(!r.ok)throw Error('Overpass HTTP '+r.status);
      d=await r.json();
      break;
    }catch(e){
      if(attempt>=3)throw e;
      const wait=2000*(attempt+1);
      log(`Overpass fetch error, retry ${attempt+1} in ${wait/1000}s: ${e.message}`,'w');
      await new Promise(res=>setTimeout(res,wait));
    }
  }

  const map={
    farmland:.03,grass:.03,meadow:.03,grassland:.03,scrub:.05,
    forest:1.5,wood:1.5,residential:.4,commercial:.4,industrial:.5,
    water:.0002,wetland:.0002,retail:.3,construction:.1,orchard:.1,
    sand:.003,desert:.003,bare_rock:.01
  };

  S.roughness=(d.elements||[]).filter(e=>e.geometry&&e.geometry.length>2).map(e=>{
    const lu=e.tags?.landuse||e.tags?.natural||'other';
    return{lu,z0:map[lu]||.03,pts:e.geometry.map(p=>({lat:p.lat,lon:p.lon}))};
  });

  log(`OSM roughness: ${S.roughness.length} polygons (${(lat1-lat0)*111:.0f}\u00d7${(lon1-lon0)*111:.0f} km)`);
  return S.roughness;
}

export function pointInPoly(pt,poly){
  let c=false;
  for(let i=0,j=poly.length-1;i<poly.length;j=i++){
    const yi=poly[i].lat,yj=poly[j].lat,xi=poly[i].lon,xj=poly[j].lon;
    if(((yi>pt.lat)!==(yj>pt.lat))&&(pt.lon<(xj-xi)*(pt.lat-yi)/((yj-yi)||1e-12)+xi))c=!c;
  }
  return c;
}

export function localZ0(lat,lon,def=.03){
  for(const z of S.roughness)if(pointInPoly({lat,lon},z.pts))return z.z0;
  return def;
}

export function effectiveZ0Fetch(lat,lon,dirDeg,def=.03){
  // WAsP-like sector roughness approximation: sample upstream roughness over logarithmic fetch.
  if(!S.roughness?.length)return def;
  const fetch=[100,250,500,1000,2000,5000,10000,15000];
  const weights=[1.7,1.5,1.3,1.1,.9,.7,.5,.35];
  const th=rad(dirDeg);
  const mx=111320*Math.cos(rad(lat))||1,my=111320;
  let sw=0,sl=0;
  for(let i=0;i<fetch.length;i++){
    const d=fetch[i];
    const la=lat+Math.cos(th)*d/my;
    const lo=lon+Math.sin(th)*d/mx;
    const z=localZ0(la,lo,def);
    const w=weights[i];
    sl+=w*Math.log(Math.max(0.0002,z));sw+=w;
  }
  return Math.max(0.0002,Math.min(3,Math.exp(sl/sw)));
}

export function roughnessChangeRatio(lat,lon,dir,hubH,mastZ0,def=.03){
  // WAsP-like roughness-change approximation combining upstream effective z0
  // with internal-boundary-layer relaxation.
  const zEff=effectiveZ0Fetch(lat,lon,dir,def);
  const zM=Math.max(0.0002,mastZ0||def),zT=Math.max(0.0002,zEff);
  let base=Math.log(hubH/zT)/Math.log(hubH/zM);
  // Detect nearest major upstream roughness transition.
  const fetch=[100,250,500,1000,2000,5000,10000,15000];
  let dChange=15000,prev=localZ0(lat,lon,def);
  const th=rad(dir),mx=111320*Math.cos(rad(lat))||1,my=111320;
  for(const d of fetch){const la=lat+Math.cos(th)*d/my,lo=lon+Math.sin(th)*d/mx;const z=localZ0(la,lo,def);if(Math.abs(Math.log(z/prev))>0.55){dChange=d;break}prev=z}
  const ibl=Math.min(1,Math.max(0,Math.log((dChange+50)/50)/Math.log(15000/50)));
  return 1+(base-1)*ibl;
}
