import {S} from './state.js';
let map=null, turbineLayer=null, terrainLayer=null, contourLayer=null, roughLayer=null;

function boundsFromTurbines(){
  // Priority: use turbines only for zoom-to-layout
  if(S.turbines?.length)return S.turbines.map(t=>({lat:t.lat,lon:t.lon}));
  if(S.terrain)return [{lat:S.terrain.lat0,lon:S.terrain.lon0},{lat:S.terrain.lat1,lon:S.terrain.lon1}];
  return [{lat:S.project.lat,lon:S.project.lon}];
}

function boundsFromAll(){
  // All data combined for full view
  const pts=[];
  if(S.turbines?.length)pts.push(...S.turbines);
  if(S.terrain)pts.push({lat:S.terrain.lat0,lon:S.terrain.lon0},{lat:S.terrain.lat1,lon:S.terrain.lon1});
  if(!pts.length)pts.push({lat:S.project.lat,lon:S.project.lon});
  return pts;
}

function initLeaflet(){
  const el=document.getElementById('leafletMap');if(!el||typeof L==='undefined')return false;
  if(map)return true;
  map=L.map(el,{preferCanvas:true,zoomControl:true}).setView([S.project.lat,S.project.lon],12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(map);
  turbineLayer=L.layerGroup().addTo(map);terrainLayer=L.layerGroup().addTo(map);contourLayer=L.layerGroup().addTo(map);roughLayer=L.layerGroup().addTo(map);
  setTimeout(()=>map.invalidateSize(),100);
  return true;
}
function colorElev(z,min,max){const t=Math.max(0,Math.min(1,(z-min)/Math.max(1,max-min)));const h=220-t*180;return `hsla(${h},80%,50%,0.35)`}

export function drawMap(focusTurbines=false){
  if(initLeaflet()){
    setTimeout(()=>map.invalidateSize(),0);
    turbineLayer.clearLayers();terrainLayer.clearLayers();contourLayer.clearLayers();roughLayer.clearLayers();
    if(S.terrain){
      const T=S.terrain;const dLat=(T.lat1-T.lat0)/(T.ny-1),dLon=(T.lon1-T.lon0)/(T.nx-1);
      // Draw terrain heat for moderate grids only, sampled for speed.
      const step=Math.max(1,Math.ceil(T.nx/45));
      for(let i=0;i<T.ny-1;i+=step)for(let j=0;j<T.nx-1;j+=step){
        const z=T.grid[i][j];L.rectangle([[T.lat0+i*dLat,T.lon0+j*dLon],[T.lat0+Math.min(i+step,T.ny-1)*dLat,T.lon0+Math.min(j+step,T.nx-1)*dLon]],{stroke:false,fillColor:colorElev(z,T.minE,T.maxE),fillOpacity:.45,interactive:false}).addTo(terrainLayer);
      }
      let drawn=0;for(const c of (S.contours||[])){for(const seg of c.segs||[]){if(drawn++>2500)break;L.polyline(seg.map(p=>[p.lat,p.lon]),{color:'#111827',weight:.7,opacity:.55,interactive:false}).addTo(contourLayer)}if(drawn>2500)break}
    }
    if(S.roughness?.length){for(const r of S.roughness.slice(0,300)){L.polygon(r.pts.map(p=>[p.lat,p.lon]),{color:'#22c55e',weight:1,fillOpacity:.12,interactive:false}).addTo(roughLayer)}}
    for(const t of S.turbines){L.circleMarker([t.lat,t.lon],{radius:5,color:'#fff',weight:1,fillColor:'#58a6ff',fillOpacity:.95}).bindPopup(`<b>${t.name||'T'+t.id}</b><br>${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}`).addTo(turbineLayer)}

    // Zoom logic: focus on turbines when layout loaded, otherwise show all data
    const pts=focusTurbines?boundsFromTurbines():boundsFromAll();
    if(pts.length>1){
      const b=L.latLngBounds(pts.map(p=>[p.lat,p.lon]));
      if(b.isValid())map.fitBounds(b.pad(.2),{maxZoom:15,padding:[30,30]});
    }else{
      map.setView([S.project.lat,S.project.lon],12);
    }
    return;
  }
  // Canvas fallback if Leaflet CDN is blocked.
  if(document.body?.classList)document.body.classList.add('map-fallback');const cv=document.getElementById('mapCanvas');if(!cv)return;const ctx=cv.getContext('2d'),W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);ctx.fillStyle='#0b1220';ctx.fillRect(0,0,W,H);let pts=[];if(S.terrain){for(let i=0;i<S.terrain.ny;i++)for(let j=0;j<S.terrain.nx;j++)pts.push({lat:S.terrain.lat0+i*(S.terrain.lat1-S.terrain.lat0)/(S.terrain.ny-1),lon:S.terrain.lon0+j*(S.terrain.lon1-S.terrain.lon0)/(S.terrain.nx-1),z:S.terrain.grid[i][j]})}pts.push(...S.turbines);if(!pts.length)pts=[{lat:S.project.lat,lon:S.project.lon}];const minLat=Math.min(...pts.map(p=>p.lat)),maxLat=Math.max(...pts.map(p=>p.lat)),minLon=Math.min(...pts.map(p=>p.lon)),maxLon=Math.max(...pts.map(p=>p.lon));const x=lon=>40+(lon-minLon)/(maxLon-minLon||1)*(W-80),y=lat=>H-40-(lat-minLat)/(maxLat-minLat||1)*(H-80);ctx.fillStyle='#58a6ff';ctx.strokeStyle='#fff';for(const t of S.turbines){ctx.beginPath();ctx.arc(x(t.lon),y(t.lat),5,0,Math.PI*2);ctx.fill();ctx.stroke()}ctx.fillStyle='#c9d1d9';ctx.font='13px monospace';ctx.fillText(`${S.turbines.length} turbines`,16,22);
}
