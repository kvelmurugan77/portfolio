import {S,log} from './state.js';import {$} from './utils.js';
import {generateGrid,parseLayout} from './layout.js';
import {downloadTerrain} from './terrain.js';
import {downloadRoughness} from './roughness.js';
import {downloadGWA,setGWAFromText} from './gwa.js';
import {downloadERA5} from './era5.js';
import {runAEP} from './flow.js';
import {runTimeSeries} from './timeseries.js';
import {parseMastClimate} from './mastClimate.js';
import {parseMastTimeSeries} from './mastTimeSeries.js';
import {parseObstacles} from './shelter.js';
import {parsePowerCtCurve} from './powerCurve.js';
import {exportReport,exportTsCsv} from './report.js';
import {refresh,readProject,setupTabs} from './ui.js';

function status(s){$('dataStatus').textContent=s;log(s)}

window.addEventListener('load',()=>{
  setupTabs();
  generateGrid(3,4,7);
  refresh();

  // Auto-save project fields on change
  ['projectName','siteLat','siteLon','hubHeight','mastHeight','rotorD','ratedKW','lossPct','wakeK','wakeModel','wakeCombination','z0','z0ref'].forEach(id=>$(id).addEventListener('change',refresh));

  // Demo site
  $('btnDemo').onclick=()=>{
    Object.assign(S.project,{name:'Al Dawadmi Screening',lat:24.3125,lon:44.375,hubHeight:144,rotorD:210,ratedKW:11000,z0:.03,lossPct:5,wakeK:.04});
    for(const [k,v] of Object.entries({projectName:S.project.name,siteLat:S.project.lat,siteLon:S.project.lon,hubHeight:S.project.hubHeight,rotorD:S.project.rotorD,ratedKW:S.project.ratedKW,z0:S.project.z0,lossPct:S.project.lossPct,wakeK:S.project.wakeK,mastHeight:S.project.mastHeight,wakeModel:S.project.wakeModel,wakeCombination:S.project.wakeCombination,z0ref:S.project.z0ref}))$(k).value=v;
    generateGrid(3,4,7);refresh();
  };

  // Layout
  $('btnGrid').onclick=()=>{readProject();generateGrid(+$('gridRows').value,+$('gridCols').value,+$('gridSpacingD').value);refresh()};
  $('btnLoadLayout').onclick=async()=>{const f=$('layoutFile').files[0];if(!f)return alert('Choose CSV/TXT/KML');parseLayout(await f.text());refresh(true)};

  // After grid generation, also update project center to layout centroid
  $('btnGrid').onclick=()=>{
    readProject();
    generateGrid(+$('gridRows').value,+$('gridCols').value,+$('gridSpacingD').value);
    // Update site coords to layout centroid for terrain/data downloads
    if(S.turbines.length){
      const cLat=S.turbines.reduce((s,t)=>s+t.lat,0)/S.turbines.length;
      const cLon=S.turbines.reduce((s,t)=>s+t.lon,0)/S.turbines.length;
      $('siteLat').value=cLat.toFixed(5);$('siteLon').value=cLon.toFixed(5);
    }
    refresh(true);
  };

  // Terrain download — reduced grid default and better rate-limit handling
  $('btnTerrain').onclick=async()=>{
    try{
      readProject();
      status('Downloading terrain… this may take a minute');
      await downloadTerrain(
        S.project.lat,S.project.lon,
        +$('terrainRadius').value,
        +$('terrainGrid').value,
        $('contourInt').value,
        (p,m)=>status(`${p.toFixed(0)}% ${m}`)
      );
      status('Terrain complete');refresh();
    }catch(e){
      status('Terrain failed: '+e.message);
      log('Terrain error: '+e.message,'e');
    }
  };

  // Roughness download — now works independently of terrain
  $('btnRoughness').onclick=async()=>{
    try{
      readProject();
      status('Downloading OSM roughness… (may take 30-60s)');
      await downloadRoughness(S.project.lat,S.project.lon,+$('terrainRadius').value);
      status('Roughness complete');refresh();
    }catch(e){
      status('Roughness failed: '+e.message);
      log('Roughness error: '+e.message,'e');
    }
  };

  // GWA download — with CORS proxy fallback
  $('btnGWA').onclick=async()=>{
    try{
      readProject();
      status('Downloading GWA climate… (trying CORS proxy if needed)');
      await downloadGWA(S.project.lat,S.project.lon,S.project.hubHeight,S.project.z0);
      status('GWA complete');refresh();
    }catch(e){
      status('GWA failed: '+e.message);
      log('GWA error: '+e.message,'e');
    }
  };

  // GWA file import
  $('btnGWAFile').onclick=async()=>{
    try{
      readProject();const f=$('gwaFile').files[0];
      if(!f)return alert('Choose GWA .lib/.gwc file');
      setGWAFromText(await f.text(),S.project.lat,S.project.lon,S.project.hubHeight,S.project.z0);
      refresh();
    }catch(e){alert('GWA import failed: '+e.message)}
  };

  // ERA5/ERA5T download — with era5t toggle
  $('btnERA5').onclick=async()=>{
    try{
      readProject();
      const era5t=$('era5tToggle')?.checked||false;
      status(`Downloading ${era5t?'ERA5T (near-real-time)':'ERA5 historical'} time series…`);
      await downloadERA5(S.project.lat,S.project.lon,$('eraYears').value,'100m',era5t);
      status(`${era5t?'ERA5T':'ERA5'} complete: ${S.era5.sp.length.toLocaleString()} records`);
      refresh();
    }catch(e){
      status('ERA5 download failed: '+e.message);
      log('ERA5 error: '+e.message,'e');
    }
  };

  // Mast climate import
  $('btnMastClimate').onclick=async()=>{
    try{readProject();const f=$('mastClimateFile').files[0];if(!f)return alert('Choose LT mast climate CSV');parseMastClimate(await f.text(),{height:S.project.mastHeight,z0:S.project.z0});refresh()}catch(e){alert('Mast climate import failed: '+e.message)}
  };

  // Mast time-series import
  $('btnMastTS').onclick=async()=>{
    try{readProject();const f=$('mastTsFile').files[0];if(!f)return alert('Choose mast/Windographer time-series CSV');parseMastTimeSeries(await f.text(),{height:S.project.mastHeight,source:f.name});refresh()}catch(e){alert('Mast time-series import failed: '+e.message)}
  };

  // Obstacles import
  $('btnObstacles').onclick=async()=>{
    try{const f=$('obstacleFile').files[0];if(!f)return alert('Choose obstacle CSV');parseObstacles(await f.text());refresh()}catch(e){alert('Obstacle import failed: '+e.message)}
  };

  // Power curve import
  $('btnPowerCurve').onclick=async()=>{
    try{const f=$('pcFile').files[0];if(!f)return alert('Choose power/CT curve CSV');parsePowerCtCurve(await f.text());$('ratedKW').value=S.project.ratedKW;refresh()}catch(e){alert('Power curve import failed: '+e.message)}
  };

  // Run AEP
  $('btnRun').onclick=()=>{
    try{readProject();runAEP();refresh()}catch(e){alert(e.message)}
  };

  // Run Time Series
  $('btnTS').onclick=()=>{
    try{readProject();runTimeSeries(1);refresh()}catch(e){alert(e.message)}
  };

  // Export CSV
  $('btnCSV').onclick=()=>{
    try{exportTsCsv()}catch(e){alert(e.message)}
  };

  // Export Report
  $('btnReport').onclick=()=>{
    try{exportReport()}catch(e){alert(e.message)}
  };
});
