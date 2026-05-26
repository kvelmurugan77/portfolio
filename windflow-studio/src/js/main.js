import {S,log} from './state.js';import {$} from './utils.js';import {generateGrid,parseLayout} from './layout.js';import {downloadTerrain} from './terrain.js';import {downloadRoughness} from './roughness.js';import {downloadGWA,setGWAFromText} from './gwa.js';import {downloadERA5} from './era5.js';import {runAEP} from './flow.js';import {runTimeSeries} from './timeseries.js';import {parseMastClimate} from './mastClimate.js';import {parseMastTimeSeries} from './mastTimeSeries.js';import {parseObstacles} from './shelter.js';import {parsePowerCtCurve} from './powerCurve.js';import {exportReport,exportTsCsv} from './report.js';import {refresh,readProject,setupTabs} from './ui.js';
function status(s){$('dataStatus').textContent=s;log(s)}
window.addEventListener('load',()=>{setupTabs();generateGrid(3,4,7);refresh();
['projectName','siteLat','siteLon','hubHeight','mastHeight','rotorD','ratedKW','lossPct','wakeK','wakeModel','wakeCombination','z0','z0ref'].forEach(id=>$(id).addEventListener('change',refresh));
$('btnDemo').onclick=()=>{Object.assign(S.project,{name:'Al Dawadmi Screening',lat:24.3125,lon:44.375,hubHeight:144,rotorD:210,ratedKW:11000,z0:.03,lossPct:5,wakeK:.04});for(const [k,v] of Object.entries({projectName:S.project.name,siteLat:S.project.lat,siteLon:S.project.lon,hubHeight:S.project.hubHeight,rotorD:S.project.rotorD,ratedKW:S.project.ratedKW,z0:S.project.z0,lossPct:S.project.lossPct,wakeK:S.project.wakeK,mastHeight:S.project.mastHeight,wakeModel:S.project.wakeModel,wakeCombination:S.project.wakeCombination,z0ref:S.project.z0ref}))$(k).value=v;generateGrid(3,4,7);refresh()};
$('btnGrid').onclick=()=>{readProject();generateGrid(+$('gridRows').value,+$('gridCols').value,+$('gridSpacingD').value);refresh()};
$('btnLoadLayout').onclick=async()=>{const f=$('layoutFile').files[0];if(!f)return alert('Choose CSV/TXT/KML');parseLayout(await f.text());refresh()};
$('btnTerrain').onclick=async()=>{try{readProject();await downloadTerrain(S.project.lat,S.project.lon,+$('terrainRadius').value,+$('terrainGrid').value,$('contourInt').value,(p,m)=>status(`${p.toFixed(0)}% ${m}`));status('Terrain complete');refresh()}catch(e){status('Terrain failed: '+e.message)}};
$('btnRoughness').onclick=async()=>{try{await downloadRoughness();refresh()}catch(e){status('Roughness failed: '+e.message)}};
$('btnGWA').onclick=async()=>{try{readProject();await downloadGWA(S.project.lat,S.project.lon,S.project.hubHeight,S.project.z0);refresh()}catch(e){status('GWA failed: '+e.message)}};
$('btnGWAFile').onclick=async()=>{try{readProject();const f=$('gwaFile').files[0];if(!f)return alert('Choose GWA .lib/.gwc file');setGWAFromText(await f.text(),S.project.lat,S.project.lon,S.project.hubHeight,S.project.z0);refresh()}catch(e){alert('GWA import failed: '+e.message)}};
$('btnERA5').onclick=async()=>{try{readProject();await downloadERA5(S.project.lat,S.project.lon,$('eraYears').value);status(`ERA5 complete: ${S.era5.sp.length} records`);refresh()}catch(e){status('ERA5 failed: '+e.message)}};

$('btnMastClimate').onclick=async()=>{try{readProject();const f=$('mastClimateFile').files[0];if(!f)return alert('Choose LT mast climate CSV');parseMastClimate(await f.text(),{height:S.project.mastHeight,z0:S.project.z0});refresh()}catch(e){alert('Mast climate import failed: '+e.message)}};
$('btnMastTS').onclick=async()=>{try{readProject();const f=$('mastTsFile').files[0];if(!f)return alert('Choose mast/Windographer time-series CSV');parseMastTimeSeries(await f.text(),{height:S.project.mastHeight,source:f.name});refresh()}catch(e){alert('Mast time-series import failed: '+e.message)}};

$('btnObstacles').onclick=async()=>{try{const f=$('obstacleFile').files[0];if(!f)return alert('Choose obstacle CSV');parseObstacles(await f.text());refresh()}catch(e){alert('Obstacle import failed: '+e.message)}};
$('btnPowerCurve').onclick=async()=>{try{const f=$('pcFile').files[0];if(!f)return alert('Choose power/CT curve CSV');parsePowerCtCurve(await f.text());$('ratedKW').value=S.project.ratedKW;refresh()}catch(e){alert('Power curve import failed: '+e.message)}};

$('btnRun').onclick=()=>{try{readProject();runAEP();refresh()}catch(e){alert(e.message)}};
$('btnTS').onclick=()=>{try{readProject();runTimeSeries(1);refresh()}catch(e){alert(e.message)}};
$('btnCSV').onclick=()=>{try{exportTsCsv()}catch(e){alert(e.message)}};
$('btnReport').onclick=()=>{try{exportReport()}catch(e){alert(e.message)}};
});
