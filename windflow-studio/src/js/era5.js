import {S,log} from './state.js';

/**
 * Download ERA5 or ERA5T time-series data from Open-Meteo API.
 * @param {number} lat - Latitude
 * @param {number} lon - Longitude
 * @param {string} years - Year range e.g. '2021-2023'
 * @param {string} height - '100m' or '10m'
 * @param {boolean} era5t - If true, use near-real-time ERA5T endpoint
 */
export async function downloadERA5(lat,lon,years,height='100m',era5t=false){
  const [y1,y2]=years.split('-').map(Number);
  const wsVar=height==='10m'?'wind_speed_10m':'wind_speed_100m';
  const wdVar=height==='10m'?'wind_direction_10m':'wind_direction_100m';

  // Determine the current year and month for ERA5T cut-off
  const now=new Date();
  const currentYear=now.getUTCFullYear();
  const currentMonth=String(now.getUTCMonth()+1).padStart(2,'0');

  const sp=[],dir=[],time=[];
  for(let y=y1;y<=y2;y++){
    let endDate;
    if(era5t&&y===currentYear){
      // ERA5T: up to current date (near-real-time)
      endDate=`${y}-${currentMonth}-${String(now.getUTCDate()).padStart(2,'0')}`;
    } else {
      endDate=`${y}-12-31`;
    }
    const startDate=`${y}-01-01`;

    // Choose API endpoint based on era5t flag
    let baseUrl;
    if(era5t){
      // ERA5T uses the forecast API which provides near-real-time data
      baseUrl='https://api.open-meteo.com/v1/forecast';
    } else {
      baseUrl='https://archive-api.open-meteo.com/v1/archive';
    }

    const url=`${baseUrl}?latitude=${lat}&longitude=${lon}&start_date=${startDate}&end_date=${endDate}&hourly=${wsVar},${wdVar},temperature_2m,surface_pressure&wind_speed_unit=ms&timezone=UTC`;

    // Retry with backoff
    let d;
    for(let attempt=0;attempt<4;attempt++){
      try{
        const r=await fetch(url);
        if(r.status===429){
          const wait=2000*(attempt+1);
          log(`${era5t?'ERA5T':'ERA5'} rate-limited (429), retrying in ${wait/1000}s…`,'w');
          await new Promise(res=>setTimeout(res,wait));
          continue;
        }
        if(!r.ok)throw Error(`${era5t?'ERA5T':'ERA5'} HTTP ${r.status}`);
        d=await r.json();
        break;
      }catch(e){
        if(attempt>=3)throw e;
        const wait=1500*(attempt+1);
        log(`${era5t?'ERA5T':'ERA5'} fetch error, retry ${attempt+1}: ${e.message}`,'w');
        await new Promise(res=>setTimeout(res,wait));
      }
    }

    const recs=d?.hourly?.time||[];
    sp.push(...(d?.hourly?.[wsVar]||[]));
    dir.push(...(d?.hourly?.[wdVar]||[]));
    time.push(...recs);
    log(`${era5t?'ERA5T':'ERA5'} ${y}: ${recs.length} records`);
  }

  S.era5={sp,dir,time,height:height==='10m'?10:100,source:era5t?'ERA5T / Open-Meteo':'ERA5 / Open-Meteo',era5t};
  log(`${era5t?'ERA5T':'ERA5'} complete: ${sp.length.toLocaleString()} records`);
  return S.era5;
}
