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
  if(!Number.isFinite(y1)||!Number.isFinite(y2)||y1>y2)throw Error('Invalid year range. Use format: 2021-2023');
  if(y2-y1>5)throw Error('Maximum 5 year range allowed (API limit)');

  const wsVar=height==='10m'?'wind_speed_10m':'wind_speed_100m';
  const wdVar=height==='10m'?'wind_direction_10m':'wind_direction_100m';

  // Determine the current year and month for ERA5T cut-off
  const now=new Date();
  const currentYear=now.getUTCFullYear();

  const sp=[],dir=[],time=[],temperatures=[],pressures=[];
  for(let y=y1;y<=y2;y++){
    let startDate,endDate;
    if(era5t&&y===currentYear){
      // ERA5T: up to current date (near-real-time)
      const mm=String(now.getUTCMonth()+1).padStart(2,'0');
      const dd=String(now.getUTCDate()).padStart(2,'0');
      startDate=`${y}-01-01`;
      endDate=`${y}-${mm}-${dd}`;
    } else if(era5t&&y<currentYear){
      // ERA5T for past years: full year from the forecast API
      startDate=`${y}-01-01`;
      endDate=`${y}-12-31`;
    } else {
      startDate=`${y}-01-01`;
      endDate=`${y}-12-31`;
    }

    // Choose API endpoint based on era5t flag
    // ERA5T uses the forecast API which provides near-real-time data
    // ERA5 historical uses the archive API
    const baseUrl=era5t
      ?'https://api.open-meteo.com/v1/forecast'
      :'https://archive-api.open-meteo.com/v1/archive';

    const url=`${baseUrl}?latitude=${lat}&longitude=${lon}&start_date=${startDate}&end_date=${endDate}&hourly=${wsVar},${wdVar},temperature_2m,surface_pressure&wind_speed_unit=ms&timezone=UTC`;

    log(`Downloading ${era5t?'ERA5T':'ERA5'} ${y} (${startDate} to ${endDate})…`,'i');

    // Retry with backoff
    let d;
    for(let attempt=0;attempt<4;attempt++){
      try{
        const r=await fetch(url);
        if(r.status===429){
          const wait=3000*(attempt+1);
          log(`${era5t?'ERA5T':'ERA5'} rate-limited (429), retrying in ${wait/1000}s…`,'w');
          await new Promise(res=>setTimeout(res,wait));
          continue;
        }
        if(r.status===400){
          const errBody=await r.text();
          throw Error(`Bad request for year ${y}: ${errBody.slice(0,200)}`);
        }
        if(!r.ok)throw Error(`${era5t?'ERA5T':'ERA5'} HTTP ${r.status}`);
        d=await r.json();
        break;
      }catch(e){
        if(attempt>=3)throw e;
        const wait=2000*(attempt+1);
        log(`${era5t?'ERA5T':'ERA5'} fetch error, retry ${attempt+1}: ${e.message}`,'w');
        await new Promise(res=>setTimeout(res,wait));
      }
    }

    const recs=d?.hourly?.time||[];
    const wsData=d?.hourly?.[wsVar];
    const wdData=d?.hourly?.[wdVar];

    if(!wsData||!wdData){
      log(`Warning: No ${era5t?'ERA5T':'ERA5'} data for ${y}. Available variables: ${Object.keys(d?.hourly||{}).join(', ')}`,'w');
      continue;
    }

    sp.push(...wsData);
    dir.push(...wdData);
    time.push(...recs);
    // Store temperature and pressure for air density correction
    const tempData=d?.hourly?.temperature_2m;
    const presData=d?.hourly?.surface_pressure;
    if(tempData)temperatures.push(...tempData);
    if(presData)pressures.push(...presData);
    log(`${era5t?'ERA5T':'ERA5'} ${y}: ${recs.length} records`);
  }

  if(!sp.length)throw Error('No ERA5 data downloaded. Check year range and parameters.');

  S.era5={sp,dir,time,temperatures,pressures,height:height==='10m'?10:100,source:era5t?'ERA5T / Open-Meteo':'ERA5 / Open-Meteo',era5t};
  // Compute average air density from temperature and pressure if available
  if(temperatures.length&&pressures.length){
    const validPairs=temperatures.filter((t,i)=>t!=null&&pressures[i]!=null);
    if(validPairs.length){
      const avgT=temperatures.reduce((s,v,i)=>v!=null&&pressures[i]!=null?s+v:s,0)/validPairs.length;
      const avgP=pressures.reduce((s,v,i)=>v!=null&&temperatures[i]!=null?s+v:s,0)/validPairs.length;
      S.era5.avgTemperature=avgT; // Kelvin
      S.era5.avgPressure=avgP; // hPa
      S.era5.avgAirDensity=avgP*100/(287.05*(avgT||288.15)); // rho = P/(R*T), P in Pa, T in K
      log(`ERA5 avg air density: ${S.era5.avgAirDensity.toFixed(3)} kg/m3 (T=${(avgT-273.15).toFixed(1)}C, P=${avgP.toFixed(1)} hPa)`);
    }
  }
  log(`${era5t?'ERA5T':'ERA5'} complete: ${sp.length.toLocaleString()} records`);
  return S.era5;
}
