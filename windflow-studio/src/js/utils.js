export const $=id=>document.getElementById(id);export const rad=d=>d*Math.PI/180;export const deg=r=>r*180/Math.PI;
export function gamma(z){if(z<.5)return Math.PI/(Math.sin(Math.PI*z)*gamma(1-z));z-=1;const p=[.9999999999998099,676.5203681218851,-1259.1392167224028,771.3234287776531,-176.6150291621406,12.507343278686905,-.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];let x=p[0];for(let i=1;i<p.length;i++)x+=p[i]/(z+i);const t=z+7.5;return Math.sqrt(2*Math.PI)*t**(z+.5)*Math.exp(-t)*x}
export function weibullPdf(v,A,k){return v<=0||A<=0||k<=0?0:(k/A)*(v/A)**(k-1)*Math.exp(-(v/A)**k)}
export function meanWeibull(A,k){return A*gamma(1+1/k)}
export function powerDensity(A,k,rho=1.225){return .5*rho*A**3*gamma(1+3/k)}
export function logLaw(ws,fromH,toH,z0){z0=Math.max(0.0002,z0);return ws*Math.log(Math.max(toH,z0*1.1)/z0)/Math.log(Math.max(fromH,z0*1.1)/z0)}
export function latLonToXY(points){if(!points.length)return[];const lat0=points.reduce((s,p)=>s+p.lat,0)/points.length,lon0=points.reduce((s,p)=>s+p.lon,0)/points.length;const mx=111320*Math.cos(rad(lat0)),my=111320;return points.map(p=>({...p,x:(p.lon-lon0)*mx,y:(p.lat-lat0)*my}))}
export function download(name,text,type='text/plain'){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500)}
export function fmt(n,d=2){return Number.isFinite(n)?n.toFixed(d):'—'}
