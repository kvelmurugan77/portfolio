$ErrorActionPreference = 'Stop'
$url = 'https://globalwindatlas.info/api/gwa/custom/Lib?lat=15.4525&long=75.8172'
$out = Join-Path $env:USERPROFILE (Join-Path 'Downloads' 'gwa_15.4525_75.8172.lib')
$headers = @{
  'X-Requested-With' = 'XMLHttpRequest'
  'Referer' = 'https://globalwindatlas.info/'
  'User-Agent' = 'Mozilla/5.0'
}
Invoke-WebRequest -Uri $url -Headers $headers -OutFile $out
Write-Host ("Saved: " + $out)
Get-Item $out | Format-List FullName, Length
