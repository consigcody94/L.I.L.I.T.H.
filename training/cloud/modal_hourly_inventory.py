"""Resolve the wider CONUS automated network against NOAA's hourly catalog on Modal."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/source')
app = modal.App('lilith-conus-hourly-inventory')
volume = modal.Volume.from_name('lilith-verified-research')
image = modal.Image.debian_slim(python_version='3.11').pip_install('numpy==2.2.6')
if modal.is_local():
    image = image.add_local_dir(ROOT/'training','/source/training',ignore=['__pycache__','cloud'])
    image = image.add_local_dir(ROOT/'data','/source/data',ignore=['raw','processed','storage','__pycache__'])


@app.function(image=image,cpu=1,memory=2048,timeout=300,max_containers=1,retries=0,
              volumes={'/artifacts':volume})
def inventory():
    import sys,csv,io,json,hashlib
    from collections import Counter,defaultdict
    from urllib.request import Request,urlopen
    from datetime import datetime,timezone
    sys.path.insert(0,'/source')
    from training.conus_catalog import homr_rows,distance_km
    root = Path('/artifacts/conus-preparation/conus-asos-awos-20261014-v1')
    export = root/'hourly-inventory'
    if (export/'HOURLY_CROSSWALK.json').exists():
        raise FileExistsError('Completed inventory is immutable')
    export.mkdir(exist_ok=True)
    url = 'https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.csv'
    with urlopen(Request(url,headers={'User-Agent':'Lilith-research/0.2'}),timeout=60) as response:
        raw = response.read()
    (export/'ghcnh-station-list.csv').write_bytes(raw)
    hourly = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    by_id = {s['GHCN_ID']:s for s in hourly}
    by_icao = defaultdict(list)
    for station in hourly:
        if station['ICAO'].strip():
            by_icao[station['ICAO'].strip()].append(station)
    links,unmatched = [],[]
    for network in ['ASOS','AWOS']:
        text = (root/'catalog'/(network.lower()+'-stations.txt')).read_text()
        for row in homr_rows(text,network):
            candidates = []
            for sid,method in [(row['ghcn_explicit'],'explicit HOMR identifier'),
                               ('USW000'+row['wban'],'WBAN identifier')]:
                if sid in by_id:
                    candidates.append((by_id[sid],method))
            for call in {row['call'],'K'+row['call']}:
                for station in by_icao.get(call,[]):
                    candidates.append((station,'ICAO catalog identity'))
            accepted = None
            for station,method in candidates:
                try:
                    separation = distance_km(row['homr_latitude'],row['homr_longitude'],
                                             float(station['LATITUDE']),float(station['LONGITUDE']))
                except ValueError:
                    continue
                if separation <= 10:
                    accepted = {**row,'ghcnh_id':station['GHCN_ID'],'hourly_name':station['NAME'],
                                'hourly_icao':station['ICAO'],'distance_km':separation,'method':method}
                    break
            (links if accepted else unmatched).append(accepted or {**row,'reason':'No identifier/ICAO match within 10 km; needs authority review'})
    result = {'created_at':datetime.now(timezone.utc).isoformat(),'scope':'CONUS','provider':'Modal',
              'catalog_url':url,'catalog_sha256':hashlib.sha256(raw).hexdigest(),
              'matched_registry_entries_by_network':dict(Counter(s['network'] for s in links)),
              'unique_hourly_station_ids':len({s['ghcnh_id'] for s in links}),
              'unmatched_registry_entries':len(unmatched),'links':links,'unmatched':unmatched,
              'status':'catalog_crosswalk_only_not_hourly_ingestion_or_training',
              'limits':['Current catalog identity and proximity are not proof of interval/history/variable coverage',
                        'No nearest-neighbor-only identity guessing; unmatched entries retained',
                        'No hourly measurements downloaded, model fit or station-skill claim in this inventory']}
    assert all(s['distance_km']<=10 for s in links)
    (export/'HOURLY_CROSSWALK.json').write_text(json.dumps(result,indent=2)+'\n')
    (export/'CHECKSUMS.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'
        for p in sorted(export.iterdir()) if p.is_file() and p.name!='CHECKSUMS.sha256'))
    volume.commit()
    return {k:result[k] for k in ['matched_registry_entries_by_network','unique_hourly_station_ids','unmatched_registry_entries','status']}


@app.local_entrypoint()
def main():
    print(inventory.remote())
