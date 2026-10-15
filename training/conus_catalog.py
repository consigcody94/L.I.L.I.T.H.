"""NOAA HOMR ASOS/AWOS registry to explicit, audited CONUS GHCN identifiers."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from data.providers.ghcn_records import download, station_metadata

CONUS_STATES = set('AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC'.split())


def homr_rows(text, network):
    for line in text.splitlines()[2:]:
        if not line[:8].isdigit():
            continue
        asos = network == 'ASOS'
        country = line[89:109] if asos else line[58:78]
        state = (line[110:112] if asos else line[79:81]).strip()
        if country.strip() != 'UNITED STATES' or state not in CONUS_STATES:
            continue
        try:
            lat = float(line[144:153] if asos else line[113:122])
            lon = float(line[154:164] if asos else line[123:133])
        except ValueError:
            continue
        yield {'ncdc_id':line[:8], 'wban':line[9:14].strip(), 'coop_id':line[15:21].strip(),
               'call':line[22:26].strip(), 'homr_name':line[27:57].strip(),
               'network':network, 'state':state, 'homr_latitude':lat, 'homr_longitude':lon,
               'ghcn_explicit':line[238:249].strip() if asos else '',
               'asos_commissioned':line[229:237].strip() if asos else None}


def distance_km(lat0, lon0, lat1, lon1):
    a,b,c,d = np.radians([lat0,lon0,lat1,lon1])
    return float(6371*2*np.arcsin(np.sqrt(np.sin((a-c)/2)**2+np.cos(a)*np.cos(c)*np.sin((b-d)/2)**2)))


def build(directory: Path, cache: Path):
    directory.mkdir(parents=True,exist_ok=True)
    metadata_text, metadata_hash = download('ghcnd-stations.txt',cache)
    metadata = station_metadata(metadata_text)
    hashes, rows = {}, []
    for network in ['ASOS','AWOS']:
        for filename in [network.lower()+'-stations.txt', network+'_Table.txt']:
            path = directory/filename
            if not path.exists():
                url = 'https://www.ncei.noaa.gov/access/homr/file/'+filename
                with urlopen(Request(url,headers={'User-Agent':'Lilith-research/0.2'}),timeout=60) as response:
                    path.write_bytes(response.read())
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend(homr_rows((directory/(network.lower()+'-stations.txt')).read_text(),network))
    selected, unmatched = {}, []
    for row in rows:
        candidates = []
        if row['ghcn_explicit']:
            candidates.append((row['ghcn_explicit'],'HOMR explicit GHCND'))
        if row['wban'].isdigit():
            candidates.append(('USW000'+row['wban'],'WBAN identifier plus coordinate crosscheck'))
        if row['coop_id'].isdigit():
            candidates.append(('USC00'+row['coop_id'],'COOP identifier plus coordinate crosscheck'))
        accepted = None
        for sid,method in candidates:
            if sid not in metadata:
                continue
            station = metadata[sid]
            separation = distance_km(row['homr_latitude'],row['homr_longitude'],station['latitude'],station['longitude'])
            if separation <= 10:
                accepted = (sid,method,separation); break
        if accepted is None:
            unmatched.append({**row,'reason':'No GHCN catalog identity within 10 km; retain for hourly acquisition'}); continue
        sid,method,separation = accepted
        station = selected.setdefault(sid,{**metadata[sid], 'state':row['state'], 'networks':[], 'homr_links':[],
                'station_holdout':int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%5 == 0})
        if row['network'] not in station['networks']:
            station['networks'].append(row['network'])
        station['homr_links'].append({**row,'matching_method':method,'catalog_distance_km':separation})
    result = {'created_at':datetime.now(timezone.utc).isoformat(),'scope':'CONUS',
              'selection_policy':'All current NOAA HOMR ASOS/AWOS entries in 48 contiguous states + DC with cataloged GHCN identity and <=10km coordinate crosscheck; no random station cap',
              'registry_url':'https://www.ncei.noaa.gov/access/homr/reports','registry_sha256':hashes,
              'ghcn_catalog_sha256':metadata_hash,'registry_counts_by_network':dict(Counter(r['network'] for r in rows)),
              'unique_ghcn_stations':len(selected),'stations':sorted(selected.values(),key=lambda s:s['id']),
              'unmatched_registry_entries':unmatched,'unmatched_count':len(unmatched),
              'limits':['Current registry is not historical instrument identity; pre-ASOS history may use other observing systems',
                        'AWOS GHCN mapping uses WBAN/COOP identity and coordinate check, not an explicit GHCND field',
                        'No GHCN daily match does not mean no observations; use GHCNh/METAR crosswalk next',
                        'Current state membership defines CONUS including coastal facilities; no coastline mask applied',
                        'Catalog identity does not guarantee complete historical or 2026 measurements']}
    (directory/'CONUS_COHORT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);args=parser.parse_args()
    result = build(args.output,args.cache)
    print(json.dumps({k:result[k] for k in ['registry_counts_by_network','unique_ghcn_stations','unmatched_count']}))
