import json
import os


def load_locations(data_dir, locations):
    """{location: records} from <data_dir>/<location>.json; the file key 'root_anchor' is renamed 'primary_anchor'."""
    data = {}
    for loc in locations:
        path = os.path.join(data_dir, f"{loc}.json")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        with open(path, encoding='utf-8') as f:
            data[loc] = json.load(f)
        for r in data[loc]:
            r['primary_anchor'] = r.pop('root_anchor', None)
        print(f"[LOAD] location {loc}: {len(data[loc])} records")
    return data
