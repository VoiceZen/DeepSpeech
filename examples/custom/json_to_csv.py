import codecs
import pandas as pd
import json

manifest_path='/home/vz/Users/ak47/projects/DeepSpeech/data/tiny/manifest.test-clean'
manifest = []
for json_line in codecs.open(manifest_path, 'r', 'utf-8'):
    try:
        json_data = json.loads(json_line)
    except Exception as e:
        raise IOError("Error reading manifest: %s" % str(e))
    
    manifest.append(json_data)

pd.DataFrame(manifest).to_csv("/home/vz/Users/ak47/projects/DeepSpeech/data/tiny/manifest.test-clean.csv",index=False)
    