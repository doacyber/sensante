#!/bin/bash
echo "Downloading models..."
mkdir -p models
python3 -c "
from huggingface_hub import hf_hub_download
import shutil
files = ['model.pkl', 'encoder_sexe.pkl', 'encoder_region.pkl', 'feature_cols.pkl']
for f in files:
    path = hf_hub_download(repo_id='doaxy/sensante-models', filename=f, repo_type='model')
    shutil.copy(path, f'models/{f}')
    print(f'{f} OK!')
print('Models ready!')
"
echo "Starting server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000
