#!/bin/bash
echo "Downloading models..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$GROQ_API_KEY')
" 2>/dev/null || true

# Telecharger les modeles si absents
if [ ! -f "models/model.pkl" ]; then
    pip install huggingface_hub -q
    python3 -c "
import urllib.request
import os
os.makedirs('models', exist_ok=True)
base = 'https://huggingface.co/spaces/doaxy/sensante/resolve/main/models/'
for f in ['model.pkl','encoder_sexe.pkl','encoder_region.pkl','feature_cols.pkl']:
    print(f'Downloading {f}...')
    urllib.request.urlretrieve(base+f, f'models/{f}')
    print(f'{f} done!')
"
fi

echo "Starting server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000
