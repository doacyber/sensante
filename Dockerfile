# Dockerfile - SenSante
FROM python:3.11-slim

WORKDIR /app

# Installer les dependances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Telecharger les modeles depuis HF au demarrage
RUN pip install --no-cache-dir huggingface_hub && \
    python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('models', exist_ok=True)
for f in ['model.pkl', 'encoder_sexe.pkl', 'encoder_region.pkl', 'feature_cols.pkl']:
    hf_hub_download(repo_id='doaxy/sensante', filename=f'models/{f}', repo_type='space', local_dir='.')
    print(f'{f} downloaded!')
"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
