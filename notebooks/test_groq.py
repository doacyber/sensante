import os
from dotenv import load_dotenv
from groq import Groq

# Charger la cle API depuis .env
load_dotenv()

# Initialiser le client Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Appel test
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "Tu es un assistant médical qui explique les diagnostics en français simple."
        },
        {
            "role": "user",
            "content": "Le modèle a prédit : paludisme (72%). Explique en 2 phrases simples."
        }
    ],
    max_tokens=200
)

# Afficher la réponse
print("Réponse du LLM :")
print(response.choices[0].message.content)
