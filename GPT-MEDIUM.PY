from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carica il modello addestrato e il tokenizer
model_name = "./gpt_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generazione di testo con parametri ottimizzati
input_text = "Tell me about the wolf and its habitat."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Genera una risposta con i parametri ottimizzati
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=150,  # Lunghezza massima della risposta
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones(input_ids.shape, dtype=torch.long),  # Attenzione per evitare problemi di padding
        temperature=0.7,  # Controlla la creatività del modello
        top_k=50,         # Limita il numero di parole candidate per ogni passaggio
        top_p=0.9,        # Controlla la diversità limitando il campionamento
        repetition_penalty=1.2  # Penalizza le ripetizioni
    )
    
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
