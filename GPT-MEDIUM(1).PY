from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset

# Carica il modello e il tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Aggiungi un pad_token se non è già presente
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Testo per l'addestramento
testo = """
The wolf (Canis lupus) is the largest carnivore in the dog family (Canidae). Wolves usually live in packs, which are family groups.
In most regions, their main prey is medium to large ungulates. The species was distributed in several subspecies throughout Europe,
much of Asia, including the Arabian Peninsula and Japan, and in North America since the late Pleistocene.

Wolves were systematically persecuted in Central Europe from the 15th century onwards. In the 19th century they were decimated in almost all regions of their global range,
mainly through human hunting, and were almost completely extinct in Western and Central Europe and in Japan.
Since the 1980s the wolf has been protected in many countries. In European countries by the Bern Convention, and in the EU since 1992 additionally by the Fauna-Flora-Habitat Directive.
In many countries, including the Middle East, there is no legal protection for the wolf.

Since the turn of the millennium, the number of wolves and wolf packs in Central and Northern Europe has increased significantly again.
For the 2020/21 recording period, 157 packs, 27 pairs and 19 territorial individuals were registered among wolves in Germany, living in 203 wolf territories.
Wolves are among the best-known predators; they have found their way into the myths and fairy tales of many peoples since early on.
They are the ancestral form of all domestic dogs and the secondarily wild dingo: dogs and wolves are not separated by a species barrier and can therefore produce reproductive offspring with each other.

Today, an increasing admixture of dog genes is observed worldwide through hybridization. The smallest number is found in the populations of the Tibetan plateau and Scandinavia.
"""

# Creare un dataset per l'addestramento
dataset = Dataset.from_dict({"text": [testo]})

# Pre-processare il testo con `labels`
def tokenize_function(examples):
    encoding = tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True)
    encoding["labels"] = encoding["input_ids"].clone()  # Imposta le etichette uguali agli input
    return encoding

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Configurazione dell'addestramento
training_args = TrainingArguments(
    output_dir="./gpt_finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
)

# Configurazione del Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Avvia l'addestramento
trainer.train()

# Salva il modello e il tokenizer addestrati
model.save_pretrained("./gpt_finetuned")
tokenizer.save_pretrained("./gpt_finetuned")

print("Addestramento completato e modello salvato in './gpt_finetuned'")
