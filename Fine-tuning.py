#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install datasets transformers')
get_ipython().system('pip install datasets transformers peft')
get_ipython().system('pip install transformers tensorflow')
get_ipython().system('pip install langchain transformers datasets peft tensorflow')
get_ipython().system('pip install langchain-community')


# In[ ]:


from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import pandas as pd
import torch


# In[ ]:


# Load dataset from CSV
def load_dataset_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    # Only keep the "story" column for fine-tuning
    data = data[["newstory"]]
    # Rename the column to "text" as expected by the tokenizer
    data = data.rename(columns={"newstory": "text"})
    dataset = Dataset.from_pandas(data)
    return dataset

# Fine-tuning the model using PEFT with LoRA
def fine_tune_model_with_peft(dataset, model_name="NousResearch/Llama-2-7b-chat-hf"):
    model_id = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, torch_dtype=torch.float16)

    # Load LoRA configuration and apply it to the model
    # lora_config = LoraConfig.from_pretrained('/content/drive/MyDrive/Story/FineTunedModel')
    # model = get_peft_model(model, lora_config)


    # Configure PEFT
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="/content/drive/MyDrive/Story/results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    trainer.train()

    return model

# Load csv
csv_file = "/content/drive/MyDrive/Story/stories_data.csv"
dataset = load_dataset_from_csv(csv_file)
fine_tuned_model = fine_tune_model_with_peft(dataset)

# Save the fine-tuned model
fine_tuned_model.save_pretrained("/content/drive/MyDrive/Story/fine_tuned_model")

