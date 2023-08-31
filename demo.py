import gradio as gr
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def classify_description(description):

    #model_name = "Donaldbassa/bert-classification-experience"

    model_name = "Donaldbassa/bert-classification-text"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(description, padding='max_length', truncation=True, max_length=256, return_tensors='pt')

    input_ids = inputs['input_ids']

    attention_mask = inputs['attention_mask']

 

    with torch.no_grad():

        outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits

        predicted_class = torch.argmax(logits, dim=1).item()

        return f"Classe prédite : {predicted_class}"

demo = gr.Interface(fn=classify_description, inputs="text", outputs="text")

demo.launch()