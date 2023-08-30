import torch
import pandas as pd
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from huggingface_hub import notebook_login


import wandb #monitoring

config = {
    "model_name":"bert-base-uncased",
    "csv_file":'datas/fake_job_postings.csv',
    "learning_rate":2e-5,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_epochs":1,
    "num_classes":8,
    "max_length":128,
    "batch_size":2
}
MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 8

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        description = str(self.data.iloc[idx]['description'])
        label = int(self.data.iloc[idx]['required_experience'])

        encoding = self.tokenizer(description, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

class CustomModel:
    def __init__(self, model_name, num_classes, max_length):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def load_and_preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['description'])

        experience_mapping = {
            'Internship': 0,
            'Not Applicable': 1,
            'Mid-Senior level': 2,
            'Associate': 3,
            'Entry level': 4,
            'Executive': 5,
            'Director': 6
        }

        df['required_experience'] = df['required_experience'].fillna('Not Applicable')
        df['required_experience'] = df['required_experience'].map(experience_mapping)
        return df

    def train(self, train_file_path, num_epochs=config['num_epochs'], batch_size=16, learning_rate=config['learning_rate']):

        wandb.init(project="bert-classification")

        df = self.load_and_preprocess_data(train_file_path)
        train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

        train_dataset = CustomDataset(data=train_data, tokenizer=self.tokenizer, max_length=self.max_length)
        val_dataset = CustomDataset(data=val_data, tokenizer=self.tokenizer, max_length=self.max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.to(config['device'])

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to((config['device']))
                attention_mask = batch['attention_mask'].to((config['device']))
                labels = batch['label'].to((config['device']))

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({'Train Loss': loss.item()})

            avg_train_loss = total_loss / len(train_dataloader)

            self.evaluation(val_dataloader, config['device'])

            print(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")

            # Validation
            self.model.eval()
            val_accuracy = 0
            num_val_batches = 0

            for batch in val_dataloader:
                input_ids = batch['input_ids'].to((config['device']))
                attention_mask = batch['attention_mask'].to((config['device']))
                labels = batch['label'].to((config['device']))

                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    predictions = torch.argmax(logits, dim=1)
                    val_accuracy += torch.sum(predictions == labels).item()
                    num_val_batches += len(labels)

            avg_val_accuracy = val_accuracy / num_val_batches
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {avg_val_accuracy:.4f}")

            wandb.log({
                "avg_loss":avg_train_loss,
                "avg_acuracy":avg_val_accuracy
                })

        # Sauvegarde du modèle
        #self.model.save_pretrained('trained_model')
        torch.save(self.model, 'bert-model.pth')

        #push to huggingface hub

        #tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        notebook_login()
        self.model.push_to_hub("bert-classification")

    def evaluation(self, dataloader, device):
        self.model.eval()
        num_correct = 0
        total_examples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(config['device'])
                attention_mask = batch['attention_mask'].to(config['device'])
                labels = batch['label'].to(config['device'])

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                predictions = torch.argmax(logits, dim=1)
                num_correct += torch.sum(predictions == labels).item()
                total_examples += len(labels)

        accuracy = num_correct / total_examples
        print(f"Validation Accuracy: {accuracy:.4f}")


def main():

    custom_model = CustomModel(config['model_name'], config['num_classes'], config['max_length'])

    custom_model.train(config['csv_file'], num_epochs=config['num_epochs'], batch_size=config['batch_size'], learning_rate=config['learning_rate'])


if __name__ == "__main__":

    main()