import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from transformers import TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer
import numpy as np 




df = pd.read_csv("preprocessed_dataset.csv")
print(df.columns)
print(df['label'].value_counts()) 


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(train_texts))
print("Testing samples:", len(test_texts))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)


print("Example tokenized input:")
print(train_encodings['input_ids'][0])



class HealthMisinformationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }


train_dataset = HealthMisinformationDataset(train_encodings, train_labels)
test_dataset = HealthMisinformationDataset(test_encodings, test_labels)


print(train_dataset[0])



model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments(
    output_dir="./results",           
    num_train_epochs=3,               
    per_device_train_batch_size=16,   
    per_device_eval_batch_size=64,    
    warmup_steps=10,                  
    weight_decay=0.01,                
    logging_dir="./logs",            
    # evaluation_strategy="epoch",     
)



class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss



trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.evaluate()
eval_results = trainer.evaluate()
print(eval_results)
