import os
import json
import random
import numpy as np
from collections import Counter
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class HealthModel:
    def __init__(self, model_dir: str):
        with open(f"{model_dir}/label2id.json", "r", encoding="utf-8") as f:
            self.label2id = json.load(f)
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.top_k = 5
    def predict(self, query: str):
        inputs = tokenizer(query, return_tensors="pt", padding="max_length", truncation=True, max_length=192).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
        top_indices = probs.argsort()[-self.top_k:][::-1]
        top_labels = [self.id2label[idx] for idx in top_indices]
        top_probs = [probs[idx] for idx in top_indices]
        
        best_label = top_labels[0]
        best_prob = top_probs[0]
        
        return best_label, best_prob, list(zip(top_labels, top_probs))
    


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro}
    

def train_health_model(outer_dir: str):
    CSV_PATH = "/content/train_v1.csv"
    BASE_MODEL = "ai-forever/ruBert-large"
    MAX_LEN = 192
    BATCH_SIZE = 16
    EPOCHS = 6
    LR = 2e-5
    WARMUP_RATIO = 0.1
    SEED = 42
    OVERSAMPLE = False
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(outer_dir, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df.columns = ["idx", "symptoms", "code"]

    df["symptoms"] = df["symptoms"].astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()
    df = df[df["symptoms"] != ""]

    LABEL_JSON_PATH = os.path.join(outer_dir, "label2id.json")
    if os.path.exists(LABEL_JSON_PATH):
        with open(LABEL_JSON_PATH, "r", encoding="utf-8") as f:
            label2id = json.load(f)
    else:
        unique_codes = sorted(df["code"].unique())
        label2id = {c: i for i, c in enumerate(unique_codes)}
        with open(LABEL_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(label2id, f, ensure_ascii=False, indent=2)

    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    df["label"] = df["code"].map(label2id)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["symptoms"].tolist(),
        df["label"].tolist(),
        test_size=0.1,
        random_state=SEED,
        stratify=df["label"].tolist()
    )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device)

    total_steps_est = (len(train_ds) // BATCH_SIZE + 1) * EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps_est))

    training_args = TrainingArguments(
        outer_dir=outer_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        seed=SEED,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(outer_dir)
    tokenizer.save_pretrained(outer_dir)
    with open(os.path.join(outer_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print("Training finished. Model saved to:", outer_dir)
