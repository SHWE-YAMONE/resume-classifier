import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import evaluate
import multiprocessing
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

from src.logger import setup_logger
from src.utils import save_label_encoder
from src.callbacks import MetricsPlotCallback
from src.augment import augment_data

logger = setup_logger("TrainLogger")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResumeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def train_with_cv(data_path="data/Resume.csv", model_name="microsoft/deberta-v3-large", output_dir="models", num_folds=5):
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path).dropna(subset=['Resume', 'Category'])
    df = df[df['Resume'].str.len() > 100]  
    logger.info(f"Loaded {len(df)} samples")

    logger.info("Starting augmentation")
    df = augment_data(df, augment_frac=0.1)
    logger.info(f"After augmentation: {len(df)} samples")

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Category'])
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'].to_numpy())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['Resume'], df['label'])):
        logger.info(f"Starting Fold {fold + 1}")

        train_texts = df['Resume'].iloc[train_idx].tolist()
        val_texts = df['Resume'].iloc[val_idx].tolist()
        train_labels = df['label'].iloc[train_idx].tolist()
        val_labels = df['label'].iloc[val_idx].tolist()

        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        train_dataset = ResumeDataset(train_encodings, train_labels)
        val_dataset = ResumeDataset(val_encodings, val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_encoder.classes_),
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3
        ).to(device)

        fold_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=1e-5,
            warmup_ratio=0.06,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_dir=os.path.join(fold_dir, "logs"),
            logging_strategy="steps",
            logging_steps=10
        )

        callbacks = [
            MetricsPlotCallback(save_dir=os.path.join(fold_dir, "plots")),
            EarlyStoppingCallback(early_stopping_patience=2)
        ]

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            class_weights=class_weights_tensor
        )


        trainer.train()
        fold_metrics = trainer.evaluate()
        all_metrics.append(fold_metrics)

        predictions = trainer.predict(val_dataset).predictions
        y_pred = np.argmax(predictions, axis=1)

        plot_confusion_matrix(val_labels, y_pred, label_encoder.classes_, os.path.join(fold_dir, "confusion_matrix.png"))

        model.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)

    avg_metrics = {
        "accuracy": float(np.mean([m["eval_accuracy"] for m in all_metrics])),
        "f1": float(np.mean([m["eval_f1"] for m in all_metrics]))
    }

    logger.info("Average metrics across folds:")
    logger.info(avg_metrics)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(avg_metrics, f)

    save_label_encoder(label_encoder, "outputs/label_classes.json")