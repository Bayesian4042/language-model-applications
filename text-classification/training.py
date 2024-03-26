from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def get_dataset_using_csv(file_path, column_names, sep=","):
    return load_dataset("csv", data_files=file_path, split="train", column_names=column_names, sep=sep)

def visualize_dataset(dataset):
    # Looking at the class distribution of the dataset
    dataset.set_format(type="pandas")
    df = dataset["train"][:]
    df["label"].value_counts(ascending=True).plot.barh()
    plt.title("Frequence of Classes")
    plt.show()

    # How long are our resume sections to check if it is below our maximum context size
    df["words per section"] = df["text"].str.split().apply(len)
    df.boxplot(column="words per section", by="label", showfliers=False)
    plt.show()
    dataset.reset_format()

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def tokenize_dataset(dataset):
    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  f1 = f1_score(labels, preds, average="weighted")
  return {"accuracy": acc, "f1": f1}


def main():
    file_path = "data/resume_data.csv"
    column_names = ["text", "label"]
    num_labels = 6
    batch_size = 16
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset_using_csv(file_path, column_names)
    visualize_dataset(dataset)
    dataset_encoded = tokenize_dataset(dataset)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

    logging_steps = len(dataset_encoded["train"])
    model_name = "bert_finetuned_resume"

    training_args = TrainingArguments(
        output_dir=model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        disable_tqdm=False,
        push_to_hub=True,
        logging_steps=logging_steps,
        log_level="error",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    preds_output = trainer.predict(dataset_encoded["test"])

    print("Metrics: ", preds_output.metrics)