import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchmetrics import F1Score, Precision, Recall
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from distutils.util import strtobool
from pytorch_lightning.loggers import WandbLogger
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLPClassifier on activation data.")
    parser.add_argument("--folder", type=str, default="data/02-processed/activation_tracker.pkl", help="Path to the data folder.")
    parser.add_argument("--object_of_study", type=str, default="mlp_act", choices=["mlp_act", "states"], help="Object of study.")
    parser.add_argument("--layer", type=int, default=1, help="Layer to analyze.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size.")
    parser.add_argument("--inner_size", type=int, default=128, help="Inner layer size.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument("--use_wandb", type=lambda x: bool(strtobool(x)), default=False, help="Use Weights & Biases for logging.")
    return parser.parse_args()

# Paso 1: Crear el Dataset personalizado
class ActivationDataset(Dataset):
    def __init__(self, activations, tasks):
        self.activations = torch.tensor(activations, dtype=torch.float32)
        self.tasks = torch.tensor(tasks, dtype=torch.long)  # asumiendo que las tareas son clases etiquetadas con enteros

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.tasks[idx]
    

# Paso 2: Crear el DataLoader
def create_dataloaders(activations, tasks, batch_size=8, val_split=0.1):
    dataset = ActivationDataset(activations, tasks)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    print(f"Train dataset size: {len(train_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_loader, val_loader

class MLPClassifier(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_size = 512, inner_size = 128, lr = 1e-3):
        super(MLPClassifier, self).__init__()
        self.lr  = lr
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, inner_size)
        self.fc3 = nn.Linear(inner_size, num_classes)
        task     = "classification" if num_classes > 2 else "binary"
        
        # Métricas
        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.val_acc   = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.test_acc  = torchmetrics.Accuracy(task=task, num_classes=num_classes)

        self.test_f1 = F1Score(num_classes=num_classes, average='macro', task=task)
        self.test_precision = Precision(num_classes=num_classes, average='macro', task=task)
        self.test_recall = Recall(num_classes=num_classes, average='macro', task=task)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Training Step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Calcular métricas
        acc = self.train_acc(preds, y)

        # Log de métricas
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # Validation Step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Calcular métricas
        acc = self.val_acc(preds, y)

        # Log de métricas
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # Test Step (usar el mismo conjunto que validación para calcular los resultados)
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Calcular métricas
        acc = self.test_acc(preds, y)
        f1 = self.test_f1(preds, y)
        precision = self.test_precision(preds, y)
        recall = self.test_recall(preds, y)

        # Log de métricas
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)

        return loss

    # Optimizer configuration
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    # Load the dataset

    args = parse_args()
    pl.seed_everything(args.seed)

    if args.use_wandb:
        wandb_logger = WandbLogger(project="llama-3.2-1B-multilingual-interpretability", name=f"{args.object_of_study}_layer_{args.layer}")
        wandb_logger.log_hyperparams({"hidden_size": args.hidden_size, "lr": args.lr, "val_split": args.val_split})

    dataset = pd.read_pickle(args.folder)
    column = f"{args.object_of_study}_{args.layer}"
    activations = np.vstack(dataset[column].tolist())
    tasks = dataset["language"].astype("category").cat.codes.values
    y_task = np.concatenate([np.full(array.shape[0], label) for array, label in zip(dataset[column].tolist(), tasks)])
    names_tasks = dataset["language"].tolist()
    y_task_names = np.concatenate([np.full(array.shape[0], label) for array, label in zip(dataset[column].tolist(), names_tasks)])
    print(f"Activations shape: {activations.shape}")
    print(f"Tasks shape: {y_task.shape}")

    # Paso 3: Crear los dataloaders
    train_loader, val_loader = create_dataloaders(activations, y_task, batch_size=args.batch_size, val_split=args.val_split)
    print("DataLoaders created")
    model = MLPClassifier(input_size=activations.shape[1], num_classes=len(np.unique(y_task)), hidden_size=args.hidden_size, lr=args.lr)
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger if args.use_wandb else None)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=val_loader)

    # tsne of the activations
    tsne = TSNE(n_components=2, random_state=args.seed)
    activations_tsne = tsne.fit_transform(activations)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=activations_tsne[:, 0], y=activations_tsne[:, 1], hue=y_task_names, palette="tab10")
    plt.title(f"TSNE of {args.object_of_study} activations at layer {args.layer}")
    # to wandb
    if args.use_wandb:
        wandb.log({"tsne_plot": wandb.Image(plt)})
    
    #save plot
    os.makedirs("plots", exist_ok=True)
    os.makedirs(f"plots/{args.object_of_study}", exist_ok=True)
    plt.savefig(f"plots/{args.object_of_study}/{args.object_of_study}_layer_{args.layer}_tsne.png")
    plt.close()
    
    


