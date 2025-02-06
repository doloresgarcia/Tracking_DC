import wandb
import random
import pytorch_lightning as L

# Crea il logger di wandb compatibile con PyTorch Lightning
wandb_logger = L.loggers.WandbLogger(
    project="my-awesome-project",   # Nome del progetto
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)

# Simula l'allenamento
epochs = 10
offset = random.random() / 5

# Inizia un run di PyTorch Lightning, specificando il logger
trainer = L.Trainer(
    logger=wandb_logger,
    max_epochs=epochs,
)

# Simula l'allenamento e logga le metriche con il logger di wandb
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # Logga le metriche a wandb
    wandb_logger.log_metrics({"acc": acc, "loss": loss}, step=epoch)

# In questo caso, il logger `wandb_logger` si occuperà di chiudere il run automaticamente
# Non è necessario usare wandb.finish() quando usi il logger in PyTorch Lightning.
