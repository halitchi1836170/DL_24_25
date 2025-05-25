import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from loadData import GraphDataset
import os
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import logging
from colorlog import ColoredFormatter

# ========================================================================================================================= #
# ======================================================== SET UPS ======================================================== #
# ========================================================================================================================= #

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(asctime)s - %(levelname) - s%(reset)s | %(log_color) - s%(message) - s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ========================================================================================================================= #
# ======================================================= CONSTANTS ======================================================= #
# ========================================================================================================================= #

num_epochs = 5
patience = 15
use_coteaching= True
charsIntestazione = 150
input_dim = 300
hidden_dim = 256
output_dim = 6
num_layers = 5
num_heads = 10
dropout = 0.1

log = logging.getLogger('GCNL-Main')
log.setLevel(LOG_LEVEL)
log.addHandler(stream)



# ========================================================================================================================= #
# ======================================================= FUNCTIONS ======================================================= #
# ========================================================================================================================= #
def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def Intestazione(stringaTBP):
    strlenght = len(stringaTBP)
    return ("="*int((charsIntestazione-strlenght)/2)+stringaTBP+"="*int((charsIntestazione-strlenght)/2))

def main_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}\n")

    log.info("Loading test dataset...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    log.info("Done \n")

    return "",""

def main(args):
    log.info(Intestazione("STARTING TRAINING"))
    log.info("Starting training with Graph Transformer")
    log.info(f"Arguments: {vars(args)}\n")

    best_models, best_accuracies = main_training(args)

    if best_accuracies:
        log.info(Intestazione("FINAL RESULTS"))
        for config, acc in best_accuracies.items():
            log.info(f"{config}: {acc:.4f}")



# ========================================================================================================================= #
# ========================================================= MAIN ========================================================== #
# ========================================================================================================================= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Transformer with multiple loss functions for noisy labels")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional)")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset")
    #parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    #parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    #parser.add_argument("--use_coteaching", action="store_true", help="Enable co-teaching (resource intensive)")

    args = parser.parse_args()
    main(args)






