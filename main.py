import argparse
import torch
import gzip
import json
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import os
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import logging
from colorlog import ColoredFormatter
from pathlib import Path
from torch_geometric.data import Dataset, Data
import pickle

# ========================================================================================================================= #
# ================================================ UNIONE FILE loadDATA =================================================== #
# ========================================================================================================================= #


class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path, use_cache=True, cache_dir="cache"):

        cache_dir = os.path.join(os.path.dirname(path),cache_dir)
        original_file = Path(path)
        cache_file = Path(cache_dir) / f"{original_file.stem}.pkl"
        cache_file.parent.mkdir(exist_ok=True)  #CREA DIRECTORY CACHE SE NON ESISTE

        if use_cache and cache_file.exists():
            cache_time = cache_file.stat().st_mtime
            original_time = original_file.stat().st_mtime

            if cache_time > original_time:
                log.debug(f"Loading cached graphs from {cache_file}...")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            else:
                log.debug("Cache outdated, reprocessing...")

        log.debug(f"Loading graphs from {path}...")
        log.debug("This may take a few minutes, please wait...")

        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)

        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))

        if use_cache:
            log.debug(f"Saving processed graphs to cache: {cache_file}...")
            with open(cache_file, "wb") as f:
                pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.debug(f"Cache saved! Next time loading will be much faster.")

        return graphs


def clear_cache(cache_dir="cache", pattern="*.pkl"):
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for cache_file in cache_path.glob(pattern):
            cache_file.unlink()
            log.debug(f"Removed cache file: {cache_file}")


def get_cache_info(cache_dir="cache"):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        log.debug("No cache directory found")
        return

    cache_files = list(cache_path.glob("*.pkl")) + list(cache_path.glob("*.joblib"))
    if not cache_files:
        log.debug("No cache files found")
        return

    log.debug("Cache files:")
    total_size = 0
    for cache_file in cache_files:
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        log.debug(f"  {cache_file.name}: {size_mb:.1f} MB")
    log.debug(f"Total cache size: {total_size:.1f} MB")


def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)



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






