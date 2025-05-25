import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from pathlib import Path
import pickle

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
                print(f"Loading cached graphs from {cache_file}...")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            else:
                print("Cache outdated, reprocessing...")

        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")

        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)

        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))

        if use_cache:
            print(f"Saving processed graphs to cache: {cache_file}...")
            with open(cache_file, "wb") as f:
                pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cache saved! Next time loading will be much faster.")

        return graphs


def clear_cache(cache_dir="cache", pattern="*.pkl"):
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for cache_file in cache_path.glob(pattern):
            cache_file.unlink()
            print(f"Removed cache file: {cache_file}")


def get_cache_info(cache_dir="cache"):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print("No cache directory found")
        return

    cache_files = list(cache_path.glob("*.pkl")) + list(cache_path.glob("*.joblib"))
    if not cache_files:
        print("No cache files found")
        return

    print("Cache files:")
    total_size = 0
    for cache_file in cache_files:
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {cache_file.name}: {size_mb:.1f} MB")
    print(f"Total cache size: {total_size:.1f} MB")


def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)