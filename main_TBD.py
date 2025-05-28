import argparse
import torch
import gzip
import json
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool
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
from datetime import datetime
import math
import matplotlib.pyplot as plt
from torch_geometric.nn import Node2Vec
from sklearn.metrics import f1_score


########################################################################################################################
#                                                       CONSTANTS
########################################################################################################################

train_path = "C:\\Users\\andrei.halitchi\\OneDrive - Zenith Global S.p.a\\Desktop\\ANDREI\\DOCUMENTI\\UNI\\PROJS\\DL_24_25\\datasets\\A\\train.json.gz"
#train_path=""
test_path = "C:\\Users\\andrei.halitchi\\OneDrive - Zenith Global S.p.a\\Desktop\\ANDREI\\DOCUMENTI\\UNI\\PROJS\\DL_24_25\\datasets\\A\\test.json.gz"

num_checkpoints=5
dropout = 0.1  #default 0.0
num_layers=7
input_dim = 30
hidden_dim = 60   #dimensione fo embeddings
batch_size = 32
epochs = 100
seed = 17
output_dim = 6
mlp_ratio= 5
num_heads = 5
learning_rate = 0.001
patience = 20

#Noisy Cross Entropy
p_noisy = 0.2

#Symmetrric Cross Entropy
alpha_SCE=0.1
beta_SCE=1.0

#Classic Cross Entropy with L2 regularization
lambda_l2_classic = 1e-4
apply_l2_to_all_params = False


########################################################################################################################
#                                                       CREATION NEED FOLDERS
########################################################################################################################

script_dir = os.getcwd()
test_dir_name = os.path.basename(os.path.dirname(test_path))

logs_folder = os.path.join(script_dir, "logs", test_dir_name)
log_file = os.path.join(logs_folder, "training.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
os.makedirs(checkpoints_folder, exist_ok=True)



########################################################################################################################
#                                                       DEFINITION LOGGER
########################################################################################################################

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(asctime)s - %(levelname) - s%(reset)s | %(log_color) - s%(message) - s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
streamFile = logging.FileHandler(filename=f'{log_file}', mode='w', encoding='utf-8')
log = logging.getLogger('GCNL-Main')
log.setLevel(LOG_LEVEL)
log.addHandler(stream)
log.addHandler(streamFile)


########################################################################################################################
#                                                       DATALOADER CLASS
########################################################################################################################

class GraphDataset(Dataset):
    def __init__(self, filename):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw)
        super().__init__()

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path, use_cache=True, cache_dir="cache"):
        cache_dir = os.path.join(os.path.dirname(path), cache_dir)
        original_file = Path(path)
        cache_file = Path(cache_dir) / f"{original_file.stem}.pkl"
        cache_file.parent.mkdir(exist_ok=True)  # CREA DIRECTORY CACHE SE NON ESISTE

        if use_cache and cache_file.exists():
            cache_time = cache_file.stat().st_mtime
            original_time = original_file.stat().st_mtime

            if cache_time > original_time:
                log.info(f"Loading cached graphs from {cache_file}...")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            else:
                log.info("Cache outdated, reprocessing...")

        log.info(f"Loading graphs from {path}...")
        log.info("This may take a few minutes, please wait...")

        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)

        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))

        if use_cache:
            log.info(f"Saving processed graphs to cache: {cache_file}...")
            with open(cache_file, "wb") as f:
                pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"Cache saved! Next time loading will be much faster.")

        return graphs

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    #log.info(f"edge_index shape: {edge_index.shape}")
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    #log.info(f"edge_attr shape: {edge_attr.shape}")
    num_nodes = graph_dict["num_nodes"]
    #log.info(f"graph num nodes: {num_nodes}")
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    #log.info(f"y shape: {y.shape}")
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)

def compute_node2vec_embeddings(data, embedding_dim=300, walk_length=20, context_size=10, walks_per_node=10):
    model = Node2Vec(
        data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_nodes=data.num_nodes,
        sparse=False
    )
    model = model.to("cpu")
    loader = model.loader(batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

    model.train()
    for _ in range(1):  # 1 epoca per velocità, aumenta se vuoi qualità migliore
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        embeddings = model.embedding.weight.clone()

    # Debug: verifica che siano corretti
    if embeddings.shape[0] != data.num_nodes:
        log.info(f"[Compute node2vec] Mismatch: {embeddings.shape[0]} embeddings vs {data.num_nodes} nodes")

    data.x = embeddings
    return data

########################################################################################################################
#                                                       UTILITIES
########################################################################################################################


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def add_node2vec_embeddings(data_in):
    #log.info(f"[Transform] Processing graph with {data_in.num_nodes} nodes")
    return compute_node2vec_embeddings(data_in, embedding_dim=input_dim)

def save_predictions(predictions, test_path):
    script_dir = os.getcwd()
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))

    os.makedirs(submission_folder, exist_ok=True)

    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })

    output_df.to_csv(output_csv_path, index=False)
    log.info(f"Predictions saved to {output_csv_path}")

def plot_training_progress(train_losses, train_accuracies, f1s, output_dir, str_phase):
    epochs = range(1, len(train_losses) + 1)
    if(len(f1s) == 0):
        f1s = 0*train_losses

    plt.figure(figsize=(18, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label=f"{str_phase} Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{str_phase} Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label=f"{str_phase} Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{str_phase} Accuracy per Epoch')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1s, label=f"{str_phase} F1", color='red')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title(f'{str_phase} F1-score')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{str_phase}_progress.png"))
    plt.close()

def train(data_loader,config_short_name ,model, optimizer, criterion, device, num_epochs, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc=f"Training | {config_short_name} | Epoch {current_epoch+1}/{num_epochs} ", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        log.info(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader),  correct / total

def evaluate(data_loader, model, device, criterion,calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    total_loss = 0
    total_entropy = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            probs = F.softmax(output, dim=1)
            entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
            total_entropy += entropy

            if calculate_accuracy:
                predictions.extend(pred.cpu().numpy())
                targets.extend(data.y.cpu().numpy())
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            else:
                predictions.extend(pred.cpu().numpy())

    if calculate_accuracy:
        accuracy = correct / total
        f1 = f1_score(targets, predictions, average='macro')
        avg_entropy = total_entropy / len(data_loader)
        return total_loss / len(data_loader), accuracy, f1, avg_entropy
    return predictions

def calculate_model_size(model: torch.nn.Module) -> int:
    total_size = 0

    for param in model.parameters():
        param_size = param.numel() * 4
        total_size += param_size

    for buffer in model.buffers():
        buffer_size = buffer.numel() * 4
        total_size += buffer_size

    return total_size

def get_model_summary_simple(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = calculate_model_size(model) / (1024 * 1024)

    log.info(f"Model: {model.__class__.__name__}")
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")
    log.info(f"Model size: {model_size_mb:.2f} MB")

def compute_class_weights(dataset, num_classes):
    counts = [0] * num_classes
    for g in dataset:
        if g.y is not None:
            counts[g.y.item()] += 1
    total = sum(counts)
    weights = [total / (c + 1e-6) for c in counts]
    norm_weights = torch.tensor(weights)
    norm_weights = norm_weights / norm_weights.sum()
    return norm_weights

########################################################################################################################
#                                                       DEFINITION ARCHITECTURE
########################################################################################################################


#-----------------------CRITERION

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()

class SymmetricCrossEntropy(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, pred, labels):
        ce = F.cross_entropy(pred, labels, reduction='none', weight=self.class_weights)

        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)

        label_one_hot = torch.zeros(pred.size()).to(pred.device)
        label_one_hot.scatter_(1, labels.view(-1, 1), 1)

        if self.class_weights is not None:
            weights_per_sample = self.class_weights[labels].view(-1, 1)  # shape [B, 1]
            rce = -torch.sum(weights_per_sample * label_one_hot * torch.log(pred_softmax), dim=1)
        else:
            rce = -torch.sum(label_one_hot * torch.log(pred_softmax), dim=1)

        loss = self.alpha * ce + self.beta * rce
        return loss.mean()

class ClassicCrossEntropyWithL2(torch.nn.Module):
    def __init__(self, lambda_l2, apply_to_all_params):
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.apply_to_all_params = apply_to_all_params

    def forward(self, pred, labels, model=None):
        ce_loss = F.cross_entropy(pred, labels)

        # Regolarizzazione L2
        l2_reg = 0.0
        if model is not None:
            if self.apply_to_all_params:
                for param in model.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param) ** 2
            else:
                for name, param in model.named_parameters():
                    if 'weight' in name and param.requires_grad:
                        if 'bias' not in name and 'norm' not in name:
                            l2_reg += torch.norm(param) ** 2

        total_loss = ce_loss + self.lambda_l2 * l2_reg

        return total_loss

#-----------------------DEFINITION GAT MODEL

class MultiHeadGraphAttention(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        self.dropout = torch.nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        self.edge_proj = None


    def forward(self, x, edge_index, edge_attr=None):
        num_nodes = x.size(0)

        # Linear projections
        Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        # Compute attention scores for edges
        row, col = edge_index
        q_i = Q[row]  # [num_edges, num_heads, head_dim]
        k_j = K[col]  # [num_edges, num_heads, head_dim]
        v_j = V[col]  # [num_edges, num_heads, head_dim]

        # Attention scores
        attn_scores = (q_i * k_j).sum(dim=-1) / self.scale  # [num_edges, num_heads]

        # Add edge features if available
        if edge_attr is not None:
            # Gestisci diverse forme di edge_attr
            if edge_attr.dim() == 3:
                edge_attr = edge_attr.squeeze(1)  # [num_edges, 1, features] -> [num_edges, features]
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)  # [num_edges] -> [num_edges, 1]

            # Crea o riusa il proiettore per edge attributes
            if self.edge_proj is None or self.edge_proj.in_features != edge_attr.size(-1):
                self.edge_proj = torch.nn.Linear(edge_attr.size(-1), num_heads).to(edge_attr.device)

            # Proietta edge_attr per avere la dimensione corretta
            edge_scores = self.edge_proj(edge_attr)  # [num_edges, num_heads]
            attn_scores = attn_scores + edge_scores

        # Softmax per node (group by source node)
        attn_scores = self.softmax_edges(attn_scores, row, num_nodes)
        attn_scores = self.dropout(attn_scores)

        weighted_values = attn_scores.unsqueeze(-1) * v_j  # [num_edges, num_heads, head_dim]
        aggregated_out = torch.zeros(num_nodes, num_heads, hidden_dim // num_heads, device=x.device, dtype=x.dtype)

        aggregated_out.index_add_(0, row, weighted_values)

        aggregated_out = aggregated_out.view(num_nodes, self.hidden_dim)
        final_out = self.out_proj(aggregated_out)

        return final_out

    def softmax_edges(self, scores, row, num_nodes):
        """Apply softmax per source node"""
        scores_max = torch.full((num_nodes,), float('-inf'), device=scores.device)
        scores_max.index_reduce_(0, row, scores.max(dim=1)[0], reduce='amax')
        scores_max = scores_max[row].unsqueeze(1)

        scores_exp = torch.exp(scores - scores_max)
        scores_sum = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        scores_sum.index_add_(0, row, scores_exp)
        scores_sum = scores_sum[row]

        return scores_exp / (scores_sum + 1e-8)

class GraphTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio):
        super().__init__()

        self.attention = MultiHeadGraphAttention(hidden_dim, num_heads, dropout)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_hidden_dim, hidden_dim),
            torch.nn.Dropout(dropout)
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gat = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        attn_out = self.attention(x, edge_index, edge_attr)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_dim, max_nodes=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_embedding = torch.nn.Embedding(max_nodes, hidden_dim)

    def forward(self, x, batch):
        pos_indices = []
        for i in range(batch.max().item() + 1):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            pos_indices.append(torch.arange(num_nodes, device=x.device))

        pos_indices = torch.cat(pos_indices)
        pos_emb = self.pos_embedding(pos_indices)

        return x + pos_emb

class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mlp_ratio, num_layers, num_heads, dropout, pool_type='mean'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim)

        self.layers = torch.nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout, mlp_ratio)
            for _ in range(num_layers)
        ])

        if pool_type == 'mean':
            self.global_pool = global_mean_pool
        elif pool_type == 'max':
            self.global_pool = global_max_pool
        elif pool_type == 'sum':
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.output_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )

        self.final_norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        # Conversione sicura del tipo
        if x.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            x = x.float()

        x = self.input_proj(x)
        x = self.pos_encoding(x, batch)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        x = self.final_norm(x)
        x = self.global_pool(x, batch)
        out = self.output_proj(x)

        return out

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else torch.nn.Identity()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(torch.nn.Dropout(dropout))

        self.global_pool = global_mean_pool
        self.output_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )

        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if x.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            x = x.float()
        x = self.input_proj(x)

        for i in range(self.num_layers):
            residual = x
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
            x = self.norms[i](x)
            x = x + residual
            x = self.dropouts[i](x)

        x = self.global_pool(x, batch)
        out = self.output_proj(x)

        return out


########################################################################################################################
#                                                       MAIN
########################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_checkpoints = num_checkpoints if num_checkpoints else 3
log.info(f"Using device: {device}")

if os.path.exists(checkpoints_folder) and len(train_path)==0:

    checkpoint_files = [f for f in os.listdir(checkpoints_folder) if f.endswith('.pth')]

    if not checkpoint_files:
        log.info(f"No checkpoint files found in {checkpoints_folder}, terminating testing procedure.\n")
    else:
        log.info(f"Checkpoint files found in {checkpoints_folder}, starting test procedure...\n")
        log.info("Loading test dataset...")
        test_dataset = GraphDataset(test_path)
        log.info("Test dataset loaded, calculating embeddings...")

        for i in range(len(test_dataset.graphs)):
            test_dataset.graphs[i] = add_node2vec_embeddings(test_dataset.graphs[i])
        log.info("Node2Vec embeddings calculation terminated for test set.\n")

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for checkpoint_file in checkpoint_files:
            checkpoint_full_path = os.path.join(checkpoints_folder, checkpoint_file)
            log.info(f"Loading model from {checkpoint_full_path}")

            weights = compute_class_weights(test_dataset, output_dim).to(device)
            criterion = SymmetricCrossEntropy(alpha_SCE, beta_SCE, output_dim, class_weights=weights)

            model = SimpleGCN(input_dim, hidden_dim, output_dim).to(device)
            model.load_state_dict(torch.load(checkpoint_full_path, map_location=device))

            predictions = evaluate(test_loader, model, device, criterion,calculate_accuracy=False)

            checkpoint_name = os.path.splitext(checkpoint_file)[0]
            save_predictions(predictions, test_path)

        log.info("Predictions completed for all checkpoints")
else:

    log.info("Loading train dataset...")
    full_dataset = GraphDataset(train_path)
    log.info("Train dataset loaded, calculating embeddings...")
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    for i in range(len(full_dataset.graphs)):
        full_dataset.graphs[i] = add_node2vec_embeddings(full_dataset.graphs[i])
    log.info("Node2Vec embeddings calculation terminated for the train test.\n")

    if len(train_path) != 0:
        log.info("Starting training...")

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size],generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        log.info("Dataset splitted into train and validation set \n")

        weights = compute_class_weights(train_dataset, output_dim).to(device)
        criterion = SymmetricCrossEntropy(alpha_SCE, beta_SCE, output_dim, class_weights=weights)

        # Elenco configurazioni da testare
        configs = {
            "SGCN model with Symmetric Cross Entropy loss": ("SGCN_CE",
                                                             SimpleGCN(input_dim, hidden_dim, output_dim).to(device),
                                                             criterion)
        }

        for config_name, (config_short_name, config_model, config_criterion) in configs.items():



            for i, batch in enumerate(train_loader):
                if i >= 0:      #NON LO FA PROPRIO
                    break

                log.info(f"Batch {i}:")
                log.info(f"  - batch.x shape: {batch.x.shape if batch.x is not None else 'None'}")
                log.info(f"  - batch.edge_index shape: {batch.edge_index.shape if batch.edge_index is not None else 'None'}")
                log.info(f"  - batch.edge_attr shape: {batch.edge_attr.shape if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else 'None'}")
                log.info(f"  - batch.y: {batch.y}")
                log.info(f"  - batch.y shape: {batch.y.shape if batch.y is not None else 'None'}")
                log.info(f"  - batch.y dtype: {batch.y.dtype if batch.y is not None else 'None'}")
                log.info(f"  - batch.batch shape: {batch.batch.shape if hasattr(batch, 'batch') else 'None'}")
                log.info(f"  - Number of graphs in batch: {batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 'Unknown'}")



            model = config_model
            log.info(f"Training configuration: {config_name}")
            log.info("Params of the model: ")
            get_model_summary_simple(model)


            criterion = config_criterion
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            num_epochs = epochs
            best_val_accuracy = 0.0
            best_val_loss = 100000
            best_val_f1 = 0.0

            train_losses = []
            train_accuracies = []
            train_f1s = []
            val_losses = []
            val_accuracies = []
            val_f1s = []

            if num_checkpoints > 1:
                checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
            else:
                checkpoint_intervals = [num_epochs]

            patience_counter = 0

            for epoch in range(num_epochs):
                train_loss, train_acc = train(
                    train_loader, config_short_name, model, optimizer, criterion, device,
                    num_epochs,
                    save_checkpoints=(epoch + 1 in checkpoint_intervals),
                    checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                    current_epoch=epoch
                )

                val_loss, val_acc, val_f1, val_entropy = evaluate(val_loader, model, device, criterion ,calculate_accuracy=True)

                log.info("-------REGAP-------")
                log.info(f"Epoch {epoch + 1}/{num_epochs}")
                log.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                log.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Entropy: {val_entropy:.4f}")

                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_f1s.append(val_f1)

                if val_f1 > best_val_f1:
                    best_val_loss = val_loss
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(),f"{os.path.join(checkpoints_folder, f"model_{test_dir_name}")}_epoch_{epoch + 1}.pth")
                    log.info(f"Best model updated and saved at {f"{os.path.join(checkpoints_folder, f"model_{test_dir_name}")}_epoch_{epoch + 1}.pth"}")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

            plot_training_progress(train_losses, train_accuracies, train_f1s, os.path.join(logs_folder, "plots"),"Training")
            plot_training_progress(val_losses, val_accuracies, val_f1s, os.path.join(logs_folder, "plotsVal"), "Validation")



