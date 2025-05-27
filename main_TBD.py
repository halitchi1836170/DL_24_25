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





########################################################################################################################
#                                                       CONSTANTS
########################################################################################################################

train_path = "C:\\Users\\halit\\Desktop\\ANDREI\\UNI\\AIRO\\DL\\progetto\\GraphClassificationNoisyLabels\\datasets\\A\\train.json.gz"
test_path = "C:\\Users\\halit\\Desktop\\ANDREI\\UNI\\AIRO\\DL\\progetto\\GraphClassificationNoisyLabels\\datasets\\A\\test.json.gz"

num_checkpoints=5
dropout = 0.1  #default 0.0
num_layers=1
input_dim = 300
hidden_dim = 250   #dimensione fo embeddings
batch_size = 32
epochs = 1
seed = 17
output_dim = 6
mlp_ratio= 2
num_heads = 1
learning_rate = 0.001

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

checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
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
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path):
        log.info(f"Loading graphs from {path}...")
        log.info("This may take a few minutes, please wait...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
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

def compute_node2vec_embeddings(data, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10):
    model = Node2Vec(
        data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_nodes=data.num_nodes,
        sparse=False
    )
    model = model.to('cpu')
    loader = model.loader(batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for _ in range(1):  # 1 epoca per velocità, aumenta se vuoi qualità migliore
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

    # Ottieni gli embedding per tutti i nodi
    embeddings = model.embedding.weight.data.clone()
    data.x = embeddings
    return data

########################################################################################################################
#                                                       UTILITIES
########################################################################################################################


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def add_node2vec_embeddings(data):
    return compute_node2vec_embeddings(data, embedding_dim=hidden_dim)

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

def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
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

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            else:
                predictions.extend(pred.cpu().numpy())
    if calculate_accuracy:
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy
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
    def __init__(self, alpha, beta, num_classes):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, pred, labels):
        ce = F.cross_entropy(pred, labels, reduction='none')

        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)

        label_one_hot = torch.zeros(pred.size()).to(pred.device)
        label_one_hot.scatter_(1, labels.view(-1, 1), 1)

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

    def forward(self, x, edge_index, edge_attr=None):
        batch_size, num_nodes = x.size(0), x.size(0)

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
            edge_attr = edge_attr.unsqueeze(1).expand(-1, self.num_heads)
            attn_scores = attn_scores + edge_attr

        # Softmax per node (group by source node)
        attn_scores = self.softmax_edges(attn_scores, row, num_nodes)
        attn_scores = self.dropout(attn_scores)

        # Apply attention to values
        out = attn_scores.unsqueeze(-1) * v_j  # [num_edges, num_heads, head_dim]

        # Aggregate messages per node
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim,
                          device=x.device, dtype=x.dtype)
        out.index_add_(0, row, out)

        # Reshape and project
        out = out.view(num_nodes, self.hidden_dim)
        out = self.out_proj(out)

        return out

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
        self.embedding = torch.nn.Embedding(1, input_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.global_pool = global_mean_pool
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.global_pool(x, batch)
        out = self.fc(x)
        return out


########################################################################################################################
#                                                       MAIN
########################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_checkpoints = num_checkpoints if num_checkpoints else 3
log.info(f"Using device: {device}")

if os.path.exists(checkpoint_path) and len(train_path)==0:
    #model.load_state_dict(torch.load(checkpoint_path))
    log.info(f"Loaded best model from {checkpoint_path}")

# Elenco configurazioni da testare
configs = {
    "SGCN model with simple Cross Entropy loss": ("SGCN_CE", SimpleGCN(input_dim, hidden_dim, output_dim).to(device),torch.nn.CrossEntropyLoss()),
    "GAT model with Symmetric Cross Entropy Loss": ("GAT_SCE",GraphTransformer(input_dim, hidden_dim, output_dim, mlp_ratio,num_layers, num_heads, dropout,pool_type='mean').to(device),SymmetricCrossEntropy(alpha_SCE, beta_SCE, output_dim)),
    "GAT model with L2 Regularized Cross Entroy Loss": ("GAT_RCE",GraphTransformer(input_dim, hidden_dim, output_dim, mlp_ratio,num_layers, num_heads, dropout,pool_type='mean').to(device),ClassicCrossEntropyWithL2(lambda_l2_classic,apply_l2_to_all_params))
}

configs = {"GAT model with Symmetric Cross Entropy Loss": ("GAT_SCE",GraphTransformer(input_dim, hidden_dim, output_dim, mlp_ratio,num_layers, num_heads, dropout,pool_type='mean').to(device),SymmetricCrossEntropy(alpha_SCE, beta_SCE, output_dim)),
    "GAT model with L2 Regularized Cross Entroy Loss": ("GAT_RCE",GraphTransformer(input_dim, hidden_dim, output_dim, mlp_ratio,num_layers, num_heads, dropout,pool_type='mean').to(device),ClassicCrossEntropyWithL2(lambda_l2_classic,apply_l2_to_all_params))
}

if len(train_path) != 0:
    log.info("Strarting training...")
    for config_name, (config_short_name, config_model, config_criterion) in configs.items():

        full_dataset = GraphDataset(train_path, transform=add_node2vec_embeddings)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        log.info("Dataset loaded \n")

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                                                   generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        log.info("Dataset splitted into train and validation set \n")

        model = config_model
        log.info(f"Training configuration: {config_name}")
        log.info("Params of the model: ")
        get_model_summary_simple(model)

        criterion = config_criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        num_epochs = epochs
        best_val_accuracy = 0.0

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                train_loader, config_short_name, model, optimizer, criterion, device,
                num_epochs,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )

            val_loss, val_acc = evaluate(val_loader, model, device, calculate_accuracy=True)

            log.info(
                f"RECAP - Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                log.info(f"Best model updated and saved at {checkpoint_path}")

        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
        plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))



