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
from datetime import datetime

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
# ======================================================== CLASSES ======================================================== #
# ========================================================================================================================= #


class SymmetricCrossEntropy(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, pred, labels):
        # Standard Cross Entropy
        ce = F.cross_entropy(pred, labels, reduction='none')

        # Reverse Cross Entropy (RCE)
        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)

        # One-hot encoding delle labels
        label_one_hot = torch.zeros(pred.size()).to(pred.device).scatter_(1, labels.view(-1, 1), 1)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        rce = (-1 * torch.sum(pred_softmax * torch.log(label_one_hot), dim=1))

        # Symmetric loss
        loss = self.alpha * ce + self.beta * rce
        return loss.mean()

class OutlierDiscountingLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, labels):
        ce_loss = F.cross_entropy(pred, labels, reduction='none')
        pt = torch.exp(-ce_loss)

        # Focal loss component per down-weight outliers
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Outlier detection: high loss samples are likely outliers
        if len(ce_loss) > 1:
            loss_threshold = torch.quantile(ce_loss, 0.7)  # Top 30% losses
            outlier_mask = (ce_loss > loss_threshold).float()
        else:
            outlier_mask = torch.zeros_like(ce_loss)

        # Discount outliers
        discount_factor = 1.0 - 0.5 * outlier_mask

        return (focal_weight * ce_loss * discount_factor).mean()

class ELRLoss(torch.nn.Module):
    def __init__(self, num_examp, num_classes=6, lambda_reg=3.0, beta=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.target = torch.zeros(num_examp, num_classes)

    def forward(self, index, output, label):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

        # Standard cross entropy
        ce_loss = F.cross_entropy(output, label)

        # ELR regularization
        y_pred_avg = self.target[index].to(output.device)
        elr_reg = ((1 - (y_pred_avg * y_pred).sum(dim=1)).log()).mean()

        # Update target
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * y_pred.detach().cpu()

        return ce_loss + self.lambda_reg * elr_reg

class CoTeaching:
    def __init__(self, model1, model2, forget_rate=0.2):
        self.model1 = model1
        self.model2 = model2
        self.forget_rate = forget_rate

    def train_step(self, data, criterion):
        # Entrambi i modelli predicono
        out1 = self.model1(data)
        out2 = self.model2(data)

        # Calcola loss per entrambi
        loss1 = F.cross_entropy(out1, data.y, reduction='none')
        loss2 = F.cross_entropy(out2, data.y, reduction='none')

        # Seleziona campioni con loss più bassa (clean samples)
        num_remember = max(1, int((1 - self.forget_rate) * len(data.y)))
        _, idx1 = torch.topk(loss1, num_remember, largest=False)
        _, idx2 = torch.topk(loss2, num_remember, largest=False)

        # Cross-training: model1 si allena sui campioni scelti da model2
        return loss1[idx2].mean(), loss2[idx1].mean()


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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_log_{timestamp}.txt"
streamFile = logging.FileHandler(filename=f'./logs/{log_filename}', mode='w', encoding='utf-8')


# ========================================================================================================================= #
# ======================================================= CONSTANTS ======================================================= #
# ========================================================================================================================= #

num_epochs = 25
mlp_ratio = 4
patience = 15
use_coteaching_FLAG= False
chars_intestazione = 150
input_dim = 300
hidden_dim = 250
output_dim = 6
num_layers = 5
num_heads = 5                       # num_head * output_dim = hidden_dim !!!
dropout = 0.1
perc_train_set = 0.8
batch_size=32
alpha_SCE = 0.1
beta_SCE = 1.0
gamma_ODL = 2.0
alpha_ODL = 0.25
lambda_reg_ELR = 3.0
forget_rate_co_teaching=0.2
learning_rate = 0.001
save_checkpoint = True
alpha_mixup=0.2

log = logging.getLogger('GCNL-Main')
log.setLevel(LOG_LEVEL)
log.addHandler(stream)
log.addHandler(streamFile)

# ========================================================================================================================= #
# ===================================================== ARCHITECTURE ====================================================== #
# ========================================================================================================================= #

class GraphTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gat = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_hidden_dim, hidden_dim),
            torch.nn.Dropout(dropout)
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        attn_out = self.gat(x, edge_index)
        x = self.norm1(x + self.dropout(attn_out))
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mlp_ratio, num_layers ,num_heads, dropout):
        super().__init__()
        self.embedding = torch.nn.Embedding(1, hidden_dim)
        # Stack di Transformer layers
        self.layers = torch.nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.global_pool = global_mean_pool
        self.output_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.global_pool(x, batch)
        out = self.output_proj(x)
        return out

# ========================================================================================================================= #
# ======================================================= FUNCTIONS ======================================================= #
# ========================================================================================================================= #
def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def Intestazione(stringaTBP):
    strlenght = len(stringaTBP)
    return ("="*int((chars_intestazione-strlenght)/2)+stringaTBP+"="*int((chars_intestazione-strlenght)/2))

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

    log.debug(f"Model: {model.__class__.__name__}")
    log.debug(f"Total parameters: {total_params:,}")
    log.debug(f"Trainable parameters: {trainable_params:,}")
    log.debug(f"Model size: {model_size_mb:.2f} MB")

def train_coteaching(coteacher, train_loader, optimizer1, optimizer2, device, criterion,config_name,epoch, num_epochs):
    coteacher.model1.train()
    coteacher.model2.train()
    total_loss1 = 0
    total_loss2 = 0

    batch_pbar = tqdm(train_loader, desc=f"Co-Teaching Training | {config_name} | Epoch {epoch+1}/{num_epochs}",unit="batch")

    for batch_idx,data in enumerate(batch_pbar):
        data = data.to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        out1 = coteacher.model1(data)
        out2 = coteacher.model1(data)

        if isinstance(criterion, ELRLoss):
            indices = torch.arange(batch_idx * train_loader.batch_size,
                                   min((batch_idx + 1) * train_loader.batch_size, len(train_loader.dataset)))
            loss1 = criterion(indices, out1, data.y)
            loss2 = criterion(indices, out2, data.y)
        else:
            loss1 = criterion(out1, data.y)
            loss2 = criterion(out2, data.y)

        loss1.backward()
        loss2.backward()

        torch.nn.utils.clip_grad_norm_(coteacher.model1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(coteacher.model2.parameters(), 1.0)

        optimizer1.step()
        optimizer2.step()

        total_loss1 += loss1.item()
        total_loss2 += loss2.item()

        batch_pbar.set_postfix({
            'Loss1': f'{loss1.item():.4f}',
            'Loss2': f'{loss2.item():.4f}'
        })

    return total_loss1 / len(train_loader), total_loss2 / len(train_loader)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    return correct / total if total > 0 else 0

def graph_mixup(data1, data2, alpha=0.2, num_classes=6):
    lam = np.random.beta(alpha, alpha)

    # Mix node features (stesso numero di nodi required)
    if data1.x.shape[0] == data2.x.shape[0]:
        mixed_x = lam * data1.x + (1 - lam) * data2.x

        # Mix labels
        y1_one_hot = F.one_hot(data1.y, num_classes).float()
        y2_one_hot = F.one_hot(data2.y, num_classes).float()
        mixed_y = lam * y1_one_hot + (1 - lam) * y2_one_hot

        # Create mixed data object
        mixed_data = data1.clone()
        mixed_data.x = mixed_x

        return mixed_data, mixed_y, lam
    else:
        # Se i grafi hanno dimensioni diverse, restituisci il primo senza mixup
        return data1, F.one_hot(data1.y, num_classes).float(), 1.0

def train_single_model(model, train_loader, criterion, optimizer, device, config_name, epoch, num_epochs, use_mixup,alpha_mixup,save_checkpoint):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    batch_pbar = tqdm(train_loader, desc=f"Single Training | {config_name} | Epoch {epoch + 1}/{num_epochs}", unit="batch")

    for batch_idx, data in enumerate(batch_pbar):
        data = data.to(device)
        optimizer.zero_grad()

        if use_mixup and len(train_loader.dataset) > batch_idx + 1:
            # Get another random sample for mixup
            next_idx = (batch_idx + 1) % len(train_loader)
            try:
                data2 = next(iter(DataLoader([train_loader.dataset[next_idx]], batch_size=1)))
                data2 = data2.to(device)

                mixed_data, mixed_y, lam = graph_mixup(data, data2, alpha_mixup, output_dim)
                output = model(mixed_data)

                # Mixup loss
                loss = lam * F.cross_entropy(output, mixed_y.argmax(dim=1)) + \
                       (1 - lam) * F.cross_entropy(output, data2.y)
            except:
                # Fallback to normal training if mixup fails
                output = model(data)
                if isinstance(criterion, ELRLoss):
                    indices = torch.arange(batch_idx * train_loader.batch_size, min((batch_idx + 1) * train_loader.batch_size, len(train_loader.dataset)))
                    loss = criterion(indices, output, data.y)
                else:
                    loss = criterion(output, data.y)
        else:
            output = model(data)
            if isinstance(criterion, ELRLoss):
                indices = torch.arange(batch_idx * train_loader.batch_size, min((batch_idx + 1) * train_loader.batch_size, len(train_loader.dataset)))
                loss = criterion(indices, output, data.y)
            else:
                loss = criterion(output, data.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if not use_mixup:  # Solo per accuracy normale
            pred = output.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        # Aggiorna progress bar
        if use_mixup:
            batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        else:
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%' if total > 0 else '0.00%'
            })

    if save_checkpoint:
        checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', f"model_{os.path.basename(os.path.dirname(args.test_path))}_best")
        checkpoints_folder = os.path.join(os.getcwd(), 'checkpoints',os.path.basename(os.path.dirname(args.test_path)))
        os.makedirs(checkpoints_folder, exist_ok=True)

        checkpoint_file = f"{checkpoint_path}_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(train_loader)

def evaluate_predictions(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

    return predictions







def main_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}\n")

    log.info(Intestazione("LOADING DATASET"))
    log.info("Loading test dataset...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_loader = None
    val_loader = None

    if args.train_path:
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        train_size = int(perc_train_set * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        log.info(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}")

    log.info("Dataset(s) loaded. \n")

    log.info(Intestazione("WORK CONFIGURATIONS"))
    log.info("Definition of work configuration...")

    configs = []

    loss_functions = {
        'SCE': SymmetricCrossEntropy(alpha=alpha_SCE, beta=beta_SCE),
        'ODL': OutlierDiscountingLoss(gamma=gamma_ODL, alpha=alpha_ODL),
        'ELR': ELRLoss(len(train_subset), num_classes=output_dim,lambda_reg=lambda_reg_ELR) if train_loader else None
    }

    loss_functions = {
        'SCE': SymmetricCrossEntropy(alpha=alpha_SCE, beta=beta_SCE)
    }

    for loss_name, criterion in loss_functions.items():
        if criterion is not None:
            configs.append({
                'name': f'{loss_name}',
                'criterion': criterion,
                'use_mixup': False,
                'use_coteaching': use_coteaching_FLAG
            })
            # Aggiungi versione con mixup
            configs.append({
                'name': f'{loss_name}_mixup',
                'criterion': criterion,
                'use_mixup': True,
                'use_coteaching': use_coteaching_FLAG
            })


    log.info("Configurations created: ")
    counter = 1
    for config in configs:
        log.info("CONFIGURATION "+str(counter)+": " + config['name']+" - CoTeaching: "+str(config['use_coteaching']))
        counter = counter + 1
    log.info("Done. \n")

    best_models = {}
    best_accuracies = {}

    #config_pbar = tqdm(configs, desc="Training Configurations")     #PER LE BARRE DI AVANZAMENTO
    for config in configs:
        log.info(Intestazione("STARTING TRAINING CONFIG: "+ config['name']+" - CoTeaching: "+str(config['use_coteaching'])))

        config_name = config['name']
        criterion_iter = config['criterion']

        if not train_loader:
            log.warning("No training data provided, skipping training configuration")
            break

        if config['use_coteaching']:
            model1 = GraphTransformer(input_dim, hidden_dim, output_dim, mlp_ratio, num_layers, num_heads, dropout).to(device)
            model2 = GraphTransformer(input_dim, hidden_dim, output_dim, mlp_ratio, num_layers, num_heads, dropout).to(device)

            log.info(f"Created two graph transformer models: model1 and model2, printing info about model1...")
            get_model_summary_simple(model1)

            coteacher = CoTeaching(model1, model2, forget_rate=forget_rate_co_teaching)
            optimizer1 = torch.optim.AdamW(model1.parameters(), lr=learning_rate, weight_decay=1e-4)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=learning_rate, weight_decay=1e-4)

            for epoch in range(num_epochs):
                loss1, loss2 = train_coteaching(coteacher, train_loader, optimizer1, optimizer2, device, criterion_iter,config_name ,epoch, num_epochs)

                # Valuta entrambi i modelli
                val_acc1 = evaluate(model1, val_loader, device)
                val_acc2 = evaluate(model2, val_loader, device)
                val_acc = max(val_acc1, val_acc2)

                if val_acc > best_acc:
                    best_acc = val_acc
                    # Salva il modello migliore
                    best_model = model1 if val_acc1 > val_acc2 else model2
                    best_models[config_name] = copy.deepcopy(best_model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        else:
            # Training normale
            model = GraphTransformer(input_dim, hidden_dim, output_dim, mlp_ratio, num_layers, num_heads, dropout).to(device)

            log.info(f"Created graph transformer with criterion : {config['name']}, printing info about model...")
            get_model_summary_simple(model)

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

            best_acc = 0
            patience_counter = 0


            for epoch in range(num_epochs):
                train_loss = train_single_model(model, train_loader, config['criterion'], optimizer, device, config_name, epoch, num_epochs,config['use_mixup'],alpha_mixup,save_checkpoint)

                val_acc = evaluate(model, val_loader, device)
                scheduler.step()

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_models[config_name] = copy.deepcopy(model.state_dict())
                    best_accuracies[config_name] = best_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        log.info(f"Best accuracy for {config_name}: {best_acc:.4f}")

        # Aggiorna progress bar principale
        current_best = max(best_accuracies.values())
        log.info({'Current Best': f'{current_best*100:.2f}%'})
        log.info("Done, next configuration... \n\n")

    # Trova il modello migliore
    if best_accuracies:
        best_config = max(best_accuracies, key=best_accuracies.get)
        log.info(f"Best overall configuration: {best_config} with accuracy: {best_accuracies[best_config]:.4f}")

        # Carica il modello migliore per le predizioni finali
        final_model = GraphTransformer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout).to(device)
        final_model.load_state_dict(best_models[best_config])
    else:
        # Se non c'è training, usa un modello base per le predizioni
        final_model = GraphTransformer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout).to(device)
        log.info("No training performed, using untrained model for predictions")

    # Genera predizioni sul test set
    log.info("Generating test predictions...")
    predictions = evaluate_predictions(final_model, test_loader, device)
    test_graph_ids = list(range(len(predictions)))

    # Salva le predizioni
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    output_csv_path = f"testset_{test_dir_name}.csv"
    output_df = pd.DataFrame({
        "GraphID": test_graph_ids,
        "Class": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    log.info(f"Test predictions saved to {output_csv_path}")

    # Salva il modello migliore
    if best_accuracies:
        model_path = f"best_model_{best_config}.pth"
        torch.save(best_models[best_config], model_path)
        log.info(f"Best model saved to {model_path}")

    return best_models, best_accuracies

def main(args):
    log.info(Intestazione("STARTING MAIN"))
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






