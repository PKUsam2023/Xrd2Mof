import re
import torch
import itertools
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

# cif_lattice[64, 50, 3], cif_atom_positions[64, 50, 3], xrd[64, 5250], metal[64, 65], linker[64, Data(x=[a, 37], edge_index=[2, b], edge_attr=[b, 10])]


class LinkerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Define GCN layers for node features (37 -> 64 -> 128 -> 64)
        self.conv1 = GCNConv(in_channels=37, out_channels=64)
        self.conv2 = GCNConv(in_channels=64, out_channels=128)
        self.conv3 = GCNConv(in_channels=128, out_channels=64)

        # Fully connected + normalization to map graph-level features to 64-dim
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, data):
        # Extract info from Data object; edge attributes are not used
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # GCN stack over node features
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index)) # [num_nodes, 64]

        # Pool to graph-level representation
        graph_features = global_mean_pool(x, batch)  # [batch_size, 64]

        # MLP projection to 64-dim + normalization
        x = F.relu(self.fc1(graph_features))
        x = F.relu(self.fc2(x))
        x = self.norm(x)
        
        return x # 64


class MetalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_in = nn.LayerNorm(65)
        # 1st conv: expand from 1 channel to 16 channels
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        # 2nd conv: further local features, to 32 channels
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        # Optional 3rd conv: keep 32 channels, deepen extraction
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        
        # Flatten then pass through a deep MLP
        # Input size becomes channels * length = 32 * 65
        self.fc1 = nn.Linear(32 * 65, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        
        self.layer_norm = nn.LayerNorm(64)

    def forward(self, x):
        # x: [batch, 65]
        x = self.ln_in(x)
        x = x.unsqueeze(1)         # [batch, 1, 65]
        x = F.relu(self.bn1(self.conv1(x)))  # [batch, 16, 65]
        x = F.relu(self.bn2(self.conv2(x)))  # [batch, 32, 65]
        x = F.relu(self.bn3(self.conv3(x)))  # [batch, 32, 65]
        x = x.view(x.size(0), -1)   # flatten -> [batch, 32*65]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.layer_norm(x)
        return x  # output dim = 64


class XrdEncoder(nn.Module):
    def __init__(self):
        super(XrdEncoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # 5250 -> 1312
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2d = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # 1312 -> 328
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3d = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 328 -> 164
        self.bn3 = nn.BatchNorm1d(16)
        self.conv4d = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 164 -> 82
        self.bn4 = nn.BatchNorm1d(32)
        # Final flatten size: 32 * 82 = 2624
        self.fc = nn.Linear(32 * 82, 1024)
        
    def forward(self, x):
        # x: [batch, 5250]
        x = x.unsqueeze(1)  # [batch, 1, 5250]
        x = F.relu(self.conv1d(x))
        x = self.pool1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.conv2d(x))
        x = self.pool2(x)
        x = F.relu(self.bn2(x))
        x = F.relu(self.conv3d(x))
        x = self.pool3(x)
        x = F.relu(self.bn3(x))
        x = F.relu(self.conv4d(x))
        x = self.pool4(x)
        x = F.relu(self.bn4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x # 1024


class LatticeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: input channels 3 -> 8
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        
        # Block 2: 8 -> 16
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        
        # Block 3: 16 -> 32
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        
        # Block 4: deepen while keeping channels
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        
        # Block 5: reduce channels to 16 for the FC layer
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(16)
        
        # Use adaptive pooling to fix temporal length to 3 for the FC
        self.fc = nn.Linear(16 * 3, 32)

    def forward(self, x):
        # x: [batch, 3, 3] -> transpose to [batch, 3, 3]
        x = x.transpose(-1, -2)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 8, 3)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 16, 3)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 32, 3)
        x = F.relu(self.bn4(self.conv4(x)))  # (batch, 32, 3)
        x = F.relu(self.bn5(self.conv5(x)))  # (batch, 16, 3)
        # Adaptive average pooling to length=3 -> (batch, 16, 3)
        x = F.adaptive_avg_pool1d(x, 3)
        x = x.view(x.size(0), -1)           # flatten -> (batch, 16*3)
        x = self.fc(x)                      # (batch, 32)
        return x # 32


class AtomEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input assumed (batch, 50, 3); transpose to (batch, 3, 50)
        # First conv stack: keep sequence length
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        # One pooling to reduce sequence length 50 -> 25
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second conv stack: deepen the network
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(128)
        
         # Global average pooling over the sequence dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # FC to map 128 channels to target dim (e.g., 256)
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        # x: (batch, 50, 3)
        x = x.transpose(1, 2)         # (batch, 3, 50)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 16, 50)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 32, 50)
        x = self.pool(x)                     # (batch, 32, 25)
        
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 64, 25)
        x = F.relu(self.bn4(self.conv4(x)))  # (batch, 64, 25)
        x = F.relu(self.bn5(self.conv5(x)))  # (batch, 128, 25)
        x = F.relu(self.bn6(self.conv6(x)))  # (batch, 128, 25)
        
        # Global average pooling over sequence dimension
        x = self.global_pool(x)              # (batch, 128, 1)
        x = x.view(x.size(0), -1)            # (batch, 128)
        x = self.fc(x)                       # (batch, 256)
        return x # 256


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(projection_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, projection_dim)
        self.batch_norm = nn.BatchNorm1d(projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = x + projected  # residual connection
        x = self.layer_norm(x)
        return x


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        temperature=self.cfg.temperature
        cif_features=self.cfg.cif_features
        combined_features=self.cfg.combined_features

        self.lattice_encoder = LatticeEncoder()
        self.atom_encoder = AtomEncoder()
        self.xrd_encoder = XrdEncoder()
        self.metal_encoder = MetalEncoder()
        self.linker_encoder = LinkerEncoder()
        
        # Fusion layers (strengthen representation power)
        # cif_features ：raw size = lattice (18) + atom (64) = 82
        self.cif_fusion = nn.Sequential(
            nn.Linear(cif_features, 128),
            nn.ReLU(),
            nn.Linear(128, cif_features)
        )
        # combined_features ： xrd (1024) + metal (64) + linker (64) = 1152
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Linear(512, combined_features)
        )

        self.cif_projection = ProjectionHead(embedding_dim=cif_features, projection_dim=self.cfg.projection_dim, dropout=self.cfg.dropout)
        self.feature_projection = ProjectionHead(embedding_dim=combined_features, projection_dim=self.cfg.projection_dim, dropout=self.cfg.dropout)
        self.temperature = temperature

    def forward(self, batch):
        cif_lattice, cif_atom_positions, xrd, metal, linker, tokenizer = batch

        # Getting Image and Text Features
        lattice_features = self.lattice_encoder(cif_lattice.float())
        atom_features = self.atom_encoder(cif_atom_positions.float())
        xrd_features = self.xrd_encoder(xrd.float())
        metal_features = self.metal_encoder(metal.float())
        linker_features = self.linker_encoder(linker)

        # Fuse structural features
        cif_features = torch.cat((lattice_features, atom_features), dim=1)
        cif_features = self.cif_fusion(cif_features)
        # Fuse chemical features
        combined_features = torch.cat((xrd_features, metal_features, linker_features), dim=1)
        combined_features = self.feature_fusion(combined_features)

        # Getting Image and Text Embeddings (with same dimension)
        cif_embeddings = self.cif_projection(cif_features)
        feature_embeddings = self.feature_projection(combined_features)

        # Calculating the Loss
        logits = (feature_embeddings @ cif_embeddings.T) / self.temperature
        cifs_similarity = cif_embeddings @ cif_embeddings.T
        features_similarity = feature_embeddings @ feature_embeddings.T
        targets = F.softmax(
            (cifs_similarity + features_similarity) / 2 * self.temperature, dim=-1
        )
        features_loss = F.cross_entropy(logits, targets, reduction='none')
        cifs_loss = F.cross_entropy(logits.T, targets, reduction='none')
        loss =  (cifs_loss + features_loss) / 2.0 # shape: (batch_size)
        return loss, cifs_loss, features_loss


    @torch.no_grad()
    def sample(self, batch):
        cif_lattice, cif_atom_positions, xrd, metal, linker, tokenizer = batch
        
        lattice_features = self.lattice_encoder(cif_lattice.float())
        atom_features = self.atom_encoder(cif_atom_positions.float())
        xrd_features = self.xrd_encoder(xrd.float())
        metal_features = self.metal_encoder(metal.float())
        linker_features = self.linker_encoder(linker)

        cif_features = torch.cat((lattice_features, atom_features), dim=1)
        cif_features = self.cif_fusion(cif_features)

        combined_features = torch.cat((xrd_features, metal_features, linker_features), dim=1)
        combined_features = self.feature_fusion(combined_features)

        cif_embeddings = self.cif_projection(cif_features)
        feature_embeddings = self.feature_projection(combined_features)

        return tokenizer, cif_embeddings, feature_embeddings


class CLIPModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = BaseModel(cfg)
        
    def forward(self, batch):
        return self.model(batch)
    
    def sample(self, batch):
        return self.model.sample(batch)


    def training_step(self, batch, batch_idx):

        loss_tuple = self.model(batch)
        clip_loss = loss_tuple[0].mean()
        cif_loss = loss_tuple[1].mean()
        feature_loss = loss_tuple[2].mean()

        # Log additional loss terms as well
        self.log("train_loss", clip_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cif_loss", cif_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_feature_loss", feature_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {
            "loss": clip_loss,
            "clip_loss": clip_loss,
            "cif_loss": cif_loss,
            "feature_loss": feature_loss,
        }
    

    def validation_step(self, batch, batch_idx):

        loss_tuple = self.model(batch)
        clip_loss = loss_tuple[0].mean()
        cif_loss = loss_tuple[1].mean()
        feature_loss = loss_tuple[2].mean()

        # Log additional loss terms as well
        self.log("val_loss", clip_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_cif_loss", cif_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_feature_loss", feature_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {
            "loss": clip_loss,
            "clip_loss": clip_loss,
            "cif_loss": cif_loss,
            "feature_loss": feature_loss,
        }


    def test_step(self, batch, batch_idx):
        tokenizer, cif_embeddings, feature_embeddings = self.sample(batch)
        results = []
        for token, cif_emb, feature_emb in zip(tokenizer, cif_embeddings, feature_embeddings):
            token = re.sub(r'_\d+$', '', token)
            results.append((token, cif_emb, feature_emb))

        return {"test_results": results}

    def test_epoch_end(self, outputs):
        # 'outputs' is a list of dicts returned by test_step
        all_results = []
        for out in outputs:
            all_results.extend(out["test_results"])
        self.all_test_results = all_results
        return {"all_test_results": all_results}


    def configure_optimizers(self):
        params = [
            {"params": self.model.lattice_encoder.parameters(), "lr": self.cfg.lattice_encoder_lr, "weight_decay": self.cfg.weight_decay},
            {"params": self.model.atom_encoder.parameters(), "lr": self.cfg.atom_encoder_lr, "weight_decay": self.cfg.weight_decay},
            {"params": self.model.xrd_encoder.parameters(), "lr": self.cfg.xrd_encoder_lr, "weight_decay": self.cfg.weight_decay},
            {"params": self.model.metal_encoder.parameters(), "lr": self.cfg.metal_encoder_lr, "weight_decay": self.cfg.weight_decay},
            {"params": self.model.linker_encoder.parameters(), "lr": self.cfg.linker_encoder_lr, "weight_decay": self.cfg.weight_decay},
            {"params": itertools.chain(
                self.model.cif_projection.parameters(), self.model.feature_projection.parameters()
            ), "lr": self.cfg.head_lr, "weight_decay": self.cfg.weight_decay}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.)
        # ReduceLROnPlateau requires monitoring a validation metric
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.cfg.patience, factor=self.cfg.factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }
