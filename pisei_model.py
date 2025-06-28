import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np

class PhysicsInformedSpatialInverter(nn.Module):
    """
    Physics-informed model for recovering undiffused gene expression from 
    observed spatially diffused data, inspired by SpotClean but with:
    1. Physics constraints (diffusion distance limits)
    2. Clustering optimization
    3. PyTorch/Pyro implementation for speed
    """
    
    def __init__(self, 
                n_genes: int,
                n_spots: int,
                spatial_dim: int = 2,
                max_diffusion_distance: float = 100.0,  # micrometers
                min_diffusion_distance: float = 10.0,
                hidden_dim: int = 128,
                n_clusters: int = 10,
                kernel_type: str = "physics_gaussian"):
        
        super().__init__()
        
        self.n_genes = n_genes
        self.n_spots = n_spots
        self.spatial_dim = spatial_dim
        self.max_diffusion_distance = max_diffusion_distance
        self.min_diffusion_distance = min_diffusion_distance
        self.n_clusters = n_clusters
        self.kernel_type = kernel_type
        
        # 1. SPATIAL GRAPH ENCODER - captures local tissue structure
        self.spatial_encoder = SpatialGraphEncoder(
            input_dim=n_genes, 
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # 2. PHYSICS-CONSTRAINED DIFFUSION MODULE
        self.diffusion_module = PhysicsConstrainedDiffusion(
            max_distance=max_diffusion_distance,
            min_distance=min_diffusion_distance,
            kernel_type=kernel_type
        )
        
        # 3. EXPRESSION DECODER with clustering regularization
        self.expression_decoder = ClusterAwareDecoder(
            input_dim=hidden_dim,
            output_dim=n_genes,
            n_clusters=n_clusters
        )
        
        # 4. UNCERTAINTY QUANTIFICATION
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, n_genes),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
        # Learnable parameters
        self.bleed_rate = nn.Parameter(torch.tensor(0.3))  # From SpotClean insights
        self.distal_rate = nn.Parameter(torch.tensor(0.1))
        
    def model(self, observed_expression, spatial_coords, tissue_mask, spot_graph):
        """Pyro probabilistic model"""
        
        batch_size, n_spots, n_genes = observed_expression.shape
        
        # Physics-informed priors
        with pyro.plate("genes", n_genes):
            # Gene-specific diffusion rates (some genes diffuse more than others)
            gene_diffusion_rates = pyro.sample(
                "gene_diffusion_rates",
                dist.Beta(2.0, 8.0)  # Most genes have low diffusion
            )
        
        with pyro.plate("spots", n_spots):
            # Spot-specific contamination rates
            spot_contamination = pyro.sample(
                "spot_contamination",
                dist.Beta(3.0, 7.0)  # ~30% contamination as in SpotClean
            )
        
        # True undiffused expression (what we want to recover)
        with pyro.plate("batch", batch_size):
            with pyro.plate("spots_batch", n_spots):
                with pyro.plate("genes_batch", n_genes):
                    true_expression = pyro.sample(
                            "true_expression",
                            dist.LogNormal(
                                torch.zeros(batch_size, n_spots, n_genes),
                                torch.ones(batch_size, n_spots, n_genes)
                            )
                    )
        
        # Physics-constrained diffusion process
        diffused_expression = self.forward_diffusion(
            true_expression, 
            spatial_coords, 
            gene_diffusion_rates,
            spot_contamination,
            tissue_mask
        )
        
        # Observation model with noise
        with pyro.plate("obs_batch", batch_size):
            with pyro.plate("obs_spots", n_spots):
                with pyro.plate("obs_genes", n_genes):
                    pyro.sample(
                        "obs",
                        dist.Poisson(diffused_expression + 1e-6),
                        obs=observed_expression
                    )
        
        return true_expression
    
    def guide(self, observed_expression, spatial_coords, tissue_mask, spot_graph):
        """Variational approximation"""
        
        batch_size, n_spots, n_genes = observed_expression.shape
        
        # Encode spatial context
        spatial_features = self.spatial_encoder(observed_expression, spot_graph)
        
        # Decode to parameters of variational distribution
        decoder_output = self.expression_decoder(spatial_features)
        uncertainty = self.uncertainty_head(spatial_features)
        
        # Variational parameters
        loc = decoder_output["mean_expression"]
        scale = uncertainty + 1e-6
        
        # Sample true expression from variational distribution
        with pyro.plate("batch", batch_size):
            with pyro.plate("spots_batch", n_spots):
                with pyro.plate("genes_batch", n_genes):
                    true_expression = pyro.sample(
                        "true_expression",
                        dist.LogNormal(loc, scale)
                    )
        
        # Gene and spot level parameters
        with pyro.plate("genes", n_genes):
            gene_diffusion_rates = pyro.sample(
                "gene_diffusion_rates",
                dist.Beta(
                    torch.ones(n_genes) * 2.0,
                    torch.ones(n_genes) * 8.0
                )
            )
        
        with pyro.plate("spots", n_spots):
            spot_contamination = pyro.sample(
                "spot_contamination", 
                dist.Beta(
                    torch.ones(n_spots) * 3.0,
                    torch.ones(n_spots) * 7.0
                )
            )
        
        return {
            "true_expression": true_expression,
            "uncertainty": uncertainty,
            "cluster_assignments": decoder_output["cluster_logits"]
        }
    
    def forward_diffusion(self, true_expression, spatial_coords, 
                        gene_diffusion_rates, spot_contamination, tissue_mask):
        """Physics-constrained forward diffusion model"""
        
        # Build physics-constrained diffusion matrix
        diffusion_matrix = self.diffusion_module(
            spatial_coords, 
            gene_diffusion_rates,
            tissue_mask
        )
        
        # Apply diffusion: observed = (1-bleed_rate) * true + bleed_rate * diffused
        batch_size, n_spots, n_genes = true_expression.shape
        
        # Vectorized diffusion for all genes
        diffused = torch.einsum('bij,bkj->bik', diffusion_matrix, true_expression)
        
        # Combine stayed + diffused expression
        stayed = true_expression * (1 - self.bleed_rate)
        leaked = diffused * self.bleed_rate
        
        return stayed + leaked


class SpatialGraphEncoder(nn.Module):
    """Graph neural network for encoding spatial relationships"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.gconv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.gconv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gconv3 = GCNConv(hidden_dim, output_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, gene_expression, graph):
        # gene_expression: [batch_size, n_spots, n_genes]
        # graph: PyG Data object with edge_index
        
        batch_size = gene_expression.size(0)
        outputs = []
        
        for b in range(batch_size):
            x = gene_expression[b]  # [n_spots, n_genes]
            
            # Graph convolutions with residual connections
            h1 = F.relu(self.gconv1(x, graph.edge_index))
            h1 = self.norm1(h1 + x if x.size(-1) == h1.size(-1) else h1)
            h1 = self.dropout(h1)
            
            h2 = F.relu(self.gconv2(h1, graph.edge_index))
            h2 = self.norm2(h2 + h1)
            h2 = self.dropout(h2)
            
            h3 = self.gconv3(h2, graph.edge_index)
            outputs.append(h3)
        
        return torch.stack(outputs, dim=0)


class PhysicsConstrainedDiffusion(nn.Module):
    """Physics-informed diffusion kernel with distance constraints"""
    
    def __init__(self, max_distance, min_distance, kernel_type="physics_gaussian"):
        super().__init__()
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.kernel_type = kernel_type
        
        # Learnable physics parameters
        self.diffusion_coefficient = nn.Parameter(torch.tensor(1.0))
        self.tissue_permeability = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, spatial_coords, gene_diffusion_rates, tissue_mask):
        """
        Build physics-constrained diffusion matrix
        
        Args:
            spatial_coords: [batch_size, n_spots, 2] - (x, y) coordinates in micrometers
            gene_diffusion_rates: [n_genes] - per-gene diffusion rates
            tissue_mask: [batch_size, n_spots] - 1 for tissue, 0 for background
        """
        batch_size, n_spots, _ = spatial_coords.shape
        
        # Compute pairwise distances
        distances = torch.cdist(spatial_coords, spatial_coords, p=2)  # [batch, n_spots, n_spots]
        
        # Physics constraint: zero diffusion beyond max distance
        distance_mask = (distances <= self.max_distance) & (distances >= self.min_distance)
        
        # Tissue constraint: no diffusion from/to background spots
        tissue_mask_expanded = tissue_mask.unsqueeze(-1) * tissue_mask.unsqueeze(-2)
        
        # Combined mask
        valid_mask = distance_mask & tissue_mask_expanded
        
        # Physics-based kernel (modified Gaussian with decay)
        if self.kernel_type == "physics_gaussian":
            # Simulate mRNA diffusion with realistic decay
            sigma = self.max_distance / 3.0  # 95% within max distance
            kernel = torch.exp(-distances**2 / (2 * sigma**2))
            
            # Add distance-dependent permeability
            permeability_factor = torch.exp(-distances / (self.tissue_permeability * self.max_distance))
            kernel = kernel * permeability_factor
            
        elif self.kernel_type == "exponential_decay":
            # Exponential decay model
            decay_rate = 3.0 / self.max_distance
            kernel = torch.exp(-decay_rate * distances)
        
        # Apply constraints
        kernel = kernel * valid_mask.float()
        
        # Normalize rows (conservation of mass)
        row_sums = kernel.sum(dim=-1, keepdim=True)
        kernel = kernel / (row_sums + 1e-8)
        
        return kernel


class ClusterAwareDecoder(nn.Module):
    """Decoder that encourages clustering while reconstructing expression"""
    
    def __init__(self, input_dim, output_dim, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        
        # Expression reconstruction branch
        self.expression_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
            nn.Softplus()  # Ensures positive expression
        )
        
        # Clustering branch
        self.cluster_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, n_clusters)
        )
        
    def forward(self, spatial_features):
        # Expression reconstruction
        mean_expression = self.expression_head(spatial_features)
        
        # Cluster assignments
        cluster_logits = self.cluster_head(spatial_features)
        
        return {
            "mean_expression": mean_expression,
            "cluster_logits": cluster_logits
        }


class PISEITrainer:
    """Training wrapper with custom loss functions"""
    
    def __init__(self, model, lr=1e-3, cluster_weight=0.1, physics_weight=1.0):
        self.model = model
        self.optimizer = Adam({"lr": lr})
        self.svi = SVI(model.model, model.guide, self.optimizer, loss=Trace_ELBO())
        self.cluster_weight = cluster_weight
        self.physics_weight = physics_weight
        
    def train_step(self, observed_expression, spatial_coords, tissue_mask, spot_graph):
        """Single training step with combined losses"""
        
        # Main ELBO loss
        elbo_loss = self.svi.step(observed_expression, spatial_coords, tissue_mask, spot_graph)
        
        # Additional clustering regularization
        with torch.no_grad():
            guide_trace = poutine.trace(self.model.guide).get_trace(
                observed_expression, spatial_coords, tissue_mask, spot_graph
            )
            cluster_logits = guide_trace.nodes["cluster_logits"]["value"]
            
        # Clustering loss (encourage separation)
        cluster_probs = F.softmax(cluster_logits, dim=-1)
        cluster_entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=-1)
        cluster_loss = cluster_entropy.mean()
        
        # Total loss
        total_loss = elbo_loss + self.cluster_weight * cluster_loss
        
        return {
            "total_loss": total_loss,
            "elbo_loss": elbo_loss,
            "cluster_loss": cluster_loss
        }
    
    def physics_regularization(self, true_expression, observed_expression, 
                            spatial_coords, tissue_mask):
        """Physics-based regularization terms"""
        
        # Conservation of mass: total expression should be preserved
        mass_conservation = torch.abs(
            true_expression.sum() - observed_expression.sum()
        ) / observed_expression.sum()
        
        # Spatial smoothness: neighboring spots should have similar expression
        # (but only for tissue spots)
        return mass_conservation


# Example usage and training loop
def create_model_and_train():
    # Model hyperparameters
    n_genes = 2000
    n_spots = 1000
    max_diffusion_distance = 100.0  # micrometers
    n_clusters = 15
    
    # Create model
    model = PhysicsInformedSpatialInverter(
        n_genes=n_genes,
        n_spots=n_spots,
        max_diffusion_distance=max_diffusion_distance,
        n_clusters=n_clusters
    )
    
    # Create trainer
    trainer = PISEITrainer(model, lr=1e-3, cluster_weight=0.1)
    
    return model, trainer

# Advanced features for optimization:
def information_maximization_loss(cluster_assignments, true_expression):
    """Maximize mutual information between clusters and expression patterns"""
    # Implement InfoNCE or similar MI estimation
    pass

def total_variation_regularizer(expression, spatial_coords):
    """Spatial smoothness regularization"""
    # Penalize large expression gradients in space
    pass