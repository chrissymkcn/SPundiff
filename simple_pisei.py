import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample, pyro_method
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


class PhysicsInformedSpatialInverter(PyroModule):
    """
    Simplified physics-informed model for spatial transcriptomics diffusion correction.
    
    Key simplifications:
    1. Direct modeling without complex encoding/decoding
    2. Spot-specific diffusion coefficients based on tissue features
    3. Graph Laplacian diffusion with adaptive similarity
    4. Cell-based penalization
    """
    
    def __init__(self, 
                X: torch.Tensor,  # [n_spots, n_genes]
                y: torch.Tensor,  # [n_spots, n_genes]
                coords: torch.Tensor,
                spatial_adjacency: torch.Tensor,
                pca_emb: torch.Tensor = None,
                image_features: torch.Tensor = None,
                ecm_scores: torch.Tensor = None,
                cell_count: torch.Tensor = None,
                cluster_labels: torch.Tensor = None,
                diffusion_steps: int = 3,
                ):  # Mass conservation weight
        
        super().__init__()
        
        self.X = self.robust_log_transform(X)
        self.Y = y
        self.n_spots, self.n_genes = X.shape
        self.coords = coords
        self.diffusion_steps = diffusion_steps
        self.pca_emb = pca_emb
        
        pca_emb_norm = (pca_emb - pca_emb.mean(dim=0, keepdim=True)) / (pca_emb.std(dim=0, keepdim=True) + 1e-6)
        cov_matrix = torch.mm(pca_emb_norm.t(), pca_emb_norm) / (pca_emb_norm.size(0))
        cov_matrix += 1e-2 * torch.eye(cov_matrix.size(0), device=pca_emb.device)  # Ensure positive definiteness
        self.cov_matrix = torch.linalg.cholesky(cov_matrix)
        
        # Store masks and features
        self.cell_count = cell_count if cell_count is not None else torch.ones(self.n_spots, dtype=torch.bool)
        self.cluster_labels = cluster_labels
        
        # Adaptive Laplacian matrices
        self.spatial_laplacian = self._compute_spatial_laplacian(spatial_adjacency)
        # pca_similarity = torch.cdist(pca_emb, pca_emb)
        # self.adaptive_laplacian = self._compute_adaptive_laplacian(
        #     pca_similarity if pca_similarity is not None else torch.eye(self.n_spots),
        #     spatial_adjacency
        # )
        self.adaptive_laplacian = None
        
        # Spot-specific diffusion coefficient encoder
        feature_dim = 0
        if image_features is not None:
            feature_dim += image_features.shape[1]
        if ecm_scores is not None:
            feature_dim += ecm_scores.shape[1]
            
        if feature_dim > 0:
            self.diffusion_encoder = SpotDiffusionEncoder(
                feature_dim=feature_dim,
                hidden_dim=64
            )
            # Concatenate features
            features = []
            if image_features is not None:
                features.append(image_features)
            if ecm_scores is not None:
                features.append(ecm_scores)
            self.img_features = torch.cat(features, dim=1)
        else:
            self.diffusion_encoder = None
            self.img_features = None
            
    def _compute_spatial_laplacian(self, adjacency):
        """Compute normalized graph Laplacian from adjacency matrix"""
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(0)  # Remove self-loops
        adjacency = adjacency / (adjacency.sum(dim=1, keepdim=True) + 1e-10)  # Normalize rows
        # Degree matrix
        degree = torch.diag(adjacency.sum(dim=1))
        laplacian = degree - adjacency        
        return laplacian
    
    def _compute_adaptive_laplacian(self, similarity, spatial_adjacency):
        """
        Compute expression-similarity based Laplacian constrained to spatial neighbors
        """
        # Apply spatial constraint - only keep similarities for spatial neighbors
        spatial_adjacency.fill_diagonal_(0)  # Remove self-loops
        similarity.fill_diagonal_(0)  # Remove self-loops
        similarity = similarity / similarity.max()  # Normalize to [0, 1]
        similarity = similarity * spatial_adjacency
        
        # Convert to Laplacian
        degree = torch.diag(similarity.sum(dim=1))
        adaptive_laplacian = degree - similarity
        
        return adaptive_laplacian
    
    def forward_diffusion(self, true_expression, spot_diffusion_rates):
        """
        Apply graph Laplacian diffusion with spot-specific rates
        
        Args:
            true_expression: [n_spots, n_genes]
            spot_diffusion_rates: [n_spots] - learned diffusion coefficients
        """
        expression = true_expression.clone()
        
        # Combine spatial and adaptive Laplacians
        # combined_laplacian = 0.5 * self.spatial_laplacian + 0.5 * self.adaptive_laplacian
        combined_laplacian = self.spatial_laplacian
        
        for _ in range(self.diffusion_steps):
            # Apply diffusion: x_new = x - D * L * x
            diffusion_update = torch.matmul(combined_laplacian, expression)
            expression = expression - spot_diffusion_rates.unsqueeze(1) * diffusion_update
            expression = torch.clamp(expression, min=0.0)
            
        return expression
    
    def robust_log_transform(self, x, clip_quantile=0.97):
        # Clip extreme values
        upper = torch.quantile(x, clip_quantile)
        x_clipped = torch.clamp(x, max=upper, min=1e-06)
        x_clipped = torch.log(x_clipped)
        # Log transform
        return x_clipped

    @pyro_method
    def model(self):
        """
        Pyro probabilistic model
        
        Args:
            self.Y: [n_spots, n_genes]
            total_counts: [n_spots]
        """
        with pyro.plate("spots", self.n_spots):
                spot_diffusion_rates = pyro.sample(
                    "spot_diffusion_rates",
                    dist.Beta(2.0, 8.0)
                )
        
        # Sample true undiffused expression
        with pyro.plate("spots_expr", self.n_spots, dim=-2):
            with pyro.plate("genes", self.n_genes, dim=-1):
                true_expression = pyro.sample(
                    "true_expression",
                    dist.LogNormal(
                        self.X,
                        torch.ones_like(self.Y) * 0.1  # Small scale for stability
                    )
                )
        
        # # to test: consider multivariate normal logtransformed?
        # with pyro.plate('spots_expr', self.n_genes):
        #     true_expression = pyro.sample(
        #         'true_expression',
        #         dist.TransformDistribution(
        #             dist.MultivariateNormal(
        #                 loc=self.X.T,
        #                 scale_tril=self.cov_matrix
        #             ),
        #             transform=dist.transforms.ExpTransform()
        #         )
        #     )
        # true_expression = true_expression.T  # [n_spots, n_genes]
        
        # Forward diffusion process
        diffused_expression = self.forward_diffusion(true_expression, spot_diffusion_rates)
        
        # Observation model
        with pyro.plate("obs_spots", self.n_spots, dim=-2):
            with pyro.plate("obs_genes", self.n_genes, dim=-1):
                pyro.sample(
                    "obs",
                    dist.Poisson(diffused_expression),
                    obs=self.Y
                )
        
        # Physics-based penalties
        self._add_physics_penalties(true_expression)
        
        return true_expression
    
    def _add_physics_penalties(self, true_expression):
        """Add physics-based penalty terms"""
        
        # 1. Mass conservation penalty
        true_total = true_expression.sum(dim=1)
        obs_total = self.Y.sum(dim=1)
        mass_conservation_loss = torch.sum((true_total - obs_total) ** 2) / torch.sum(obs_total ** 2)  # Normalize by total observed expression
        pyro.factor("mass_conservation", - 1.0 * mass_conservation_loss)
        
        # 2. Spatial smoothness penalty using combined Laplacian
        if self.adaptive_laplacian is not None:
            combined_laplacian = 0.9 * self.spatial_laplacian + 0.1 * self.adaptive_laplacian
        else:
            combined_laplacian = self.spatial_laplacian
        observed_smoothness = torch.trace(self.Y.T @ combined_laplacian @ self.Y)
        smoothness_penalty = torch.trace(true_expression.T @ combined_laplacian @ true_expression)
        smoothness_penalty = smoothness_penalty / (observed_smoothness + 1e-8)  # Avoid division by zero
        pyro.factor("spatial_smoothness", - 1.0 * smoothness_penalty)
        
        # 3. Non-cell penalty - penalize expression in non-cell containing spots
        if self.cell_count is not None:
            non_cell_expression_dist = self.cell_count.float() / (self.cell_count.sum() + 1e-8)  # Normalize by total cell count
            non_cell_expression_dist = non_cell_expression_dist.unsqueeze(1)
            norm_true_expression = true_expression.sum(dim=1)
            norm_true_expression = norm_true_expression / (norm_true_expression.sum())
            # Compute non-cell penalty as KL divergence
            cell_count_penalty = torch.mean(
                torch.abs(norm_true_expression - non_cell_expression_dist)
            )
            pyro.factor("non_cell_penalty", - 1.0 * cell_count_penalty)
            
        # 4. Cluster separation loss
        if self.cluster_labels is not None:
            cluster_separation_loss = self._cluster_separation_loss(true_expression, self.cluster_labels)
            pyro.factor("cluster_separation", - 1.0 * cluster_separation_loss)

    def _cluster_separation_loss(self, true_expr: torch.Tensor, cluster_labels: torch.Tensor) -> torch.Tensor:
        """Maximize between-cluster distance, minimize within-cluster distance"""
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        if n_clusters < 2:
            return torch.tensor(0.0)
        
        # Compute cluster centroids
        centroids = []
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            centroid = true_expr[:, mask].mean(dim=1)
            centroids.append(centroid)
        centroids = torch.stack(centroids)
        
        # Between-cluster distance (maximize)
        between_dist = torch.pdist(centroids, p=2).mean()
        
        # Within-cluster distance (minimize)
        within_dist = 0.0
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            cluster_expr = true_expr[:, mask]
            if cluster_expr.shape[1] > 1:
                centroid = centroids[i].unsqueeze(1)
                within_dist += torch.norm(cluster_expr - centroid, p=2, dim=0).mean()
        
        # Minimize within/between ratio
        return within_dist / (between_dist + 1e-6)
    
    @pyro_method
    def guide(self):
        """Variational guide"""
        
        # Variational parameters for diffusion rates
        if self.diffusion_encoder is not None:
            # Encode tissue features to diffusion parameters
            diffusion_params = self.diffusion_encoder(self.img_features)
            alpha_param = F.softplus(diffusion_params[:, 0]) + 1.0
            beta_param = F.softplus(diffusion_params[:, 1]) + 1.0
        else:
            alpha_param = pyro.param("alpha_param", torch.ones(self.n_spots) * 2.0, 
                                constraint=dist.constraints.positive)
            beta_param = pyro.param("beta_param", torch.ones(self.n_spots) * 8.0,
                                constraint=dist.constraints.positive)
        
        with pyro.plate("spots", self.n_spots):
            spot_diffusion_rates = pyro.sample(
                "spot_diffusion_rates",
                dist.Beta(alpha_param, beta_param)
            )
        
        # Variational parameters for true expression
        loc_param = pyro.param("loc_param", self.robust_log_transform(self.Y))
        scale_param = pyro.param("scale_param", torch.ones_like(self.Y) * 0.1,
                            constraint=dist.constraints.positive)
        
        with pyro.plate("spots_expr", self.n_spots, dim=-2):
            with pyro.plate("genes", self.n_genes, dim=-1):
                true_expression = pyro.sample(
                    "true_expression",
                    dist.LogNormal(loc_param,
                                scale_param)
                )
        # # to test: consider multivariate normal logtransformed?
        # loc_param = pyro.param("loc_param", self.X.T)
        # scale_tril_param = pyro.param("scale_tril_param", torch.eye(self.cov_matrix.size(0)), constraint=dist.constraints.lower_cholesky)
        # with pyro.plate('spots_expr', self.n_genes):
        #     true_expression = pyro.sample(
        #         "true_expression",
        #         dist.TransformDistribution(
        #             dist.MultivariateNormal(
        #                 loc=loc_param.T,
        #                 scale_tril=scale_tril_param
        #             ),
        #             transform=dist.transforms.ExpTransform()
        #         )
        #     )
        # true_expression = true_expression.T  # [n_spots, n_genes]
        
        return true_expression


class SpotDiffusionEncoder(PyroModule):
    """
    Encoder that learns spot-specific diffusion coefficients from tissue features
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Output alpha, beta for Beta distribution
        )
        
    def forward(self, features):
        """
        Args:
            features: [n_spots, feature_dim] - concatenated image and ECM features
        Returns:
            params: [n_spots, 2] - (alpha, beta) parameters for Beta distribution
        """
        return self.encoder(features)

