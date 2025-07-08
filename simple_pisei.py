import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample, pyro_method
import torch.nn.functional as F
import numpy as np

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
        
        self.X = X
        self.Y = y
        self.n_spots, self.n_genes = X.shape
        self.coords = coords
        self.diffusion_steps = diffusion_steps
        self.pca_emb = pca_emb
        # Compute and store total counts per spot (observed)
        self.total_counts = X.sum(dim=1, keepdim=True)  # [n_spots, 1]
        
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
        self.scale_factor = self._compute_simple_data_scale()

    def _compute_simple_data_scale(self):
        """Simple, fast scale computation from data statistics"""
        # Use simple data statistics that are fast to compute
        n_total = self.n_spots * self.n_genes
        sum_Y = self.Y.sum().item()
        mean_Y = self.Y.mean().item()
        max_Y = self.Y.max().item()
        
        # Rough Poisson likelihood magnitude: sum_Y * log(mean_Y)
        likelihood_magnitude = sum_Y * np.log(mean_Y + 1)
        
        # Simple penalty magnitude estimate: roughly O(1) for normalized penalties
        typical_penalty_magnitude = 1.0
        
        # Scale to make penalties ~5% of likelihood
        scale_factor = likelihood_magnitude * 0.1 / typical_penalty_magnitude
        
        # Add adjustment for count ranges
        if max_Y > 1000:
            scale_factor *= 2
        elif max_Y < 10:
            scale_factor *= 0.5
        
        final_scale = max(100.0, min(scale_factor, 50000.0))
        
        print(f"Simple data scale: {final_scale:.0f} (likelihood est: {likelihood_magnitude:.0f})")
        return final_scale
    
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
    
    def robust_log_transform(self, x, clip_quantile=0.99):
        # Clip extreme values
        upper = torch.quantile(x, clip_quantile)
        x_clipped = torch.clamp(x, max=upper, min=1e-06)
        x_clipped = torch.log(x_clipped)
        # Log transform
        return x_clipped
    
    @pyro_method
    def model(self):
        with pyro.plate("spot_diffrate", self.n_spots):
                spot_diffusion_rates = pyro.sample(
                    "spot_diffusion_rates",
                    dist.Beta(1.1, 8.0)
                )
        
        # Sample true undiffused expression
        with pyro.plate("spots_expr", self.n_spots, dim=-2):
            with pyro.plate("genes", self.n_genes, dim=-1):
                true_expression = pyro.sample(
                    "true_expression",
                    dist.LogNormal(
                        self.robust_log_transform(self.X),
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
        
        # # Observation model
        # with pyro.plate("obs_spots", self.n_spots, dim=-2):
        #     with pyro.plate("obs_genes", self.n_genes, dim=-1):
        #         pyro.sample(
        #             "obs",
        #             dist.Poisson(diffused_expression),
        #             obs=self.Y
        #         )
        
    
        # Overdispersion parameter for Negative Binomial
        concentration = pyro.param("concentration", torch.ones_like(self.Y) * 10.0,  # Start with moderate overdispersion
                            constraint=dist.constraints.positive)
        
        # Observation model with Negative Binomial
        with pyro.plate("obs_spots", self.n_spots, dim=-2):
            with pyro.plate("obs_genes", self.n_genes, dim=-1):
                logits = torch.log(concentration + 1e-8) - torch.log(diffused_expression + 1e-8)
                pyro.sample(
                    "obs",
                    dist.NegativeBinomial(
                        total_count=concentration,  # Dispersion parameter
                        logits=logits  # Log-odds of success
                    ),
                    obs=self.Y
                )
        
        # Physics-based penalties
        self._add_physics_penalties(true_expression)
        
        return true_expression
    
    def _add_physics_penalties(self, true_expression):
        """Add physics-based penalty terms with balanced weights"""
        scale_factor = self.scale_factor
        
        # # 1. Mass conservation penalty (keep current weight)
        # true_total = true_expression.sum(dim=0)
        # obs_total = self.Y.sum(dim=0)
        # mass_conservation_loss = torch.sum((true_total - obs_total) ** 2) / torch.sum(obs_total ** 2)
        # mass_conservation_loss = - mass_conservation_loss * scale_factor
        # # print(f"Mass conservation loss: {mass_conservation_loss.item()}")
        # pyro.factor("mass_conservation", mass_conservation_loss)  # Increased from 1.0
        
        # 2. Spatial smoothness penalty (reduce weight)
        if self.adaptive_laplacian is not None:
            combined_laplacian = 0.9 * self.spatial_laplacian + 0.1 * self.adaptive_laplacian
        else:
            combined_laplacian = self.spatial_laplacian
        observed_smoothness = torch.trace(self.Y.T @ combined_laplacian @ self.Y)
        smoothness_penalty = torch.trace(true_expression.T @ combined_laplacian @ true_expression)
        smoothness_penalty = smoothness_penalty / (observed_smoothness + 1e-8)
        smoothness_penalty = - 0.1 * smoothness_penalty * scale_factor  # Reduced from 1.0
        # print(f"Spatial smoothness penalty: {smoothness_penalty.item()}")
        pyro.factor("spatial_smoothness", smoothness_penalty)  # Reduced from 1.0
        
        # 3. Non-cell penalty (increase base penalty calculation)
        if self.cell_count is not None:
            cell_count_dist = self.cell_count.float() / (self.cell_count.sum() + 1e-8)
            norm_true_expression = true_expression.sum(dim=1)
            norm_true_expression = norm_true_expression / (norm_true_expression.sum() + 1e-8)
            
            # Add small epsilon for numerical stability
            eps = 1e-8
            target_dist = cell_count_dist.squeeze() + eps
            pred_dist = norm_true_expression + eps
            
            # KL divergence: naturally scaled and meaningful
            kl_div = torch.sum(target_dist * torch.log(target_dist / pred_dist))
            cell_count_kl = - 2.0 * kl_div * scale_factor
            # Apply KL divergence (already well-scaled)
            pyro.factor("cell_count_kl", cell_count_kl)
                        
        # 4. Cluster separation loss (keep current)
        if self.cluster_labels is not None:
            cluster_separation_loss = self._cluster_separation_loss(true_expression, self.cluster_labels)
            cluster_separation_loss = - 0.2 * cluster_separation_loss * scale_factor
            pyro.factor("cluster_separation", cluster_separation_loss)
            
        # 6. Diffusion rate smoothness
        alpha = pyro.get_param_store()["alpha_param"]
        beta = pyro.get_param_store()["beta_param"]
        mean_diffrates = alpha / (alpha + beta)
        threshold = 0.15
        above_threshold = torch.relu(mean_diffrates - threshold)
        threshold_penalty = torch.sum(above_threshold ** 2) / torch.sum(mean_diffrates ** 2)
        threshold_penalty = - threshold_penalty * scale_factor
        # print(f"Diffusion rate threshold penalty: {threshold_penalty.item()}")
        pyro.factor("diffusion_rate_threshold", threshold_penalty)
        
        if torch.rand(1) < 0.01:  # Print occasionally
            print(
                # f"Penalties: mass={mass_conservation_loss.item():.3f}, ",
                f"spatial_smoothness={smoothness_penalty.item():.3f}, ",
                f'cell_count={cell_count_kl.item():.3f}, ',
                f"cluster={cluster_separation_loss.item()}, ",
                f"diffrate_threshold={threshold_penalty.item():.3f}",
                # f"entropy={entropy.item():.3f}"
                )
            
    def _cluster_separation_loss(self, true_expr: torch.Tensor, cluster_labels: torch.Tensor) -> torch.Tensor:
        """Maximize between-cluster distance, minimize within-cluster distance"""
        unique_clusters = np.unique(cluster_labels)  # Use torch.unique instead of np.unique
        n_clusters = len(unique_clusters)
        
        if n_clusters < 2:
            return torch.tensor(0.0)
        
        # Compute cluster centroids
        centroids = []
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id  # [n_spots] boolean mask
            # FIX: Apply mask to correct dimension (spots, not genes)
            cluster_expr = true_expr[mask, :]  # [n_spots_in_cluster, n_genes]
            centroid = cluster_expr.mean(dim=0)   # [n_genes] - average across spots in cluster
            centroids.append(centroid)
        centroids = torch.stack(centroids)  # [n_clusters, n_genes]
        
        # Between-cluster distance (maximize)
        between_dist = torch.pdist(centroids, p=2).mean()
        
        # Within-cluster distance (minimize)
        within_dist = 0.0
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            cluster_expr = true_expr[mask, :]  # FIX: Correct indexing
            if cluster_expr.shape[0] > 1:  # FIX: Check number of spots, not genes
                centroid = centroids[i].unsqueeze(0)  # [1, n_genes]
                within_dist += torch.norm(cluster_expr - centroid, p=2, dim=1).mean()  # FIX: dim=1 for genes
        
        # Minimize within/between ratio
        return within_dist / (between_dist + 1e-6)
    
    @pyro_method
    def guide(self):
        """Variational guide"""
        
        # Variational parameters for diffusion rates
        # if self.diffusion_encoder is not None:
        #     # Encode tissue features to diffusion parameters
        #     diffusion_params = self.diffusion_encoder(self.img_features)
        #     alpha_param = F.softplus(diffusion_params[:, 0]) 
        #     alpha_param = pyro.param("alpha_param", alpha_param, constraint=dist.constraints.positive)
        #     beta_param = F.softplus(diffusion_params[:, 1]) + 2.0
        #     beta_param = pyro.param("beta_param", beta_param, constraint=dist.constraints.positive)
        # else:
        alpha_param = pyro.param("alpha_param", torch.ones(self.n_spots) * 1.2, 
                            constraint=dist.constraints.positive)
        beta_param = pyro.param("beta_param", torch.ones(self.n_spots) * 10.0,
                            constraint=dist.constraints.positive)
        
        with pyro.plate("spot_diffrate", self.n_spots):
            spot_diffusion_rates = pyro.sample(
                "spot_diffusion_rates",
                dist.Beta(alpha_param, beta_param)
            )
        
        # Variational parameters for true expression
        loc_param = pyro.param("loc_param", self.robust_log_transform(self.X))
        scale_param = pyro.param("scale_param", torch.ones_like(self.X) * 0.1,
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

