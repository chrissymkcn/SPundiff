import torch.nn as nn
import torch
from torch.distributions import constraints
import torch.nn.functional as F

from .utils import EarlyStopping

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib import gp
from pyro.nn.module import PyroParam, pyro_method
import pyro.ops.stats as stats


# class PhysicsInformedDiffusionModel(nn.Module):
#     """
#     Physics-informed probabilistic model for spatial transcriptomics diffusion correction
#     Combines diffusion physics with probabilistic modeling using Pyro
#     """
#     def __init__(
#         self,
#         n_spots: int,
#         n_genes: int,
#         coords: torch.Tensor,
#         total_counts: torch.Tensor,
#         neighbors: torch.Tensor,  # Spatial neighborhood matrix
#         D_out: float = 0.1,
#         D_in: float = 0.02,
#         initial_counts_guess: torch.Tensor = None,
#         in_tiss_mask: torch.Tensor = None,
#         alpha: float = 0.8,  # Tissue mask weight
#         beta: float = 0.2,  # Spatial regularization weight
#     ):
#         super().__init__()
#         self.n_spots = n_spots
#         self.n_genes = n_genes
#         self.coords = coords
#         # self.neighbor_graph = neighbor_graph
#         self.alpha = alpha
#         self.beta = beta
#         self.total_counts = total_counts.unsqueeze(-1) if total_counts.dim() == 1 else total_counts  # Ensure shape is [n_spots, 1]
#         # self.total_counts = torch.log1p(self.total_counts)  # Log-transform to stabilize training
#         self.in_tiss_mask = in_tiss_mask if in_tiss_mask is not None else torch.ones(n_spots, dtype=torch.bool, device=coords.device)
        
#         if initial_counts_guess is not None:
#             self.initial_counts_guess = self.robust_log_transform(initial_counts_guess)  # Log-transform to stabilize training
#         else:
#             self.initial_counts_guess = torch.ones((n_spots, n_genes))
                
#         # Add RBF-FD specific initialization
#         self.neighbors = neighbors
#         self.D = D_out * (1 - in_tiss_mask) + D_in * in_tiss_mask

#     def robust_log_transform(self, x, clip_quantile=0.99):
#         # Clip extreme values
#         upper = torch.quantile(x, clip_quantile)
#         x_clipped = torch.clamp(x, max=upper)
#         # Log transform
#         return torch.log1p(x_clipped)

#     def compute_laplacian(self):
#         self.neighbors.fill_diagonal_(0)
#         degree = torch.diag(self.neighbors.sum(dim=1))
#         laplacian = degree - self.neighbors
#         D = pyro.param('D')
#         alpha_diag = torch.diag(D)  # Pyro param
#         return alpha_diag @ laplacian  

    
#     def forward_diffusion(self, heat, steps=10):
#         laplacian = self.compute_laplacian()  # Recompute each time
#         total_heat = heat.sum(dim=0, keepdim=True)
#         gene_specific_D = pyro.param("gene_specific_D")
#         for _ in range(steps):
#             laplacian_update = laplacian @ heat
#             laplacian_update = laplacian_update * gene_specific_D  # Pyro param
#             heat = heat - laplacian_update
#             heat = torch.clamp(heat, min=0.0)
#         heat = (heat / heat.sum(dim=0, keepdim=True)) * total_heat
#         return heat

#     def tissue_boundary_constraint(self, diffused_counts, observed_counts):
#         """
#         More sophisticated tissue boundary constraint that considers
#         distance-dependent diffusion patterns
#         """
#         # Compute distance to tissue boundary for each spot
#         boundary_distances = self.compute_boundary_distances()  # [n_spots]
        
#         # Expected decay pattern from observed data
#         observed_decay = self.compute_decay_pattern(
#             observed_counts, 
#             boundary_distances
#         )  # [n_bins, n_genes]
        
#         # Actual decay pattern from diffused counts
#         diffused_decay = self.compute_decay_pattern(
#             diffused_counts, 
#             boundary_distances
#         )  # [n_bins, n_genes]
        
#         # Compare decay patterns
#         decay_deviation = torch.nn.functional.mse_loss(
#             diffused_decay,
#             observed_decay
#         )
        
#         return -self.alpha * decay_deviation

#     def compute_boundary_distances(self):
#         """Compute distances to tissue boundary for each spot"""
#         from scipy.ndimage import distance_transform_edt
        
#         # Convert tissue mask to numpy for distance transform
#         mask_np = self.in_tiss_mask.cpu().numpy()
        
#         # Compute distance transform
#         dist_in = distance_transform_edt(mask_np)
#         dist_out = distance_transform_edt(1 - mask_np)
        
#         # Combine: negative outside tissue, positive inside
#         distances = torch.tensor(dist_in - dist_out, device=self.in_tiss_mask.device)
        
#         return distances
    
#     def compute_decay_pattern(self, counts, boundary_distances, n_bins=20):
#         """Compute expression decay pattern relative to tissue boundary"""
#         # Define distance bins
#         bins = torch.linspace(
#             boundary_distances.min(),
#             boundary_distances.max(),
#             n_bins + 1
#         )
        
#         # Compute mean expression for each distance bin
#         patterns = []
#         for i in range(n_bins):
#             mask = (boundary_distances >= bins[i]) & (boundary_distances < bins[i+1])
#             bin_mean = (counts[mask].mean(dim=0) if mask.any() else 
#                     torch.zeros(counts.size(1), device=counts.device))
#             patterns.append(bin_mean)
        
#         return torch.stack(patterns)  # [n_bins, n_genes]
    
#     def debug_tensor(self, name, tensor, print_stats=True):
#         has_nan = torch.isnan(tensor).any()
#         has_inf = torch.isinf(tensor).any()
#         if print_stats:
#             print(f"\n{name} stats:")
#             print(f"Shape: {tensor.shape}")
#             print(f"Range: [{tensor.min().item():.6e}, {tensor.max().item():.6e}]")
#             print(f"Mean: {tensor.mean().item():.6e}")
#             print(f"Has NaN: {has_nan}")
#             print(f"Has Inf: {has_inf}")
#         if has_nan or has_inf:
#             print(f"\nFound NaN/Inf in {name}!")
#             if has_nan:
#                 nan_indices = torch.where(torch.isnan(tensor))
#                 print(f"First few NaN positions: {[(i.item(), j.item()) for i, j in zip(*nan_indices[:2])][:5]}")
#             if has_inf:
#                 inf_indices = torch.where(torch.isinf(tensor))
#                 print(f"First few Inf positions: {[(i.item(), j.item()) for i, j in zip(*inf_indices[:2])][:5]}")
#             return True
#         return False

#     def test(
#         self,
#         observed_counts: torch.Tensor,
#         steps=10,
#     ) -> torch.Tensor:
#         """Probabilistic model with debugging"""

#         # 1. Check input tensors
#         print("\nChecking inputs:")
#         self.debug_tensor("observed_counts", observed_counts)
#         self.debug_tensor("in_tiss_mask", self.in_tiss_mask)
#         self.debug_tensor("neighbors", self.neighbors)
#         self.debug_tensor('initial_counts_guess', self.initial_counts_guess)
        
#         original_counts_loc = pyro.param(
#             "original_counts_loc",
#             self.initial_counts_guess,
#         )
#         original_counts_scale = pyro.param(
#             "original_counts_scale",
#             0.01 * torch.ones_like(observed_counts),
#             constraint=constraints.nonnegative
#         )
        
#         print("\nChecking parameters:")
#         self.debug_tensor("original_counts_loc", original_counts_loc)
#         self.debug_tensor("original_counts_scale", original_counts_scale)

#         # 3. Sample original counts
#         original_counts = pyro.sample(
#             "original_counts",
#             dist.LogNormal(original_counts_loc, original_counts_scale).to_event(2)
#         )
        
#         self.debug_tensor("original_counts (after sampling)", original_counts)

#         # 4. Check diffusion process
#         print("\nChecking diffusion process:")
#         # Debug Laplacian computation
#         L = self.compute_laplacian()
#         self.debug_tensor("Laplacian", L.to_dense() if L.is_sparse else L)

#         # Forward diffusion with intermediate checks
#         diffused_counts = original_counts.clone()
#         old_diffused = diffused_counts.clone()
#         diffused_counts = self.forward_diffusion(old_diffused, steps=steps)

#         self.debug_tensor("diffused_counts (after diffusion)", diffused_counts, print_stats=True)
#         # 5. Check total_counts and probs computation
#         print("\nChecking likelihood computation:")
#         total_counts = self.total_counts
#         self.debug_tensor("total_counts", total_counts)

#         # Compute probs with additional safety
#         probs = diffused_counts / total_counts
#         self.debug_tensor("probs (before clamping)", probs)
#         probs.clamp_(min=1e-6, max=1.0 - 1e-6)  # Ensure probabilities are valid
#         self.debug_tensor("probs (after clamping)", probs)
        
#         # 6. Sample observations
#         with pyro.plate("spots", self.n_spots):
#             obs = pyro.sample(
#                 "obs",
#                 dist.NegativeBinomial(
#                     total_count=total_counts,
#                     probs=probs
#                 ).to_event(1),
#                 obs=observed_counts
#             )
#         self.debug_tensor("obs", obs)

#         return {
#             "original_counts": original_counts,
#             "diffused_counts": diffused_counts,
#             "probs": probs,
#             'Laplacian': L.to_dense() if L.is_sparse else L,
#             "obs": obs
#         }
    
#     def model(
#         self,
#         observed_counts: torch.Tensor,  # Observed gene expression matrix [n_spots, n_genes]
#         diffusion_steps: int = 10,  # Number of diffusion steps
#     ) -> torch.Tensor:
#         """
#         Probabilistic model incorporating physics and data
        
#         Args:
#             observed_counts: Observed gene expression matrix [n_spots, n_genes]
#             in_tiss_mask: Binary mask for tissue locations [n_spots]
#             neighbors: Spatial neighborhood matrix [n_spots, n_spots]
#         """
#         D = pyro.param(
#             "D",
#             self.D,
#             constraint=constraints.nonnegative
#         )
#         gene_specific_D = pyro.param(
#             "gene_specific_D",
#             torch.ones(self.n_genes),
#             constraint=constraints.nonnegative
#         )
        
#         # Prior for original counts
#         original_counts_loc = pyro.param(
#             "original_counts_loc",
#             self.initial_counts_guess,
#         )
#         original_counts_scale = pyro.param(
#             "original_counts_scale",
#             0.01 * torch.ones_like(observed_counts),
#             constraint=torch.distributions.constraints.nonnegative
#         )  # [n_spots, n_genes]
        
#         # Sample original counts
#         original_counts = pyro.sample(
#             "original_counts",
#             dist.LogNormal(original_counts_loc, original_counts_scale).to_event(2)
#         )  # [n_spots, n_genes]
        
#         # Physics-based diffusion
#         diffused_counts = self.forward_diffusion(original_counts, steps=diffusion_steps)  # [n_spots, n_genes]
#         # Likelihood incorporating:
#         # 1. Observation model
#         # 2. Tissue mask constraint
#         # 3. Spatial regularization

#         # 1. Observation likelihood
#         total_counts = self.total_counts
#         probs = diffused_counts / total_counts
#         probs.clamp_(min=1e-6, max=1.0 - 1e-6)  # Ensure probabilities are valid
        
#         # Use Negative Binomial observation model
#         with pyro.plate("spots", self.n_spots):
#             pyro.sample(
#                 "obs",
#                 dist.NegativeBinomial(
#                     total_count=total_counts,  # shape: [n_spots, 1]
#                     probs=probs  # shape: [n_spots, n_genes]
#                 ).to_event(1),  # event_shape = [n_genes]
#                 obs=observed_counts
#             )
        
#         # Tissue boundary constraint
#         pyro.factor(
#             'tissue_constraint', self.tissue_boundary_constraint(
#                 diffused_counts, observed_counts)
#         )
        
#         # Spatial regularization using neighbor structure to encourage smoothness
#         # spatial_diff = torch.sum(
#         #     self.neighbors.unsqueeze(2) * 
#         #     (diffused_counts.unsqueeze(1) - diffused_counts.unsqueeze(0))**2
#         # )
#         # spatial_factor = -self.beta * spatial_diff
#         # pyro.factor("spatial_regularization", spatial_factor)
        
#         return original_counts

#     def guide(
#         self,
#         observed_counts: torch.Tensor,
#         diffusion_steps: int = 10,  # Number of diffusion steps
#     ) -> torch.Tensor:
#         """Variational guide/posterior"""
#         # Variational parameters
#         original_counts_loc = pyro.param(
#             "original_counts_loc",
#             self.initial_counts_guess,
#         )
#         original_counts_scale = pyro.param(
#             "original_counts_scale",
#             0.01 * torch.ones_like(observed_counts),
#             constraint=torch.distributions.constraints.nonnegative
#         )
        
#         # Variational distribution
#         return pyro.sample(
#             "original_counts",
#             dist.LogNormal(original_counts_loc, original_counts_scale).to_event(2)
#         )




#     # def setup_rbf_weights(self):
#     #     """Setup RBF-FD weights for spatial discretization"""
#     #     # Compute pairwise distances
#     #     dists = torch.cdist(self.coords, self.coords)
        
#     #     # Find nearest neighbors for each point
#     #     values, indices = torch.topk(dists, k=self.neighbors, dim=1, largest=False)
        
#     #     # Compute RBF shape parameter (epsilon)
#     #     epsilon = torch.median(values[:, 1])  # Use median distance to nearest neighbor
        
#     #     # Initialize weights matrix
#     #     lap_weights = torch.zeros_like(dists)
        
#     #     for i in range(self.n_spots):
#     #         # Get local coordinates
#     #         local_points = self.coords[indices[i]]  # [neighbors, 2]
#     #         center = self.coords[i]  # [2]
            
#     #         # Compute RBF matrix
#     #         A = self._rbf_matrix(local_points, epsilon)  # [neighbors, neighbors]
            
#     #         # Compute Laplacian values
#     #         b = self._rbf_laplacian_values(local_points, center, epsilon)  # [neighbors]
            
#     #         try:
#     #             # Solve for weights
#     #             weights = torch.linalg.solve(A, b)
#     #         except:
#     #             # Fallback to pseudo-inverse if system is ill-conditioned
#     #             weights = torch.linalg.pinv(A) @ b
            
#     #         # Store weights
#     #         lap_weights[i, indices[i]] = weights
        
#     #     # Register buffer (persistent state)
#     #     self.register_buffer('lap_weights', lap_weights)
#     #     self.register_buffer('neighbor_indices', indices)

#     # @staticmethod
#     # def _rbf_matrix(points, epsilon):
#     #     """Compute RBF interpolation matrix"""
#     #     dists = torch.cdist(points, points)
#     #     return torch.exp(-dists**2 / (2 * epsilon**2))
    
#     # @staticmethod
#     # def _rbf_laplacian_values(points, center, epsilon):
#     #     """Compute RBF Laplacian values"""
#     #     dists = torch.cdist(points.unsqueeze(0), center.unsqueeze(0)).squeeze()
#     #     r2 = dists**2
#     #     eps2 = epsilon**2
#     #     # For 2D Laplacian of Gaussian RBF
#     #     return torch.exp(-r2 / (2 * eps2)) * (r2 / eps2 - 2)
    
#     # def compute_laplacian(self, X: torch.Tensor) -> torch.Tensor:
#     #     """
#     #     Compute Laplacian using RBF-FD weights with spot-specific diffusion coefficients
#     #     """
#     #     # Get diffusion coefficients
#     #     D_values = self.D() if callable(self.D) else self.D  # [n_spots]
        
#     #     # Create diffusion coefficient matrix (average D between spots)
#     #     D_matrix = (D_values.unsqueeze(1) + D_values.unsqueeze(0)) / 2  # [n_spots, n_spots]
        
#     #     # Scale weights by diffusion coefficients
#     #     scaled_weights = self.lap_weights * D_matrix  #### not defined yet
        
#     #     # Convert to sparse tensor for efficiency
#     #     indices = torch.nonzero(scaled_weights).t()
#     #     values = scaled_weights[indices[0], indices[1]]
#     #     sparse_L = torch.sparse_coo_tensor(
#     #         indices, values, scaled_weights.shape,
#     #         device=X.device
#     #     )
        
#     #     # Return Laplacian operator
#     #     return sparse_L
    
#     # def forward_diffusion(self, X: torch.Tensor, steps: int = 1) -> torch.Tensor:
#     #     """
#     #     Solve diffusion equation using implicit scheme with RBF-FD discretization
#     #     """
#     #     X_t = X.clone()
#     #     dt = self.dt
        
#     #     # Pre-compute sparse matrices
#     #     I = torch.eye(self.n_spots, device=X.device)
#     #     L = self.compute_laplacian(X_t)
#     #     self.debug_tensor("Laplacian", L, print_stats=False)
#     #     # Add regularization
#     #     eps = 1e-6
#     #     A = I - dt * L.to_dense() + eps * I
        
#     #     try:
#     #         # LU factorization
#     #         LU, pivots = torch.linalg.lu_factor(A)
            
#     #         for _ in range(steps):
#     #             try:
#     #                 # Store original sum
#     #                 original_sum = X_t.sum(dim=0, keepdim=True)
                    
#     #                 # Solve system
#     #                 X_t_new = torch.linalg.lu_solve(LU, pivots, X_t)
                    
#     #                 # Ensure non-negativity
#     #                 X_t_new = torch.log1p(torch.exp(X_t_new))
                    
#     #                 # Mass conservation
#     #                 new_sum = X_t_new.sum(dim=0, keepdim=True)
#     #                 X_t = X_t_new * (original_sum / (new_sum + eps))
                    
#     #                 # Numerical stability
#     #                 X_t = X_t + eps
                    
#     #             except RuntimeError as e:
#     #                 print(f"Solver failed: {e}")
#     #                 # Fallback to explicit scheme
#     #                 X_t = X_t - dt * torch.sparse.mm(L, X_t)
#     #                 X_t = torch.log1p(torch.exp(X_t))
#     #                 X_t = X_t * (original_sum / (X_t.sum(dim=0, keepdim=True) + eps))
#     #                 X_t = X_t + eps
                    
#     #     except RuntimeError as e:
#     #         print(f"LU factorization failed: {e}")
#     #         # Direct sparse solve
#     #         for _ in range(steps):
#     #             original_sum = X_t.sum(dim=0, keepdim=True)
#     #             X_t = torch.linalg.solve(A, X_t)
#     #             X_t = torch.log1p(torch.exp(X_t))
#     #             X_t = X_t * (original_sum / (X_t.sum(dim=0, keepdim=True) + eps))
#     #             X_t = X_t + eps
        
#     #     return X_t
    
class PhysicsInformedDiffusionModel(nn.Module):
    """
    Physics-informed probabilistic model for spatial transcriptomics diffusion correction
    Incorporates gene-specific spatial covariance structure
    """
    def __init__(
        self,
        n_spots: int,
        n_genes: int,
        coords: torch.Tensor,
        total_counts: torch.Tensor,
        neighbors: torch.Tensor,
        D_out: float = 0.1,
        D_in: float = 0.02,
        initial_counts_guess: torch.Tensor = None,
        in_tiss_mask: torch.Tensor = None,
        alpha: float = 0.8,  # Tissue mask weight
        beta: float = 0.2,  # Spatial regularization weight
        pca_components: int = 20,  # Number of PCA components for covariance approximation
    ):
        super().__init__()
        self.n_spots = n_spots
        self.n_genes = n_genes
        self.coords = coords
        self.alpha = alpha
        self.beta = beta
        self.pca_components = min(pca_components, n_spots - 1)  # Ensure valid PCA components number
        
        # Store total counts with proper shape
        self.total_counts = total_counts.unsqueeze(-1) if total_counts.dim() == 1 else total_counts
        
        self.in_tiss_mask = in_tiss_mask if in_tiss_mask is not None else torch.ones(n_spots, dtype=torch.bool, device=coords.device)
        
        # Handle initial counts guess and compute PCA decomposition for covariance
        if initial_counts_guess is not None:
            self.initial_counts_guess = initial_counts_guess.clone()
            # Store the raw version and compute log-transformed version
            self.log_initial_counts = torch.log1p(initial_counts_guess)
            # Compute covariance structure for each gene
            self.compute_spatial_covariance()
        else:
            self.initial_counts_guess = torch.ones((n_spots, n_genes))
            self.log_initial_counts = torch.zeros((n_spots, n_genes))
            # Initialize with identity covariance
            self.setup_default_covariance()
                
        # Add RBF-FD specific initialization
        self.neighbors = neighbors
        self.D = D_out * (1 - in_tiss_mask) + D_in * in_tiss_mask
        
    def compute_spatial_covariance(self):
        """Compute spatial covariance structure using PCA decomposition"""
        # Center data for each gene
        centered_data = self.log_initial_counts - self.log_initial_counts.mean(dim=0, keepdim=True)
        
        # Initialize storage for PCA components and variances
        self.pca_components_list = []
        self.pca_variances_list = []
        
        # Process each gene separately
        for g in range(self.n_genes):
            gene_data = centered_data[:, g:g+1]  # Keep dim=1 for matrix ops
            
            try:
                # Compute covariance matrix for this gene across spots
                cov_matrix = gene_data @ gene_data.T / (self.n_spots - 1)
                
                # Add small regularization to ensure positive-definiteness
                cov_matrix = cov_matrix + torch.eye(self.n_spots, device=cov_matrix.device) * 1e-6
                
                # Perform eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
                
                # Sort in descending order
                sorted_indices = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
                
                # Keep only top components
                top_eigenvalues = eigenvalues[:self.pca_components].clamp(min=1e-6)
                top_eigenvectors = eigenvectors[:, :self.pca_components]
                
                # Store components and variances
                self.pca_components_list.append(top_eigenvectors)
                self.pca_variances_list.append(top_eigenvalues)
                
            except Exception as e:
                print(f"Error computing PCA for gene {g}: {e}")
                # Fall back to default simplified covariance
                identity_components = torch.eye(self.n_spots, self.pca_components, device=centered_data.device)
                identity_variances = torch.ones(self.pca_components, device=centered_data.device)
                
                self.pca_components_list.append(identity_components)
                self.pca_variances_list.append(identity_variances)
        
        # Convert lists to tensors for easier access
        self.pca_components_tensor = torch.stack(self.pca_components_list, dim=0)  # [n_genes, n_spots, pca_components]
        self.pca_variances_tensor = torch.stack(self.pca_variances_list, dim=0)    # [n_genes, pca_components]

    def setup_default_covariance(self):
        """Setup default covariance when no initial counts are provided"""
        self.pca_components_list = []
        self.pca_variances_list = []
        
        default_components = torch.eye(self.n_spots, self.pca_components, device=self.coords.device)
        default_variances = torch.ones(self.pca_components, device=self.coords.device)
        
        for _ in range(self.n_genes):
            self.pca_components_list.append(default_components)
            self.pca_variances_list.append(default_variances)
        
        self.pca_components_tensor = torch.stack(self.pca_components_list, dim=0)
        self.pca_variances_tensor = torch.stack(self.pca_variances_list, dim=0)

    def compute_laplacian(self):
        """Compute the Laplacian matrix for diffusion"""
        self.neighbors.fill_diagonal_(0)
        degree = torch.diag(self.neighbors.sum(dim=1))
        laplacian = degree - self.neighbors
        D = pyro.param('D', self.D, constraint=constraints.positive)
        alpha_diag = torch.diag(D)
        return alpha_diag @ laplacian
    
    def forward_diffusion(self, heat, steps=10):
        """Apply diffusion process to the input heat matrix"""
        laplacian = self.compute_laplacian()
        total_heat = heat.sum(dim=0, keepdim=True)
        gene_specific_D = pyro.param("gene_specific_D", torch.ones(self.n_genes), constraint=constraints.positive)
        
        for _ in range(steps):
            laplacian_update = laplacian @ heat
            laplacian_update = laplacian_update * gene_specific_D  # Gene-specific diffusion
            heat = heat - laplacian_update
            heat = torch.clamp(heat, min=0.0)
            
        # Normalize to maintain mass conservation
        heat = (heat / heat.sum(dim=0, keepdim=True)) * total_heat
        return heat

    def tissue_boundary_constraint(self, diffused_counts, observed_counts):
        """Constraint based on tissue boundary decay patterns"""
        # Compute distance to tissue boundary for each spot
        boundary_distances = self.compute_boundary_distances()
        
        # Expected decay pattern from observed data
        observed_decay = self.compute_decay_pattern(observed_counts, boundary_distances)
        
        # Actual decay pattern from diffused counts
        diffused_decay = self.compute_decay_pattern(diffused_counts, boundary_distances)
        
        # Compare patterns and return negative loss as a factor
        decay_deviation = torch.nn.functional.mse_loss(diffused_decay, observed_decay)
        return -self.alpha * decay_deviation

    def compute_boundary_distances(self):
        """Compute distances to tissue boundary for each spot"""
        from scipy.ndimage import distance_transform_edt
        
        # Convert tissue mask to numpy for distance transform
        mask_np = self.in_tiss_mask.cpu().numpy()
        
        # Compute distance transform
        dist_in = distance_transform_edt(mask_np)
        dist_out = distance_transform_edt(1 - mask_np)
        
        # Combine: negative outside tissue, positive inside
        distances = torch.tensor(dist_in - dist_out, device=self.in_tiss_mask.device)
        
        return distances
    
    def compute_decay_pattern(self, counts, boundary_distances, n_bins=20):
        """Compute expression decay pattern relative to tissue boundary"""
        # Define distance bins
        bins = torch.linspace(
            boundary_distances.min(),
            boundary_distances.max(),
            n_bins + 1
        )
        
        # Compute mean expression for each distance bin
        patterns = []
        for i in range(n_bins):
            mask = (boundary_distances >= bins[i]) & (boundary_distances < bins[i+1])
            bin_mean = (counts[mask].mean(dim=0) if mask.any() else 
                    torch.zeros(counts.size(1), device=counts.device))
            patterns.append(bin_mean)
        
        return torch.stack(patterns)  # [n_bins, n_genes]
    
    def model(
        self,
        observed_counts: torch.Tensor,  # Observed gene expression matrix [n_spots, n_genes]
        diffusion_steps: int = 10,  # Number of diffusion steps
    ) -> torch.Tensor:
        """
        Probabilistic model incorporating physics, data, and spatial covariance
        """
        # Global parameters for diffusion
        D = pyro.param(
            "D",
            self.D,
            constraint=constraints.positive
        )
        gene_specific_D = pyro.param(
            "gene_specific_D",
            torch.ones(self.n_genes),
            constraint=constraints.positive
        )
        
        # Gene-specific mean parameters
        gene_means = pyro.param(
            "gene_means",
            self.log_initial_counts.mean(dim=0),  # Initialize with mean of log counts
        )
        
        # Sample factor scores for low-rank spatial covariance
        original_counts = torch.zeros(self.n_spots, self.n_genes, device=observed_counts.device)
        
        # Process each gene separately with its own multivariate distribution
        for g in range(self.n_genes):
            # Sample latent factors (reduces dimensionality)
            with pyro.plate(f"gene_{g}_factors", self.pca_components):
                z = pyro.sample(
                    f"z_{g}",
                    dist.Normal(0, 1)
                )  # [pca_components]
                
            # Scale factors by eigenvalues
            scaled_z = z * torch.sqrt(self.pca_variances_tensor[g])  # [pca_components]
            
            # Project back to observation space
            gene_counts_centered = self.pca_components_tensor[g] @ scaled_z  # [n_spots]
            
            # Add gene-specific mean
            gene_counts = gene_counts_centered + gene_means[g]
            
            # Store in the output matrix
            original_counts[:, g] = gene_counts
        
        # Apply exponentiation to convert from log space
        original_counts = torch.exp(original_counts) - 1.0  # inverse of log1p
        
        # Register as a pyro random variable for posterior inference
        original_counts = pyro.deterministic("original_counts", original_counts)
        
        # Physics-based diffusion
        diffused_counts = self.forward_diffusion(original_counts, steps=diffusion_steps)
        
        # Ensure non-negativity
        diffused_counts = torch.clamp(diffused_counts, min=1e-8)
        
        # Compute probabilities for negative binomial model
        total_counts = self.total_counts
        probs = diffused_counts / (diffused_counts + total_counts)
        probs = probs.clamp(min=1e-6, max=1.0 - 1e-6)
        
        # Gene-specific dispersion parameters
        gene_dispersion = pyro.param(
            "gene_dispersion",
            torch.ones(self.n_genes),
            constraint=constraints.positive
        )
        
        # Use negative binomial observation model with gene-specific dispersion
        with pyro.plate("genes", self.n_genes):
            for g in range(self.n_genes):
                with pyro.plate(f"spots_gene_{g}", self.n_spots):
                    pyro.sample(
                        f"obs_gene_{g}",
                        dist.NegativeBinomial(
                            total_count=gene_dispersion[g],
                            probs=probs[:, g]
                        ),
                        obs=observed_counts[:, g]
                    )
        
        # Tissue boundary constraint
        pyro.factor(
            'tissue_constraint', 
            self.tissue_boundary_constraint(diffused_counts, observed_counts)
        )
        
        return original_counts

    def guide(
        self,
        observed_counts: torch.Tensor,
        diffusion_steps: int = 10,
    ) -> torch.Tensor:
        """
        Variational guide for the spatial covariance model
        """
        # Inferred factor means and scales for each gene
        factor_means = {}
        factor_scales = {}
        
        for g in range(self.n_genes):
            # Parameters for each gene's latent factors
            factor_means[g] = pyro.param(
                f"factor_means_{g}",
                torch.zeros(self.pca_components),
            )
            
            factor_scales[g] = pyro.param(
                f"factor_scales_{g}",
                torch.ones(self.pca_components),
                constraint=constraints.positive
            )
            
            # Sample latent factors for this gene
            with pyro.plate(f"gene_{g}_factors", self.pca_components):
                z = pyro.sample(
                    f"z_{g}",
                    dist.Normal(factor_means[g], factor_scales[g])
                )
        
        # Note: We don't need to explicitly return anything here as the 
        # sampled values are registered with pyro.sample()
        return None
    

class SparseGPRegression(GPModel):
    def __init__(
        self, X, y, kernel, 
        coords, in_tiss_mask, ttl_cnts,
        noise=None, approx=None, jitter=1e-6
    ):
        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
                y, torch.Tensor
            ), "y needs to be a torch Tensor instead of a {}".format(type(y))
            
        self.n_genes, self.n_spots = X.shape
        n_inducing = int(torch.sqrt(torch.tensor(self.n_genes)))  # number of inducing points
        mean_function = partial(self.mean_function, coords=coords, ref_count=y,
            in_tiss_mask=in_tiss_mask, ttl_cnts=ttl_cnts)
        super().__init__(X, y, kernel=kernel, mean_function=mean_function, jitter=jitter)
        X = torch.clamp(X, min=1e-6)  # Ensure non-negative counts
        self.X_init = X
        
        X_centered = X - X.mean(dim=1, keepdim=True)
        X_normalized = X_centered / (X_centered.std(dim=1, keepdim=True) + 1e-10)  # Normalize to unit variance, avoid division by zero
        # Calculate correlation matrix [n_spots, n_spots]
        correlation_matrix = torch.mm(X_normalized.t(), X_normalized) / self.n_genes
        # Add small diagonal regularization to ensure positive-definiteness
        epsilon = 1e-2
        correlation_matrix += epsilon * torch.eye(correlation_matrix.size(0), device=correlation_matrix.device)
        # Cholesky decomposition
        shared_scale_tril = torch.linalg.cholesky(correlation_matrix)

        self.X = pyro.nn.PyroSample(
            lambda self: dist.TransformedDistribution(
                base_distribution=dist.MultivariateNormal(
                    loc=X.log(),  # Transform to log space
                    scale_tril=shared_scale_tril
                ),
                transforms=[dist.transforms.ExpTransform()]  # Transform back to original space
            ).expand([self.n_genes]).to_event(1)
        )
        
        Xu = stats.resample(X.log(), n_inducing)  # Initialize in log-space
        self.Xu = PyroParam(Xu, constraint=constraints.real)
        assert not torch.isnan(self.Xu).any(), "Xu contains NaNs"
        assert not torch.isinf(self.Xu).any(), "Xu contains Infs"
        
        noise = self.X.new_tensor(1.0) if noise is None else noise
        self.noise = PyroParam(noise, constraints.real)

        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError(
                "The sparse approximation method should be one of "
                "'DTC', 'FITC', 'VFE'."
            )
        
        self.in_tiss_mask = in_tiss_mask
        self.ttl_cnts = ttl_cnts if isinstance(ttl_cnts, torch.Tensor) else torch.tensor(ttl_cnts, dtype=torch.float32)
        self.coords = coords

    @staticmethod
    def mean_function(X, coords, ref_count, in_tiss_mask, ttl_cnts):
        ## TO DO: Implement a proper mean function like forward diffusion in PhysicsInformedDiffusionModel

    @pyro_method
    def model(self):
        self.set_mode("model")

        # W = (inv(Luu) @ Kuf).T
        # Qff = Kfu @ inv(Kuu) @ Kuf = W @ W.T
        # Fomulas for each approximation method are
        # DTC:  y_cov = Qff + noise,                   trace_term = 0
        # FITC: y_cov = Qff + diag(Kff - Qff) + noise, trace_term = 0
        # VFE:  y_cov = Qff + noise,                   trace_term = tr(Kff-Qff) / noise
        # y_cov = W @ W.T + D
        # trace_term is added into log_prob

        N = self.X.size(0)
        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[:: M + 1] += self.jitter  # add jitter to the diagonal
        assert not torch.isnan(Kuu).any(), "Kuu contains NaNs"
        assert not torch.isinf(Kuu).any(), "Kuu contains Infs"
        Luu = torch.linalg.cholesky(Kuu)  # the Cholesky decomposition of Kuu = Luu @ Luu.T
        Kuf = self.kernel(self.Xu, self.X)
        W = torch.linalg.solve_triangular(Luu, Kuf, upper=False).t()  # W = inv(Luu).T @ Kuf = Kfu @ inv(Luu).T (an approximation of Kfu @ inv(Kuu))

        D = self.noise.expand(N)
        if self.approx == "FITC" or self.approx == "VFE":
            Kffdiag = self.kernel(self.X, diag=True)  # diagonal of Kff
            Qffdiag = W.pow(2).sum(dim=-1)  # 
            if self.approx == "FITC":
                D = D + Kffdiag - Qffdiag
            else:  # approx = "VFE"
                trace_term = (Kffdiag - Qffdiag).sum() / self.noise
                trace_term = trace_term.clamp(min=0)

        f_loc = self.mean_function(self.X)
        if self.y is None:
            f_var = D + W.pow(2).sum(dim=-1)
            return f_loc, f_var
        else:
            if self.approx == "VFE":
                pyro.factor(self._pyro_get_fullname("trace_term"), -trace_term / 2.0)
            print(f"f_loc shape: {f_loc.shape}")  # Should match y.shape
            print(f"W shape: {W.shape}")
            print(f"D shape: {D.shape}")
            print(f"y shape: {self.y.shape if self.y is not None else None}")
            
            return pyro.sample(
                self._pyro_get_fullname("y"),
                dist.LowRankMultivariateNormal(f_loc, W, D)
                # .expand_by(self.y.shape[:-1])
                .to_event(self.y.dim() - 1),
                obs=self.y,
            )

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()
        self.X_loc = PyroParam(
            torch.log(self.X_init).detach().clone().to(self.X.device),
            constraint=constraints.real
        )
        self.X_scale_tril = PyroParam(torch.eye(self.n_spots, device=self.X.device), 
                                    constraint=constraints.lower_cholesky)
        self.X = pyro.nn.PyroSample(
            lambda self: dist.TransformedDistribution(
                base_distribution=dist.MultivariateNormal(
                    loc=self.X_loc, 
                    scale_tril=self.X_scale_tril
                ),
                transforms=[dist.transforms.ExpTransform()]  # Transform back to original space
            ).expand([self.n_genes]).to_event(1)
        )

    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`), the inducing-point
            parameter ``Xu``, together with kernel's parameters have been learned from
            a training procedure (MCMC or SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        # W = inv(Luu) @ Kuf
        # Ws = inv(Luu) @ Kus
        # D as in self.model()
        # K = I + W @ inv(D) @ W.T = L @ L.T
        # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
        #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
        #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
        #   = inv(Luu).T @ inv(K) @ inv(Luu)
        #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
        # loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
        # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
        #     = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

        N = self.X.size(0)
        M = self.Xu.size(0)
        # TODO: cache these calculations to get faster inference
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[:: M + 1] += self.jitter  # add jitter to the diagonal
        Luu = torch.linalg.cholesky(Kuu)
        Kuf = self.kernel(self.Xu, self.X)
        W = torch.linalg.solve_triangular(Luu, Kuf, upper=False)
        D = self.noise.expand(N)
        if self.approx == "FITC":
            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=0)
            D = D + Kffdiag - Qffdiag

        W_Dinv = W / D
        K = W_Dinv.matmul(W.t()).contiguous()
        K.view(-1)[:: M + 1] += 1  # add identity matrix to K
        L = torch.linalg.cholesky(K)

        # get y_residual and convert it into 2D tensor for packing
        y_residual = self.y - self.mean_function(self.X)
        y_2D = y_residual.reshape(-1, N).t()
        W_Dinv_y = W_Dinv.matmul(y_2D)

        # End caching ----------

        Kus = self.kernel(self.Xu, Xnew)
        Ws = torch.linalg.solve_triangular(Luu, Kus, upper=False)
        pack = torch.cat((W_Dinv_y, Ws), dim=1)
        Linv_pack = torch.linalg.solve_triangular(L, pack, upper=False)
        # unpack
        Linv_W_Dinv_y = Linv_pack[:, : W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1] :]

        C = Xnew.size(0)
        loc_shape = self.y.shape[:-1] + (C,)
        loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)

        if full_cov:
            Kss = self.kernel(Xnew).contiguous()
            if not noiseless:
                Kss.view(-1)[:: C + 1] += self.noise  # add noise to the diagonal
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
            cov_shape = self.y.shape[:-1] + (C, C)
            cov = cov.expand(cov_shape)
        else:
            Kssdiag = self.kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + self.noise
            Qssdiag = Ws.pow(2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)
            cov_shape = self.y.shape[:-1] + (C,)
            cov = cov.expand(cov_shape)

        return loc + self.mean_function(Xnew), cov

    def train(self, train_steps=1000):
        from pyro.infer import SVI, TraceMeanField_ELBO
        from pyro.optim import ClippedAdam
        def train_model(model, num_epochs=1000, lr=0.01, patience=10, min_delta=1.0):
            # Setup optimizer and ELBO
            optimizer = ClippedAdam({"lr": lr})
            elbo = TraceMeanField_ELBO()
            svi = SVI(model.model, model.guide, optimizer, loss=elbo)
            losses = []
            # Early stopping monitor
            early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode='min')
            for epoch in range(num_epochs):
                loss = svi.step()
                losses.append(loss)
                print(f"Epoch {epoch}, ELBO loss: {loss:.4f}")
                if early_stopper(loss):
                    print("Early stopping triggered.")
                    break
            return model, losses
        trained_model, losses = train_model(self, num_epochs=train_steps, lr=0.01, patience=10, min_delta=0.5)
        # return the model 
        return trained_model, losses

    def infer(self, Xnew, full_cov=False, noiseless=True):
        """
        Run inference on new data using the trained SparseGPRegression model.

        Parameters
        ----------
        Xnew : torch.Tensor
            New input data for prediction (shape: [n_test, input_dim]).
        full_cov : bool, optional
            Whether to return full covariance matrix. Default is False.
        noiseless : bool, optional
            Whether to exclude noise in the prediction. Default is True.

        Returns
        -------
        loc : torch.Tensor
            Posterior mean of the predictions.
        cov : torch.Tensor
            Posterior variance or covariance matrix of the predictions.
        """
        self.eval()
        with torch.no_grad():
            loc, cov = self.forward(Xnew, full_cov=full_cov, noiseless=noiseless)
        return loc, cov


    def save(self, filename_prefix):
        """
        Save the model parameters and metadata using Pyro's param store.

        Parameters
        ----------
        filename_prefix : str
            Prefix for the saved files (e.g., 'model' will create 'model.pt' and 'model_meta.pt')
        """
        # Save Pyro parameters
        pyro.get_param_store().save(f"{filename_prefix}.pt")

        # Save metadata separately
        metadata = {
            'coords': self.coords.detach().cpu(),
            'in_tiss_mask': self.in_tiss_mask.detach().cpu(),
            'ttl_cnts': self.ttl_cnts.detach().cpu(),
            'approx': self.approx,
            'jitter': self.jitter,
            'Xu': self.Xu.detach().cpu(),
            'X': self.X.detach().cpu(),
            'y': self.y.detach().cpu() if self.y is not None else None,
            'kernel_state': {
                'name': self.kernel.__class__.__name__,
                'input_dim': self.kernel.input_dim,
                'params': {k: v.detach().cpu() for k, v in self.kernel.named_parameters()}
            }
        }
        torch.save(metadata, f"{filename_prefix}_meta.pt")
        print(f"Model saved to {filename_prefix}.pt and metadata to {filename_prefix}_meta.pt")


    @staticmethod
    def load(filename_prefix, custom_mean_func=None):
        """
        Load the model parameters and metadata using Pyro's param store.

        Parameters
        ----------
        filename_prefix : str
            Prefix for the saved files (e.g., 'model' will load 'model.pt' and 'model_meta.pt')
        custom_mean_func : callable, optional
            Custom mean function to override the default

        Returns
        -------
        model : SparseGPRegression
            The loaded model
        """
        # Load Pyro parameters
        params = torch.load(f"{filename_prefix}/model.pt", weights_only=False)
        pyro.get_param_store().set_state(params)
        
        # Load metadata
        metadata = torch.load(f"{filename_prefix}/meta.pt")

        # Reconstruct kernel
        kernel_name = metadata['kernel_state']['name']
        input_dim = metadata['kernel_state']['input_dim']
        params = metadata['kernel_state']['params']

        if kernel_name == 'RBF':
            kernel = gp.kernels.RBF(input_dim=input_dim)
        elif kernel_name == 'Matern32':
            kernel = gp.kernels.Matern32(input_dim=input_dim)
        elif kernel_name == 'Matern52':
            kernel = gp.kernels.Matern52(input_dim=input_dim)
        elif kernel_name == 'Periodic':
            kernel = gp.kernels.Periodic(input_dim=input_dim)
        elif kernel_name == 'Linear':
            kernel = gp.kernels.Linear(input_dim=input_dim)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_name}")

        for name, param in params.items():
            getattr(kernel, name).data = param

        # Reconstruct model
        model = SparseGPRegression(
            X=metadata['X'],
            y=metadata['y'],
            kernel=kernel,
            coords=metadata['coords'],
            in_tiss_mask=metadata['in_tiss_mask'],
            ttl_cnts=metadata['ttl_cnts'],
            noise=pyro.param("noise") if "noise" in pyro.get_param_store().keys() else None,
            approx=metadata['approx'],
            jitter=metadata['jitter']
        )

        print(f"Model loaded from {filename_prefix}/model.pt and metadata from {filename_prefix}/meta.pt")
        return model

    def infer_latent_X_quantiles(self, quantiles=[0.05, 0.5, 0.95], n_samples=1000):
        """
        Sample from the posterior of X and compute quantiles.

        Parameters
        ----------
        quantiles : list of float
            Quantiles to compute (e.g., [0.05, 0.5, 0.95]).
        n_samples : int
            Number of posterior samples to draw.

        Returns
        -------
        torch.Tensor
            Quantile estimates of X with shape [len(quantiles), n_genes, n_spots].
        """
        self.set_mode("guide")
        self._load_pyro_samples()

        loc = pyro.param("X_loc")
        scale_tril = pyro.param("X_shared_scale_tril")
        dist_X = dist.MultivariateNormal(loc, scale_tril=scale_tril)

        samples = dist_X.sample((n_samples,))
        return torch.quantile(samples, torch.tensor(quantiles), dim=0)
    

