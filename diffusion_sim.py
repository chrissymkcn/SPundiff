import torch.nn as nn
import numpy as np
import torch
from torch.distributions import constraints
from torch.nn.functional import pdist
import torch.nn.functional as F

import pyro
import pyro.distributions as dist


def calculate_domain_parameters(coords: torch.Tensor, divideby: float = 1.0):
    """
    Calculate appropriate domain parameters based on input coordinates.
    
    Args:
        coords: Tensor of shape (n, 2) containing x, y coordinates
        divideby: Factor to divide minimum distance by for voxel size
    
    Returns:
        domain_sizes: Physical size of domain in micrometers
        grid_sizes: Number of voxels in each dimension
        voxel_sizes: Size of each voxel in micrometers
        diffusion_const: Diffusion coefficient in μm²/s
        padding_sizes: Size of padding in each dimension
    """
    # Calculate the physical extent of the domain
    x_min, y_min = torch.min(coords, dim=0)[0]
    x_max, y_max = torch.max(coords, dim=0)[0]
    
    padding = 0.0 # 10% padding
    
    # Add padding to domain boundaries
    domain_width = (x_max - x_min) * (1 + 2*padding)
    domain_height = (y_max - y_min) * (1 + 2*padding)
    domain_sizes = torch.tensor([domain_width, domain_height], device=coords.device)
    padding_sizes = torch.tensor([
        (x_max - x_min) * padding,
        (y_max - y_min) * padding
    ], device=coords.device)
    
    # Calculate minimum distance between coords
    # Handle x coordinates
    x_coords = torch.unique(coords[:, 0:1], dim=0)
    x_dists = torch.unique(pdist(x_coords), return_counts=True)
    min_distance_x = x_dists[0][torch.argmax(x_dists[1])]
    
    # Handle y coordinates
    y_coords = torch.unique(coords[:, 1:2], dim=0)
    y_dists = torch.unique(pdist(y_coords), return_counts=True)
    min_distance_y = y_dists[0][torch.argmax(y_dists[1])]
    
    # Set voxel size to be 1/divideby of minimum distance between coords
    voxel_sizes = torch.tensor([
        min_distance_x / divideby,
        min_distance_y / divideby
    ], device=coords.device)
    
    # Calculate grid size (number of voxels)
    grid_sizes = domain_sizes / voxel_sizes + 1
    grid_sizes = torch.ceil(grid_sizes).to(torch.int64)
    
    # Set diffusion coefficient based on literature x for mRNA
    diffusion_const = 1.0  # μm²/s
    
    return domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes


def coords_to_grid(
    grid_sizes,
    dx, 
    dy,
    padding_sizes,
    x: torch.Tensor,  # Shape: (n_coords,)
    coords: torch.Tensor = None,  # Shape: (n_coords, 2)
) -> torch.Tensor:
    """
    Convert point observations to grid x with gap filling
    
    Args:
        coords: Point coordinates (n_coords, 2)
        x: Observed x at coords (n_coords,)
        
    Returns:
        Grid with interpolated x
    """
    # Create empty grid
    grid = torch.zeros(grid_sizes[0], grid_sizes[1], dtype=torch.float32)
    
    # Convert coords to nearest grid indices
    x_idx = torch.round((coords[:, 0] - coords[:, 0].min() + padding_sizes[0]) / dx).long()
    y_idx = torch.round((coords[:, 1] - coords[:, 1].min() + padding_sizes[1]) / dy).long()
    
    # # Clamp indices to valid range
    # x_idx = torch.clamp(x_idx, 0, grid_sizes[0] - 1)
    # y_idx = torch.clamp(y_idx, 0, grid_sizes[1] - 1)
    grid_coords = torch.stack((x_idx, y_idx), dim=1)
    
    # Assign x to nearest grid coords
    for i in range(len(coords)):
        grid[x_idx[i], y_idx[i]] = x[i]
    return grid, grid_coords


def fill_grid_gaps(grid: torch.Tensor, min_neighs=3) -> torch.Tensor:
    """
    Fill gaps in the grid by averaging neighboring non-zero x.
    Args:
        grid: Input grid with some zero x (shape: (H, W))
    Returns:
        grid: Grid with gaps filled
        grid_coords: Coordinates of the grid coords
    """
    # Create mask for original non-zero coords
    original_mask = grid != 0
    
    # Iteratively fill gaps
    max_iterations = 10  # Limit iterations to avoid infinite loops
    for _ in range(max_iterations):
        # Store previous grid for convergence check
        previous_grid = grid.clone()
        
        # Add batch and channel dimensions before padding
        grid_expanded = grid.unsqueeze(0).unsqueeze(0)
        padded_expanded = F.pad(grid_expanded, (1,1,1,1), mode='replicate')
        padded = padded_expanded.squeeze(0).squeeze(0)
        
        # Get neighboring x
        neighbors = torch.stack([
            padded[1:-1, :-2],  # left
            padded[1:-1, 2:],   # right
            padded[:-2, 1:-1],  # up
            padded[2:, 1:-1],   # down
        ])
        
        # Count non-zero neighbors
        non_zero_neighbors = (neighbors != 0).sum(dim=0)
        
        # Calculate mean of non-zero neighbors
        neighbor_sum = neighbors.sum(dim=0)
        neighbor_count = (neighbors != 0).sum(dim=0).clamp(min=1)  # Avoid division by zero
        neighbor_mean = neighbor_sum / neighbor_count
        
        # Create mask for coords to be filled
        fill_mask = (grid == 0) & (non_zero_neighbors >= min_neighs)
        
        # Fill gaps
        grid[fill_mask] = neighbor_mean[fill_mask]
        
        # Check for convergence
        if torch.allclose(grid, previous_grid):
            break
            
    # Restore original x
    grid[original_mask] = previous_grid[original_mask]    
    return grid


def coords_to_filled_grid(
    grid_size,
    dx, 
    dy,
    padding_sizes,
    x: torch.Tensor,  # Shape: (n_coords,)
    coords: torch.Tensor = None,  # Shape: (n_coords, 2)
) :
    """
    Convert point observations to grid x with gap filling
    
    Args:
        coords: Point coordinates (n_coords, 2)
        x: Observed x at coords (n_coords,)
        
    Returns:
        Grid with interpolated x
    """
    grid, grid_coords = coords_to_grid(grid_size, dx, dy, padding_sizes, x, coords)
    filled_grid = fill_grid_gaps(grid)
    
    return filled_grid, grid_coords


class PhysicsInformedDiffusionModel(nn.Module):
    """
    Physics-informed probabilistic model for spatial transcriptomics diffusion correction
    Combines diffusion physics with probabilistic modeling using Pyro
    """
    def __init__(
        self,
        n_spots: int,
        n_genes: int,
        coords: torch.Tensor,
        D_out: float = 0.1,
        D_in: float = 0.02,
        initial_counts_guess: torch.Tensor = None,
        total_counts: torch.Tensor = None,
        neighbors: torch.Tensor = None,  # Spatial neighborhood matrix
        in_tiss_mask: torch.Tensor = None,
        alpha: float = 0.8,  # Tissue mask weight
        beta: float = 0.2,  # Spatial regularization weight
    ):
        super().__init__()
        self.n_spots = n_spots
        self.n_genes = n_genes
        self.coords = coords
        # self.neighbor_graph = neighbor_graph
        self.alpha = alpha
        self.beta = beta
        self.total_counts = total_counts if isinstance(total_counts, torch.Tensor) else torch.full((self.n_spots, 1), 10.0)
        self.total_counts = self.total_counts.unsqueeze(-1) if self.total_counts.dim() == 1 else self.total_counts  # Ensure shape is [n_spots, 1]
        self.in_tiss_mask = in_tiss_mask if in_tiss_mask is not None else torch.ones(n_spots, dtype=torch.bool, device=coords.device)
        self.D = D_out * (1 - in_tiss_mask) + D_in * in_tiss_mask
        self.D = pyro.param(
            "D",
            self.D,
            constraint=constraints.real
        )

        if initial_counts_guess is not None:
            self.initial_counts_guess = initial_counts_guess + 1e-6
        else:
            self.initial_counts_guess = torch.ones((n_spots, n_genes))
                
        # Add RBF-FD specific initialization
        self.neighbors = neighbors

    def compute_laplacian(self):
        self.neighbors.fill_diagonal_(0)
        degree = torch.diag(self.neighbors.sum(dim=1))
        laplacian = degree - self.neighbors
        alpha_diag = torch.diag(self.alpha_vector)
        scaled_laplacian = alpha_diag @ laplacian
        self.laplacian = scaled_laplacian

    def forward_diffusion(self, heat_init, steps=10):
        heat = heat_init.clone()
        for _ in range(steps):
            laplacian_update = self.laplacian @ heat
            heat = heat - laplacian_update
            heat = torch.clamp(heat, min=0.0)
        return heat

    def tissue_boundary_constraint(self, diffused_counts, observed_counts):
        """
        More sophisticated tissue boundary constraint that considers
        distance-dependent diffusion patterns
        """
        in_tiss_mask = self.in_tiss_mask.float().unsqueeze(1)
        
        # Compute distance to tissue boundary for each spot
        boundary_distances = self.compute_boundary_distances(self.in_tiss_mask)  # [n_spots]
        
        # Expected decay pattern from observed data
        observed_decay = self.compute_decay_pattern(
            observed_counts, 
            boundary_distances
        )  # [n_bins, n_genes]
        
        # Actual decay pattern from diffused counts
        diffused_decay = self.compute_decay_pattern(
            diffused_counts, 
            boundary_distances
        )  # [n_bins, n_genes]
        
        # Compare decay patterns
        decay_deviation = torch.nn.functional.mse_loss(
            diffused_decay,
            observed_decay
        )
        
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
    
    def debug_tensor(self, name, tensor, print_stats=True):
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        if print_stats:
            print(f"\n{name} stats:")
            print(f"Shape: {tensor.shape}")
            print(f"Range: [{tensor.min().item():.6e}, {tensor.max().item():.6e}]")
            print(f"Mean: {tensor.mean().item():.6e}")
            print(f"Has NaN: {has_nan}")
            print(f"Has Inf: {has_inf}")
        if has_nan or has_inf:
            print(f"\nFound NaN/Inf in {name}!")
            if has_nan:
                nan_indices = torch.where(torch.isnan(tensor))
                print(f"First few NaN positions: {[(i.item(), j.item()) for i, j in zip(*nan_indices[:2])][:5]}")
            if has_inf:
                inf_indices = torch.where(torch.isinf(tensor))
                print(f"First few Inf positions: {[(i.item(), j.item()) for i, j in zip(*inf_indices[:2])][:5]}")
            return True
        return False

    def test(
        self,
        observed_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Probabilistic model with debugging"""

        # 1. Check input tensors
        print("\nChecking inputs:")
        self.debug_tensor("observed_counts", observed_counts)
        self.debug_tensor("in_tiss_mask", self.in_tiss_mask)
        self.debug_tensor("neighbors", self.neighbors)

        # 2. Check parameters
        original_counts_loc = pyro.param(
            "original_counts_loc",
            torch.ones_like(observed_counts),
            constraint=constraints.nonnegative
        )
        original_counts_scale = pyro.param(
            "original_counts_scale",
            0.1 * torch.ones_like(observed_counts),
            constraint=constraints.nonnegative
        )
        
        print("\nChecking parameters:")
        self.debug_tensor("original_counts_loc", original_counts_loc)
        self.debug_tensor("original_counts_scale", original_counts_scale)

        # 3. Sample original counts
        original_counts = pyro.sample(
            "original_counts",
            dist.LogNormal(original_counts_loc, original_counts_scale).to_event(2)
        )
        
        self.debug_tensor("original_counts (after sampling)", original_counts)

        # 4. Check diffusion process
        print("\nChecking diffusion process:")
        # Debug Laplacian computation
        L = self.compute_laplacian(original_counts)
        self.debug_tensor("Laplacian", L.to_dense() if L.is_sparse else L)

        # Forward diffusion with intermediate checks
        diffused_counts = original_counts.clone()
        for step in range(10):  # Assuming steps=1
            print(f"\nDiffusion step {step + 1}:")
            try:
                old_diffused = diffused_counts.clone()
                diffused_counts = self.forward_diffusion(old_diffused, steps=10)
                
                if self.debug_tensor(f"diffused_counts (step {step + 1})", diffused_counts):
                    print("Intermediate values in forward_diffusion:")
                    self.debug_tensor("old_diffused", old_diffused)
                    # Add more specific checks based on your forward_diffusion implementation
            except Exception as e:
                print(f"Error in diffusion step {step + 1}: {str(e)}")
                raise

        # 5. Check total_counts and probs computation
        print("\nChecking likelihood computation:")
        total_counts = self.total_counts

        # Compute probs with additional safety
        probs = diffused_counts / total_counts
        self.debug_tensor("probs (before clamping)", probs)
        
        # 6. Sample observations
        with pyro.plate("spots", self.n_spots):
            obs = pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=total_counts,
                    probs=probs
                ).to_event(1),
                obs=observed_counts
            )
        self.debug_tensor("obs", obs)

        return {
            "original_counts": original_counts,
            "diffused_counts": diffused_counts,
            "probs": probs,
            'Laplacian': L.to_dense() if L.is_sparse else L,
            "obs": obs
        }
    
    def model(
        self,
        observed_counts: torch.Tensor,  # Observed gene expression matrix [n_spots, n_genes]
    ) -> torch.Tensor:
        """
        Probabilistic model incorporating physics and data
        
        Args:
            observed_counts: Observed gene expression matrix [n_spots, n_genes]
            in_tiss_mask: Binary mask for tissue locations [n_spots]
            neighbors: Spatial neighborhood matrix [n_spots, n_spots]
        """
        # Prior for original counts
        original_counts_loc = pyro.param(
            "original_counts_loc",
            self.initial_counts_guess,
            constraint=constraints.nonnegative
        )
        original_counts_scale = pyro.param(
            "original_counts_scale",
            0.1 * torch.ones_like(observed_counts),
            constraint=torch.distributions.constraints.nonnegative
        )  # [n_spots, n_genes]
        
        # Sample original counts
        original_counts = pyro.sample(
            "original_counts",
            dist.LogNormal(original_counts_loc, original_counts_scale).to_event(2)
        )  # [n_spots, n_genes]
        
        # Physics-based diffusion
        diffused_counts = self.forward_diffusion(original_counts, steps=1)  # [n_spots, n_genes]
        self.debug_tensor("diffused_counts", diffused_counts, print_stats=True)
        
        # Likelihood incorporating:
        # 1. Observation model
        # 2. Tissue mask constraint
        # 3. Spatial regularization
        
        # 1. Observation likelihood
        total_counts = self.total_counts
        probs = diffused_counts / total_counts
        
        # Use Negative Binomial observation model
        with pyro.plate("spots", self.n_spots):
            pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=total_counts,  # shape: [n_spots, 1]
                    probs=probs  # shape: [n_spots, n_genes]
                ).to_event(1),  # event_shape = [n_genes]
                obs=observed_counts
            )
        
        pyro.factor(
            'tissue_constraint', self.tissue_boundary_constraint(
                self, diffused_counts, self.in_tiss_mask, observed_counts)
        )
        
        # 3. Spatial regularization using neighbor structure
        spatial_diff = torch.sum(
            self.neighbors.unsqueeze(2) * 
            (diffused_counts.unsqueeze(1) - diffused_counts.unsqueeze(0))**2
        )
        spatial_factor = -self.beta * spatial_diff
        pyro.factor("spatial_regularization", spatial_factor)
        
        return original_counts

    def guide(
        self,
        observed_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Variational guide/posterior"""
        # Variational parameters
        original_counts_loc = pyro.param(
            "original_counts_loc",
            self.initial_counts_guess,
            constraint=torch.distributions.constraints.nonnegative
        )
        original_counts_scale = pyro.param(
            "original_counts_scale",
            0.1 * torch.ones_like(observed_counts),
            constraint=torch.distributions.constraints.nonnegative
        )
        
        # Variational distribution
        return pyro.sample(
            "original_counts",
            dist.LogNormal(original_counts_loc, original_counts_scale).to_event(2)
        )




    # def setup_rbf_weights(self):
    #     """Setup RBF-FD weights for spatial discretization"""
    #     # Compute pairwise distances
    #     dists = torch.cdist(self.coords, self.coords)
        
    #     # Find nearest neighbors for each point
    #     values, indices = torch.topk(dists, k=self.neighbors, dim=1, largest=False)
        
    #     # Compute RBF shape parameter (epsilon)
    #     epsilon = torch.median(values[:, 1])  # Use median distance to nearest neighbor
        
    #     # Initialize weights matrix
    #     lap_weights = torch.zeros_like(dists)
        
    #     for i in range(self.n_spots):
    #         # Get local coordinates
    #         local_points = self.coords[indices[i]]  # [neighbors, 2]
    #         center = self.coords[i]  # [2]
            
    #         # Compute RBF matrix
    #         A = self._rbf_matrix(local_points, epsilon)  # [neighbors, neighbors]
            
    #         # Compute Laplacian values
    #         b = self._rbf_laplacian_values(local_points, center, epsilon)  # [neighbors]
            
    #         try:
    #             # Solve for weights
    #             weights = torch.linalg.solve(A, b)
    #         except:
    #             # Fallback to pseudo-inverse if system is ill-conditioned
    #             weights = torch.linalg.pinv(A) @ b
            
    #         # Store weights
    #         lap_weights[i, indices[i]] = weights
        
    #     # Register buffer (persistent state)
    #     self.register_buffer('lap_weights', lap_weights)
    #     self.register_buffer('neighbor_indices', indices)

    # @staticmethod
    # def _rbf_matrix(points, epsilon):
    #     """Compute RBF interpolation matrix"""
    #     dists = torch.cdist(points, points)
    #     return torch.exp(-dists**2 / (2 * epsilon**2))
    
    # @staticmethod
    # def _rbf_laplacian_values(points, center, epsilon):
    #     """Compute RBF Laplacian values"""
    #     dists = torch.cdist(points.unsqueeze(0), center.unsqueeze(0)).squeeze()
    #     r2 = dists**2
    #     eps2 = epsilon**2
    #     # For 2D Laplacian of Gaussian RBF
    #     return torch.exp(-r2 / (2 * eps2)) * (r2 / eps2 - 2)
    
    # def compute_laplacian(self, X: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute Laplacian using RBF-FD weights with spot-specific diffusion coefficients
    #     """
    #     # Get diffusion coefficients
    #     D_values = self.D() if callable(self.D) else self.D  # [n_spots]
        
    #     # Create diffusion coefficient matrix (average D between spots)
    #     D_matrix = (D_values.unsqueeze(1) + D_values.unsqueeze(0)) / 2  # [n_spots, n_spots]
        
    #     # Scale weights by diffusion coefficients
    #     scaled_weights = self.lap_weights * D_matrix  #### not defined yet
        
    #     # Convert to sparse tensor for efficiency
    #     indices = torch.nonzero(scaled_weights).t()
    #     values = scaled_weights[indices[0], indices[1]]
    #     sparse_L = torch.sparse_coo_tensor(
    #         indices, values, scaled_weights.shape,
    #         device=X.device
    #     )
        
    #     # Return Laplacian operator
    #     return sparse_L
    
    # def forward_diffusion(self, X: torch.Tensor, steps: int = 1) -> torch.Tensor:
    #     """
    #     Solve diffusion equation using implicit scheme with RBF-FD discretization
    #     """
    #     X_t = X.clone()
    #     dt = self.dt
        
    #     # Pre-compute sparse matrices
    #     I = torch.eye(self.n_spots, device=X.device)
    #     L = self.compute_laplacian(X_t)
    #     self.debug_tensor("Laplacian", L, print_stats=False)
    #     # Add regularization
    #     eps = 1e-6
    #     A = I - dt * L.to_dense() + eps * I
        
    #     try:
    #         # LU factorization
    #         LU, pivots = torch.linalg.lu_factor(A)
            
    #         for _ in range(steps):
    #             try:
    #                 # Store original sum
    #                 original_sum = X_t.sum(dim=0, keepdim=True)
                    
    #                 # Solve system
    #                 X_t_new = torch.linalg.lu_solve(LU, pivots, X_t)
                    
    #                 # Ensure non-negativity
    #                 X_t_new = torch.log1p(torch.exp(X_t_new))
                    
    #                 # Mass conservation
    #                 new_sum = X_t_new.sum(dim=0, keepdim=True)
    #                 X_t = X_t_new * (original_sum / (new_sum + eps))
                    
    #                 # Numerical stability
    #                 X_t = X_t + eps
                    
    #             except RuntimeError as e:
    #                 print(f"Solver failed: {e}")
    #                 # Fallback to explicit scheme
    #                 X_t = X_t - dt * torch.sparse.mm(L, X_t)
    #                 X_t = torch.log1p(torch.exp(X_t))
    #                 X_t = X_t * (original_sum / (X_t.sum(dim=0, keepdim=True) + eps))
    #                 X_t = X_t + eps
                    
    #     except RuntimeError as e:
    #         print(f"LU factorization failed: {e}")
    #         # Direct sparse solve
    #         for _ in range(steps):
    #             original_sum = X_t.sum(dim=0, keepdim=True)
    #             X_t = torch.linalg.solve(A, X_t)
    #             X_t = torch.log1p(torch.exp(X_t))
    #             X_t = X_t * (original_sum / (X_t.sum(dim=0, keepdim=True) + eps))
    #             X_t = X_t + eps
        
    #     return X_t
    
