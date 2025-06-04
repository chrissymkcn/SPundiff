from functools import partial
from pde import CartesianGrid, ScalarField
import torch
from torch.nn.functional import pdist
import torch.nn.functional as F


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
) -> torch.Tensor:
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



import numpy as np

from pde import DiffusionPDE, ScalarField, UnitGrid
from pde.tools.numba import jit
from numba.extending import register_jitable


class SpatialDiffusionPDE(DiffusionPDE):
    """Diffusion PDE with custom diffusivity and noise implementations."""

    def __init__(self, diffusivity=0.1, noise=0.1, 
                out_tissue_mask=None, diff_mask=None, post_step_hook=None):
        """
        Parameters:
        -----------
        base_diffusivity : float
            Base diffusion coefficient
        noise : float
            Noise amplitude
        out_tissue_mask : ndarray
            Mask for out-of-tissue regions for noise scaling
        diff_mask : ndarray
            Spatial mask to modify diffusivity at different locations
        """
        super().__init__(diffusivity=diffusivity, noise=noise)
        self.out_tissue_mask = out_tissue_mask
        self.diff_mask = diff_mask if diff_mask is not None else 1.0
        self.post_step_hook = post_step_hook

    # def evolution_rate(self, state, t=0):
    #     """Numpy implementation of the evolution equation"""
    #     # Calculate the Laplacian term
    #     state_lap = state.laplace(bc="auto_periodic_neumann")
    #     # Calculate the gradient terms
    #     state_grad = state.gradient(bc="auto_periodic_neumann")
    #     diff_grad = ScalarField(state.grid, self.diff_mask).gradient(bc="auto_periodic_neumann")
        
    #     # Combine terms: D∇²c + ∇D·∇c
    #     result = (self.diff_mask * state_lap + 
    #              sum(g1 * g2 for g1, g2 in zip(diff_grad, state_grad)))
    #     return self.diffusivity * result

    # def _make_pde_rhs_numba(self, state):
    #     """Numba implementation of the PDE"""
    #     base_diff = float(self.diffusivity)
    #     diff_mask = self.diff_mask.copy()  # Make a copy for numba
        
    #     # Get operators
    #     laplace = state.grid.make_operator("laplace", bc="auto_periodic_neumann")
    #     gradient = state.grid.make_operator("gradient", bc="auto_periodic_neumann")
        
    #     @jit
    #     def pde_rhs(state_data, t):
    #         # Calculate Laplacian term: D∇²c
    #         lap_term = diff_mask * laplace(state_data)
            
    #         # Calculate gradient terms: ∇D·∇c
    #         state_grad = gradient(state_data)
    #         diff_grad = gradient(diff_mask)
            
    #         # Sum up the dot product of gradients
    #         grad_term = np.zeros_like(state_data)
    #         for i in range(len(state_grad)):
    #             grad_term += diff_grad[i] * state_grad[i]
            
    #         return base_diff * (lap_term + grad_term)
            
    #     return pde_rhs

    def _make_noise_realization_numba(self, state):
        """Numba implementation of spatially-dependent noise."""
        # Cache variables for numba
        noise_amplitude = float(self.noise)
        out_tiss_mask = self.out_tissue_mask.copy() if self.out_tissue_mask is not None else 1.0

        @jit
        def noise_realization(state_data, t):
            mask = out_tiss_mask + 0.1
            noise_field = np.random.uniform(0, noise_amplitude, size=state_data.shape)
            return noise_field * mask

        return noise_realization
    
    def make_post_step_hook(self, state):
        """Returns a function that is called after each step.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            tuple: The first entry is the function that implements the hook. The second
                entry gives the initial data that is used as auxiliary data in the hook.
                This can be `None` if no data is used.

        Raises:
            NotImplementedError: When :attr:`post_step_hook` is `None`.
        """
        if self.post_step_hook is None:
            raise NotImplementedError("`post_step_hook` not set")
        else:
            post_step_hook = register_jitable(self.post_step_hook)

            @register_jitable
            def post_step_hook_impl(state_data, t, post_step_data):
                post_step_hook(state_data, t)

            return post_step_hook_impl, 0  # hook function and initial value

class forward_diffusion():
    def __init__(self, grid_sizes, voxel_sizes, padding_sizes,
                x, y, coords, in_tiss_mask, ttl_cnts,
                diffusivity=0.2, noise=0.01, 
                ):
        self.grid_sizes = grid_sizes
        self.voxel_sizes = voxel_sizes
        self.padding_sizes = padding_sizes
        self.x = x
        self.y = y
        self.coords = coords
        self.diffusivity = diffusivity
        self.noise = noise
        self.in_tiss_mask = in_tiss_mask
        self.ttl_cnts = ttl_cnts

    @staticmethod
    def post_step_hook(state_data, t, threshold):
        if state_data.max() < threshold:
            raise StopIteration

    def run(self):
        grids = self.define_grids()
        x_grid = grids['x_grid']
        y_grid = grids['y_grid']
        grid_coords = grids['grid_coords']
        out_tissue_mask = grids['out_tissue_mask']
        diff_mask = grids['diff_mask']
        
        # diff params 
        diffusivity = self.diffusivity
        noise = self.noise
        n_x, n_y = self.grid_sizes.tolist()
        grid = CartesianGrid([[0, n_x], [0, n_y]], [n_x, n_y])
        state = ScalarField(grid, x_grid)  # generate initial condition
        post_step_hook = partial(forward_diffusion.post_step_hook, threshold=self.y.max())  # Define a post-step hook to stop simulation if max value exceeds threshold

        eq = SpatialDiffusionPDE(diffusivity=diffusivity, noise=noise, 
                                    out_tissue_mask=out_tissue_mask, 
                                    diff_mask=diff_mask, 
                                    post_step_hook=post_step_hook)
        result = eq.solve(state, t_range=30, dt=0.1)
        
        y_end = result.data[grid_coords[:, 0], grid_coords[:, 1]]
        print(y_end.sum(), self.y.sum())
        y_end = y_end / (y_end.sum() / self.y.sum())  # normalize to match original data
        return y_end

    def define_grids(self):
        grid_sizes = self.grid_sizes
        voxel_sizes = self.voxel_sizes
        padding_sizes = self.padding_sizes
        x = self.x
        coords = self.coords
        y = self.y
        in_tiss_mask = self.in_tiss_mask
        out_tiss_mask = (in_tiss_mask==0).int()
        ttl_cnts = self.ttl_cnts
        
        x_grid, grid_coords = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=x,
            coords=coords
        )

        y_grid = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=y,
            coords=coords
        )[0]

        #### Custom noise spatially dependent
        out_tissue_mask = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=out_tiss_mask,
            coords=coords
        )[0]
        ttl_cnts = torch.tensor(ttl_cnts, dtype=torch.float32)
        ttl_grid = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=ttl_cnts,
            coords=coords
        )[0]
        diff_mask = ttl_grid / ttl_grid.max()  # normalize to [0, 1]
        return {
            'x_grid': x_grid.detach().numpy() if x_grid.requires_grad else x_grid.numpy(),
            'y_grid': y_grid.detach().numpy() if y_grid.requires_grad else y_grid.numpy(),
            'grid_coords': grid_coords.detach().numpy() if grid_coords.requires_grad else grid_coords.numpy(),
            'out_tissue_mask': out_tissue_mask.detach().numpy() if out_tissue_mask.requires_grad else out_tissue_mask.numpy(),
            'diff_mask': diff_mask.detach().numpy() if diff_mask.requires_grad else diff_mask.numpy()
        }
