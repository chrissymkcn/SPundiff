import torch
from torch.distributions import constraints
from torch.nn import Parameter
from torch.nn.functional import pdist
import torch.nn.functional as F

from functools import partial
from joblib import Parallel, delayed

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.nn.module import PyroParam, pyro_method
import pyro.ops.stats as stats

from pde import CartesianGrid, ScalarField
from pde import DiffusionPDE
import numpy as np
from pde.tools.numba import jit
from numba.extending import register_jitable


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

class SparseGPRegression(GPModel):
    """
    Sparse Gaussian Process Regression model.

    In :class:`.GPRegression` model, when the number of input data :math:`X` is large,
    the covariance matrix :math:`k(X, X)` will require a lot of computational steps to
    compute its inverse (for log likelihood and for prediction). By introducing an
    additional inducing-input parameter :math:`X_u`, we can reduce computational cost
    by approximate :math:`k(X, X)` by a low-rank Nystr\u00f6m approximation :math:`Q`
    (see reference [1]), where

    .. math:: Q = k(X, X_u) k(X_u,X_u)^{-1} k(X_u, X).

    Given inputs :math:`X`, their noisy observations :math:`y`, and the inducing-input
    parameters :math:`X_u`, the model takes the form:

    .. math::
        u & \\sim \\mathcal{GP}(0, k(X_u, X_u)),\\\\
        f & \\sim q(f \\mid X, X_u) = \\mathbb{E}_{p(u)}q(f\\mid X, X_u, u),\\\\
        y & \\sim f + \\epsilon,

    where :math:`\\epsilon` is Gaussian noise and the conditional distribution
    :math:`q(f\\mid X, X_u, u)` is an approximation of

    .. math:: p(f\\mid X, X_u, u) = \\mathcal{N}(m, k(X, X) - Q),

    whose terms :math:`m` and :math:`k(X, X) - Q` is derived from the joint
    multivariate normal distribution:

    .. math:: [f, u] \\sim \\mathcal{GP}(0, k([X, X_u], [X, X_u])).

    This class implements three approximation methods:

    + Deterministic Training Conditional (DTC):

        .. math:: q(f\\mid X, X_u, u) = \\mathcal{N}(m, 0),

      which in turns will imply

        .. math:: f \\sim \\mathcal{N}(0, Q).

    + Fully Independent Training Conditional (FITC):

        .. math:: q(f\\mid X, X_u, u) = \\mathcal{N}(m, diag(k(X, X) - Q)),

      which in turns will correct the diagonal part of the approximation in DTC:

        .. math:: f \\sim \\mathcal{N}(0, Q + diag(k(X, X) - Q)).

    + Variational Free Energy (VFE), which is similar to DTC but has an additional
      `trace_term` in the model's log likelihood. This additional term makes "VFE"
      equivalent to the variational approach in :class:`.VariationalSparseGP`
      (see reference [2]).

    .. note:: This model has :math:`\\mathcal{O}(NM^2)` complexity for training,
        :math:`\\mathcal{O}(NM^2)` complexity for testing. Here, :math:`N` is the number
        of train inputs, :math:`M` is the number of inducing inputs.

    References:

    [1] `A Unifying View of Sparse Approximate Gaussian Process Regression`,
    Joaquin Qui\u00f1onero-Candela, Carl E. Rasmussen

    [2] `Variational learning of inducing variables in sparse Gaussian processes`,
    Michalis Titsias

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param str approx: One of approximation methods: "DTC", "FITC", and "VFE"
        (default).
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """

    def __init__(
        self, X, y, kernel, 
        coords, in_tiss_mask, ttl_cnts,
        noise=None, mean_function=None, approx=None, jitter=1e-6
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
            in_tiss_mask=in_tiss_mask, ttl_cnts=ttl_cnts) if mean_function is None else None
        super().__init__(X, y, kernel=kernel, mean_function=mean_function, jitter=jitter)
        
        # gene_variances = PyroParam(
        #     torch.ones(self.n_genes) * 0.1,
        #     constraint=constraints.positive
        # )
        # self.X = pyro.nn.PyroSample(
        #     dist.Normal(
        #         X,  # [n_genes, n_spots] mean  
        #         gene_variances.unsqueeze(1)  # [n_genes, 1] learnable variances
        #     ).to_event(1)
        # )
        
        X_centered = X - X.mean(dim=1, keepdim=True)
        X_normalized = X_centered / X_centered.std(dim=1, keepdim=True)
        
        # Calculate correlation matrix [n_genes, n_genes]
        correlation_matrix = torch.mm(X_normalized, X_normalized.t()) / self.n_spots
        self.X = pyro.nn.PyroSample(
            dist.MultivariateNormal(
                X,  # [n_genes, n_spots] mean
                covariance_matrix=correlation_matrix  # [n_genes, n_genes] fixed correlation
            ).to_event(1)
        )
        self.autoguide("X", dist.MultivariateNormal)
        
        Xu = stats.resample(Parameter(X), n_inducing)
        self.Xu = Parameter(Xu) if not isinstance(Xu, Parameter) else Xu

        noise = self.X.new_tensor(1.0) if noise is None else noise
        self.noise = PyroParam(noise, constraints.positive)

        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError(
                "The sparse approximation method should be one of "
                "'DTC', 'FITC', 'VFE'."
            )

    @staticmethod
    def mean_function(X, coords, ref_count, in_tiss_mask, ttl_cnts):
        domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes = calculate_domain_parameters(coords, divideby=1)

        def process(i):
            x = X[i, :]
            y = ref_count[:, i]
            model = forward_diffusion(
                grid_sizes=grid_sizes, voxel_sizes=voxel_sizes, padding_sizes=padding_sizes,
                x=x, y=y, coords=coords, in_tiss_mask=in_tiss_mask, ttl_cnts=ttl_cnts,
                diffusivity=0.2, noise=0.01,
            )
            return torch.tensor(model.run(), dtype=torch.float32)

        res = Parallel(n_jobs=-1)(delayed(process)(i) for i in range(X.shape[0]))
        return torch.stack(res, dim=0).T

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

        zero_loc = self.X.new_zeros(self.y.shape)
        f_loc = zero_loc + self.mean_function(self.X)
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

