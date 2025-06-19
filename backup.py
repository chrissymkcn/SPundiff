import torch.nn as nn
from typing import Optional, Tuple

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
from pyro.contrib import gp
from pyro.nn.module import PyroParam, pyro_method
import pyro.ops.stats as stats

from pde import CartesianGrid, ScalarField
from pde import DiffusionPDE
from pde.tools.numba import jit
from numba.extending import register_jitable
import numpy as np
from scipy.sparse import issparse


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
        domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes = calculate_domain_parameters(coords, divideby=1)
        out_tiss_mask = (in_tiss_mask==0).int()
        #### Custom noise spatially dependent
        out_tissue_mask = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=out_tiss_mask,
            coords=coords
        )[0]
        out_tissue_mask = out_tissue_mask.detach().numpy() if out_tissue_mask.requires_grad else out_tissue_mask.numpy()
        ttl_cnts = torch.tensor(ttl_cnts, dtype=torch.float32) if not isinstance(ttl_cnts, torch.Tensor) else ttl_cnts
        ttl_grid = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=ttl_cnts,
            coords=coords
        )[0]
        diff_mask = ttl_grid / ttl_grid.max()  # normalize to [0, 1]
        diff_mask = diff_mask.detach().numpy() if diff_mask.requires_grad else diff_mask.numpy()

        def process(i):
            x = X[i, :]
            y = ref_count[:, i]
            model = forward_diffusion(
                grid_sizes=grid_sizes, voxel_sizes=voxel_sizes, padding_sizes=padding_sizes,
                x=x, y=y, coords=coords, 
                out_tissue_mask=out_tissue_mask, diff_mask=diff_mask,
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
    def __init__(self, 
                grid_sizes: tuple, 
                voxel_sizes: tuple, 
                padding_sizes: tuple,
                x: torch.Tensor,  # Shape: (n_coords,)
                y: torch.Tensor,  # Shape: (n_coords,)
                coords: torch.Tensor,  # Shape: (n_coords, 2) 
                out_tissue_mask: np.ndarray,
                diff_mask: np.ndarray,  
                diffusivity=0.2, noise=0.01, 
                ):
        '''
        Simulate 2d spatial diffusion from initial condition x to reference condition y. 
        
        Args:
            grid_sizes: a tuple or list of two integers defining the grid size (n_x, n_y)
            voxel_sizes: a tuple or list of two floats defining the voxel size (dx, dy)
            padding_sizes: a tuple or list of two floats defining the padding size (pad_x, pad_y) at the scale of original coordinate space (not grids)
            x: a 1D tensor of shape (n_coords,) representing the initial condition at coordinates
            y: a 1D tensor of shape (n_coords,) representing the reference condition at coordinates (for restraining the diffusion)
            out_tissue_mask: a 1D tensor of shape (n_coords,) representing the mask for out-tissue regions (0 for out-of-tissue)
            coords: a 2D tensor of shape (n_coords, 2) representing the coordinates of the points
            diff_mask: a 2D tensor or list of shape (n_coords,) representing the total counts at each coordinate
            diffusivity: a float representing the diffusivity coefficient (default: 0.2)
            noise: a float representing the noise amplitude (default: 0.01)
        '''
        
        self.grid_sizes = grid_sizes
        self.voxel_sizes = voxel_sizes
        self.padding_sizes = padding_sizes
        self.x = x
        self.y = y
        self.coords = coords
        self.diffusivity = diffusivity
        self.noise = noise
        self.out_tissue_mask = out_tissue_mask
        self.diff_mask = diff_mask

    @staticmethod
    def post_step_hook(state_data, t, threshold):
        if state_data.max() < threshold:
            raise StopIteration

    def run(self):
        grids = self.define_grids()
        x_grid = grids['x_grid']
        y_grid = grids['y_grid']
        grid_coords = grids['grid_coords']
        out_tissue_mask = self.out_tissue_mask
        diff_mask = self.diff_mask
        
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
        result = eq.solve(state, t_range=100, dt=0.1)
        
        y_end = result.data[grid_coords[:, 0], grid_coords[:, 1]]
        print('simulated y_end sum, original y sum:')
        print(y_end.sum(), self.y.sum())
        y_end = y_end / (y_end.sum() / self.y.sum())  # normalize to match original data
        return y_end.detach()

    def define_grids(self):
        grid_sizes = self.grid_sizes
        voxel_sizes = self.voxel_sizes
        padding_sizes = self.padding_sizes
        x = self.x
        coords = self.coords
        y = self.y
        
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

        return {
            'x_grid': x_grid.detach().numpy() if x_grid.requires_grad else x_grid.numpy(),
            'y_grid': y_grid.detach().numpy() if y_grid.requires_grad else y_grid.numpy(),
            'grid_coords': grid_coords.detach().numpy() if grid_coords.requires_grad else grid_coords.numpy(),
        }

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        improvement = (current_score < self.best_score - self.min_delta) if self.mode == 'min' else (current_score > self.best_score + self.min_delta)

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
