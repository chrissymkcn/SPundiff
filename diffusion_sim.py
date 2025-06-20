from sklearn.cluster import KMeans

import torch.nn as nn
import torch
from torch.distributions import constraints
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib import gp
from pyro.nn.module import PyroParam, pyro_method
import pyro.ops.stats as stats


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
        total_counts: torch.Tensor,
        neighbors: torch.Tensor,  # Spatial neighborhood matrix
        D_out: float = 0.1,
        D_in: float = 0.02,
        initial_counts_guess: torch.Tensor = None,
        observed_counts: torch.Tensor = None,  # Observed counts [n_spots, n_genes]
        in_tiss_mask: torch.Tensor = None,
        diffusion_steps: int = 10,
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
        self.total_counts = total_counts.unsqueeze(-1) if total_counts.dim() == 1 else total_counts  # Ensure shape is [n_spots, 1]
        # self.total_counts = torch.log1p(self.total_counts)  # Log-transform to stabilize training
        self.in_tiss_mask = in_tiss_mask if in_tiss_mask is not None else torch.ones(n_spots, dtype=torch.bool, device=coords.device)
        self.observed_counts = observed_counts if observed_counts is not None else torch.zeros((n_spots, n_genes), device=coords.device)
        if initial_counts_guess is not None:
            self.initial_counts_guess = self.robust_log_transform(initial_counts_guess)  # Log-transform to stabilize training
        else:
            self.initial_counts_guess = torch.ones((n_spots, n_genes))
                
        # Add RBF-FD specific initialization
        self.neighbors = neighbors
        self.D = D_out * (1 - in_tiss_mask) + D_in * in_tiss_mask
        self.diffusion_steps = diffusion_steps
        
    def robust_log_transform(self, x, clip_quantile=0.99):
        # Clip extreme values
        upper = torch.quantile(x, clip_quantile)
        x_clipped = torch.clamp(x, max=upper)
        # Log transform
        return torch.log1p(x_clipped)

    def compute_laplacian(self):
        self.neighbors.fill_diagonal_(0)
        degree = torch.diag(self.neighbors.sum(dim=1))
        laplacian = degree - self.neighbors
        D = pyro.param('D')
        alpha_diag = torch.diag(D)  # Pyro param
        return alpha_diag @ laplacian  

    
    def forward_diffusion(self, heat, steps=10):
        laplacian = self.compute_laplacian()  # Recompute each time
        total_heat = heat.sum(dim=0, keepdim=True)
        gene_specific_D = pyro.param("gene_specific_D")
        for _ in range(steps):
            laplacian_update = laplacian @ heat
            laplacian_update = laplacian_update * gene_specific_D  # Pyro param
            heat = heat - laplacian_update
            heat = torch.clamp(heat, min=0.0)
        heat = (heat / heat.sum(dim=0, keepdim=True)) * total_heat
        return heat

    def tissue_boundary_constraint(self, diffused_counts, observed_counts):
        """
        More sophisticated tissue boundary constraint that considers
        distance-dependent diffusion patterns
        """
        # Compute distance to tissue boundary for each spot
        boundary_distances = self.compute_boundary_distances()  # [n_spots]
        
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
    ) -> torch.Tensor:
        """Probabilistic model with debugging"""

        # 1. Check input tensors
        print("\nChecking inputs:")
        self.debug_tensor("observed_counts", self.observed_counts)
        self.debug_tensor("in_tiss_mask", self.in_tiss_mask)
        self.debug_tensor("neighbors", self.neighbors)
        self.debug_tensor('initial_counts_guess', self.initial_counts_guess)
        
        original_counts_loc = pyro.param(
            "original_counts_loc",
            self.initial_counts_guess,
        )
        original_counts_scale = pyro.param(
            "original_counts_scale",
            0.01 * torch.ones_like(self.observed_counts),
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
        L = self.compute_laplacian()
        self.debug_tensor("Laplacian", L.to_dense() if L.is_sparse else L)

        # Forward diffusion with intermediate checks
        diffused_counts = original_counts.clone()
        old_diffused = diffused_counts.clone()
        steps = self.diffusion_steps
        diffused_counts = self.forward_diffusion(old_diffused, steps=steps)

        self.debug_tensor("diffused_counts (after diffusion)", diffused_counts, print_stats=True)
        # 5. Check total_counts and probs computation
        print("\nChecking likelihood computation:")
        total_counts = self.total_counts
        self.debug_tensor("total_counts", total_counts)

        # Compute probs with additional safety
        probs = diffused_counts / total_counts
        self.debug_tensor("probs (before clamping)", probs)
        probs.clamp_(min=1e-6, max=1.0 - 1e-6)  # Ensure probabilities are valid
        self.debug_tensor("probs (after clamping)", probs)
        
        # 6. Sample observations
        with pyro.plate("spots", self.n_spots):
            obs = pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=total_counts,
                    probs=probs
                ).to_event(1),
                obs=self.observed_counts
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
    ) -> torch.Tensor:
        """
        Probabilistic model incorporating physics and data
        
        """
        D = pyro.param(
            "D",
            self.D,
            constraint=constraints.nonnegative
        )
        gene_specific_D = pyro.param(
            "gene_specific_D",
            torch.ones(self.n_genes),
            constraint=constraints.nonnegative
        )
        
        # Prior for original counts
        original_counts_loc = pyro.param(
            "original_counts_loc",
            self.initial_counts_guess,
        )
        original_counts_scale = pyro.param(
            "original_counts_scale",
            0.01 * torch.ones_like(self.observed_counts),
            constraint=torch.distributions.constraints.nonnegative
        )  # [n_spots, n_genes]
        
        # Sample original counts
        original_counts = pyro.sample(
            "original_counts",
            dist.LogNormal(original_counts_loc, original_counts_scale).to_event(2)
        )  # [n_spots, n_genes]
        
        # Physics-based diffusion
        diffused_counts = self.forward_diffusion(original_counts, steps=self.diffusion_steps)  # [n_spots, n_genes]
        # Likelihood incorporating:
        # 1. Observation model
        # 2. Tissue mask constraint
        # 3. Spatial regularization

        # 1. Observation likelihood
        total_counts = self.total_counts
        probs = diffused_counts / total_counts
        probs.clamp_(min=1e-6, max=1.0 - 1e-6)  # Ensure probabilities are valid
        
        # Use Negative Binomial observation model
        with pyro.plate("spots", self.n_spots):
            pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=total_counts,  # shape: [n_spots, 1]
                    probs=probs  # shape: [n_spots, n_genes]
                ).to_event(1),  # event_shape = [n_genes]
                obs=self.observed_counts
            )
        
        # Tissue boundary constraint
        pyro.factor(
            'tissue_constraint', self.tissue_boundary_constraint(
                diffused_counts, self.observed_counts)
        )
        
        # Spatial regularization using neighbor structure to encourage smoothness
        # spatial_diff = torch.sum(
        #     self.neighbors.unsqueeze(2) * 
        #     (diffused_counts.unsqueeze(1) - diffused_counts.unsqueeze(0))**2
        # )
        # spatial_factor = -self.beta * spatial_diff
        # pyro.factor("spatial_regularization", spatial_factor)
        
        return original_counts

    def guide(
        self,
    ) -> torch.Tensor:
        """Variational guide/posterior"""
        # Variational parameters
        original_counts_loc = pyro.param(
            "original_counts_loc",
            self.initial_counts_guess,
        )
        original_counts_scale = pyro.param(
            "original_counts_scale",
            0.01 * torch.ones_like(self.observed_counts),
            constraint=torch.distributions.constraints.nonnegative
        )
        
        # Variational distribution
        return pyro.sample(
            "original_counts",
            dist.LogNormal(original_counts_loc, original_counts_scale).to_event(2)
        )



class PhysicsInformedSparseGP(GPModel):
    """
    Hybrid model combining:
    - Sparse GP's covariance structure
    - Physics-informed diffusion mean function
    """
    
    def __init__(
        self,
        X: torch.Tensor,  # Initial counts guess [n_genes, n_spots]
        y: torch.Tensor,  # Observed counts [n_genes, n_spots]
        coords: torch.Tensor,  # Spatial coordinates [n_spots, 2]
        in_tiss_mask: torch.Tensor,  # Binary tissue mask [n_spots]
        ttl_cnts: torch.Tensor,  # Total counts per spot [n_spots]
        neighbors: torch.Tensor,  # Spatial neighborhood matrix [n_spots, n_spots]
        X_prior: torch.Tensor = None,  # Prior counts [n_genes, n_spots]
        kernel=None,
        D_out: float = 0.1,
        D_in: float = 0.02,
        noise=torch.tensor(0.1),  # Observation noise
        approx="VFE",
        jitter=1e-6,
        diffusion_steps: int = 10,
    ):
        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
            y, torch.Tensor
        ), "y needs to be a torch Tensor instead of a {}".format(type(y))
        
        n_genes, n_spots = X.shape
                
        n_inducing_genes = min(50, int(n_genes**0.5))
        
        if kernel is None: 
            kernel = gp.kernels.RBF(input_dim=X.shape[1], lengthscale=torch.ones(X.shape[1]))  # kernel input_dim is the number of features in X, which is the number of genes
        super().__init__(X, y, kernel, jitter=jitter)
        
        self.noise = PyroParam(noise, constraints.real) if isinstance(noise, torch.Tensor) else PyroParam(torch.tensor(noise, dtype=torch.float32), constraints.real)
        
        # Store diffusion-related parameters
        self.coords = coords
        self.in_tiss_mask = in_tiss_mask
        self.ttl_cnts = ttl_cnts.unsqueeze(0) if ttl_cnts.dim() == 1 else ttl_cnts # Ensure shape is [1, n_spots]
        self.neighbors = neighbors
        self.diffusion_steps = diffusion_steps
        self.n_genes = n_genes
        self.n_spots = n_spots
        self.n_inducing_genes = n_inducing_genes
                
        # Diffusion coefficients
        self.D = PyroParam(
            D_out * (1 - in_tiss_mask.float()) + D_in * in_tiss_mask.float(),
            constraint=constraints.nonnegative
        )
        
        # Gene-specific diffusion coefficients
        self.gene_specific_D = PyroParam(
            torch.ones(X.size(0)),  # [n_genes]
            constraint=constraints.positive
        )
        
        #### for variational inference
        # Initialize counts in log space 
        self.log_X_init = self._robust_log_transform(X)
        self.X_loc = PyroParam(
            self.log_X_init, # [n_genes, n_spots]
            constraint=constraints.real
        )
        # Initialize scale_tril with correlation structure 
        X_normalized = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-6)
        corr_matrix = torch.mm(X_normalized.t(), X_normalized) / X.size(0)
        corr_matrix = corr_matrix + 1e-2 * torch.eye(corr_matrix.size(0), device=X.device)
        self.shared_scale_tril = PyroParam(
            torch.linalg.cholesky(corr_matrix),
            constraint=constraints.lower_cholesky
        )
        
        self.X_prior = X_prior if X_prior is not None else torch.ones_like(X)  # Default prior is zero
        
        # Initialize inducing points using k-means clustering on genes
        inducing_values = self._initialize_inducing_points(X)
        # Initialize inducing points (shape: [n_inducing_genes, n_spots])
        self.Xu = PyroParam(inducing_values, constraint=constraints.real)
        
        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError(
                "The sparse approximation method should be one of "
                "'DTC', 'FITC', 'VFE'."
            )
            
    def _initialize_inducing_points(self, X):
        """Initialize inducing points using k-means on genes"""
        # Convert to numpy for k-means
        X_np = X.cpu().numpy()
        
        # Cluster genes to find representative inducing genes
        kmeans = KMeans(n_clusters=self.n_inducing_genes, random_state=42)
        kmeans.fit(X_np)
        
        # Get cluster centers and convert back to torch
        inducing_values = torch.from_numpy(kmeans.cluster_centers_).to(X.device)
        return inducing_values

    def _robust_log_transform(self, x, clip_quantile=0.99):
        """Helper for stable log transform"""
        upper = torch.quantile(x, clip_quantile)
        x_clipped = torch.clamp(x, max=upper)
        return torch.log1p(x_clipped)

    def compute_laplacian(self):
        """Compute graph Laplacian with diffusion coefficients"""
        self.neighbors.fill_diagonal_(0)
        degree = torch.diag(self.neighbors.sum(dim=1))
        laplacian = degree - self.neighbors
        D_diag = torch.diag(self.D)  # Spot-specific diffusion coefficients
        return D_diag @ laplacian

    def forward_diffusion(self, X, steps=None):
        """
        Physics-informed mean function using diffusion process
        X: [n_genes, n_spots] in original count space
        """
        if steps is None:
            steps = self.diffusion_steps
            
        laplacian = self.compute_laplacian()
        gene_D = self.gene_specific_D.unsqueeze(-1)  # [n_genes, 1]
        X = X.t()  # Transpose to [n_spots, n_genes] for diffusion
        total_counts = X.sum(dim=0, keepdim=True)  # [n_genes, 1]

        for _ in range(steps):
            # Transpose for spatial operation (now [n_spots, n_genes])
            update = laplacian @ X  # [n_spots, n_genes]
            update = update * gene_D  # [n_genes, n_spots]
            X = X - update
            X = torch.clamp(X, min=0.0)
            
        # Preserve total counts per gene
        X = (X / X.sum(dim=0, keepdim=True)) * total_counts
        X = X.t()  # Transpose back to [n_genes, n_spots]
        
        return X

    def mean_function(self, X):
        """
        The physics-informed mean function for the GP
        X: [n_genes, n_spots] in original count space (not log space)
        Returns: [n_genes, n_spots] in count space
        """
        # Apply diffusion (operates in count space)
        diffused = self.forward_diffusion(X)  # [n_genes, n_spots]
        log_diffused = self._robust_log_transform(diffused)  # Log-transform for stability
        return log_diffused
        # # Convert to proportions and scale by total counts
        # probs = diffused / self.ttl_cnts  # [n_genes, n_spots] / [1, n_spots]
        # probs = probs.clamp(min=1e-6, max=1-1e-6)
        
        # # Return in same shape as y [n_genes, n_spots]
        # return probs * self.ttl_cnts.t()

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
        X = pyro.sample(dist.TransformedDistribution(
                base_distribution = dist.MultivariateNormal(
                    self.X_prior, scale_tril=torch.eye(self.n_spots)
                    ),
                transforms = [dist.transforms.ExpTransform()]
            )
        ) 
        N = X.size(0)
        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[:: M + 1] += self.jitter  # add jitter to the diagonal
        assert not torch.isnan(Kuu).any(), "Kuu contains NaNs"
        assert not torch.isinf(Kuu).any(), "Kuu contains Infs"
        Luu = torch.linalg.cholesky(Kuu)  # the Cholesky decomposition of Kuu = Luu @ Luu.T
        Kuf = self.kernel(self.Xu, X)
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

        f_loc = self.mean_function(X)
        
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
                dist.TransformedDistribution(                
                    dist.LowRankMultivariateNormal(f_loc, W, D).to_event(1),
                transforms=[dist.transforms.ExpTransform()]),  # Transform back to original space
                obs=self.y,
            )

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()
        pyro.sample(
            'X',
            dist.TransformedDistribution(
                base_distribution=dist.MultivariateNormal(
                    self.X_loc, scale_tril=self.shared_scale_tril
                ),
                transforms=[dist.transforms.ExpTransform()]
            )
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
