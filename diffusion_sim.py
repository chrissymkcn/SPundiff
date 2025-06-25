import numpy as np
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
        X: torch.Tensor,  # Initial counts guess [n_genes, n_spots]
        y: torch.Tensor,  # Observed counts [n_genes, n_spots]
        X_prior: torch.Tensor,  # prior counts [n_genes, n_spots]
        coords: torch.Tensor,
        in_tiss_mask: torch.Tensor,
        ttl_cnts: torch.Tensor,
        neighbors: torch.Tensor,  # Spatial neighborhood matrix
        D_out: float = 0.1,
        D_in: float = 0.02,
        diffusion_steps: int = 10,
        alpha: float = 0.1,  # Tissue mask weight
        beta: float = 0.1,  # Spatial regularization weight
        cluster_labels: torch.Tensor = None,  # Optional cluster labels for spatial regions
    ):
        super().__init__()
        n_genes, n_spots = X.shape
        self.n_spots = n_spots
        self.n_genes = n_genes
        self.coords = coords
        # self.neighbor_graph = neighbor_graph
        self.alpha = alpha
        self.beta = beta
        if ttl_cnts.dim() == 1:
            cntlen = ttl_cnts.shape[0]
            if cntlen == n_spots:
                ttl_cnts = ttl_cnts.unsqueeze(0)
            elif cntlen == n_genes:
                ttl_cnts = ttl_cnts.unsqueeze(1)
        self.ttl_counts = ttl_cnts
        self.in_tiss_mask = in_tiss_mask if in_tiss_mask is not None else torch.ones(n_spots, dtype=torch.bool, device=coords.device)
        self.y = y
        self. X = self.robust_log_transform(X)  # Log-transform to stabilize training
        # Initialize scale_tril with correlation structure 
        X_normalized = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)
        corr_matrix = torch.mm(X_normalized.t(), X_normalized) / X.size(0)
        corr_matrix = corr_matrix + 1e-2 * torch.eye(corr_matrix.size(0), device=X.device)
        self.corr_matrix = torch.linalg.cholesky(corr_matrix)

        self.X_prior = self.robust_log_transform(X_prior)
        # Add RBF-FD specific initialization
        self.neighbors = neighbors
        self.D = D_out * (1 - in_tiss_mask) + D_in * in_tiss_mask
        self.diffusion_steps = diffusion_steps
        self.laplacian = self.compute_laplacian().to(torch.float32).detach()
        # self.laplacian = self.laplacian / (self.laplacian.norm(p=2) + 1e-6)  # Normalize Laplacian for stability
        self.cluster_labels = cluster_labels
        
    def robust_log_transform(self, x, clip_quantile=0.97):
        # Clip extreme values
        upper = torch.quantile(x, clip_quantile)
        x_clipped = torch.clamp(x, max=upper, min=0.00)
        # x_clipped = torch.log(x_clipped)
        # Log transform
        return x_clipped

    def compute_laplacian(self):
        self.neighbors.fill_diagonal_(0)
        degree = torch.diag(self.neighbors.sum(dim=1))
        laplacian = degree - self.neighbors
        return laplacian  
    
    def forward_diffusion(self, initial_heat, steps=1):
        """Simulate multi-step heat diffusion"""
        heat = initial_heat.t().clone()  # [n_spots, n_genes]
        laplacian = self.laplacian
        alpha = pyro.param('D')  # [n_spots]
        gene_specific_D = pyro.param('gene_specific_D')  # [n_genes]

        for _ in range(steps):
            update = laplacian @ heat  # [n_spots, n_genes]
            update = update * gene_specific_D.unsqueeze(0)  # scale per gene
            heat = heat - alpha.unsqueeze(1) * update
            heat = torch.clamp(heat, min=0.0)

        return heat.t()  # [n_genes, n_spots]

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
        """Probabilistic model with debugging (all params are variational parameters)"""
        # 1. Check input tensors
        print("\nChecking inputs:")
        self.debug_tensor("y", self.y)
        self.debug_tensor("in_tiss_mask", self.in_tiss_mask)
        self.debug_tensor("neighbors", self.neighbors)
        self.debug_tensor('X', self. X)
        
        X_loc = pyro.param(
            "X_loc",
            self. X,
        )
        X_scale = pyro.param(
            "X_scale",
            self.corr_matrix,
            constraint=constraints.lower_cholesky
        )
        
        print("\nChecking parameters:")
        self.debug_tensor("X_loc", X_loc)
        self.debug_tensor("X_scale", X_scale)

        # 3. Sample original counts
        X = X_loc  # Convert log counts back to original space
        self.debug_tensor("X (after sampling)", X)

        # 4. Check diffusion process
        print("\nChecking diffusion process:")
        # Debug Laplacian computation
        L = self.laplacian
        self.debug_tensor("Laplacian", L.to_dense() if L.is_sparse else L)

        # Forward diffusion with intermediate checks
        steps = self.diffusion_steps
        pyro.param('D', self.D, constraint=constraints.nonnegative)  # Ensure D is a Pyro param
        gene_specific_D = torch.ones(self.n_genes, device=X.device)  # Default gene-specific D
        pyro.param('gene_specific_D', gene_specific_D, constraint=constraints.positive)  # Ensure gene-specific D is a Pyro param
        
        diffused_counts = self.forward_diffusion(X.clone(), steps=steps)

        self.debug_tensor("diffused_counts (after diffusion)", diffused_counts, print_stats=True)
        # 5. Check ttl_counts and probs computation
        print("\nChecking likelihood computation:")
        ttl_counts = self.ttl_counts
        self.debug_tensor("ttl_counts", ttl_counts)

        # Compute probs with additional safety
        probs = diffused_counts / ttl_counts
        self.debug_tensor("probs (before clamping)", probs)
        probs.clamp_(min=1e-6, max=1.0 - 1e-6)  # Ensure probabilities are valid
        self.debug_tensor("probs (after clamping)", probs)
        
        # 6. Sample observations
        with pyro.plate("genes", self.n_genes):
            obs = pyro.sample(
                'obs',
                dist.NegativeBinomial(
                    total_count=ttl_counts,  # shape: [1, n_spots]
                    probs=probs  # shape: [n_genes, n_spots]
                ).to_event(1),  # Treat first dimension as batch size
                obs=self.y  # Observed counts [n_genes, n_spots]
            )
        self.debug_tensor("obs (after sampling)", obs)

        return {
            "X": X,
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
            constraint=constraints.half_open_interval(0.0, 0.1)
        )
        gene_specific_D = pyro.param(
            "gene_specific_D",
            torch.ones(self.n_genes),
            constraint=constraints.half_open_interval(0.0, 0.1)
        )
        
        # Prior for original counts
        # Sample original counts
        # xd = dist.TransformedDistribution(
        #         base_distribution = dist.MultivariateNormal(
        #             self.X_prior, scale_tril=self.corr_matrix
        #             ),
        #         transforms = [dist.transforms.ExpTransform()]
        #     )
        # print("\nxd batch shape:", xd.batch_shape, "event shape:", xd.event_shape)
        # with pyro.plate("genes", self.n_genes):
        #     X = pyro.sample('X', xd)
        
        X = pyro.param('X', self. X, constraint=constraints.nonnegative)
        
        # Physics-based diffusion
        diffused_counts = self.forward_diffusion(X, steps=self.diffusion_steps)  # [n_genes, n_spots]
        # self.debug_tensor("diffused_counts (after diffusion)", diffused_counts, print_stats=True)

        # 1. Observation likelihood
        ttl_counts = pyro.param('ttl_cnts', self.ttl_counts, constraint=constraints.nonnegative)  # [1, n_spots]
        probs = diffused_counts / ttl_counts  # [n_genes, n_spots]
        probs.clamp_(min=1e-6, max=1.0 - 1e-6)  # Ensure probabilities are valid
        
        # Use Negative Binomial observation model
        # with pyro.plate("genes", self.n_genes):
        d = dist.NegativeBinomial(
                    total_count=ttl_counts,  # shape: [n_genes, n_spots]
                    probs=probs              # shape: [n_genes, n_spots]
                ).to_event(1)  # Treat first dimension as batch size

        with pyro.plate("genes_obs", self.n_genes):
            pyro.sample(    
                    'obs',
                    d,
                    obs=self.y
                )
        
        # increase cluster separation
        if self.cluster_labels is not None:
            cluster_separation = self._cluster_separation_loss(X, self.cluster_labels)
            pyro.factor("cluster_separation_loss", -self.alpha * cluster_separation)

        # spatial smoothness penalty
        lap = self.laplacian
        norm_lap = lap / (lap.sum(dim=1, keepdim=True) + 1e-6)
        smoothness_penalty = torch.sum((self.X @ norm_lap) * self.X)
        pyro.factor("spatial_smoothness", -self.beta * smoothness_penalty)
                
        return X


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
    
    def guide(
        self,
    ) -> torch.Tensor:
        """Variational guide/posterior"""
        # # Variational parameters
        # X_loc = pyro.param(
        #     "X_loc",
        #     self. X,
        # )
        # X_scale = pyro.param(
        #     "X_scale",
        #     self.corr_matrix,
        #     constraint=constraints.lower_cholesky
        # )
        
        # # Variational distribution
        # xd = dist.TransformedDistribution(
        #         base_distribution=dist.MultivariateNormal(
        #             X_loc, scale_tril=X_scale
        #         ),
        #         transforms=[dist.transforms.ExpTransform()]
        #     )
        # with pyro.plate("genes", self.n_genes):
        #     pyro.sample('X', xd)
        pass




class PhysicsInformedSparseGP(GPModel):
    """
    Hybrid model combining:
    - Sparse GP's covariance structure
    - Physics-informed diffusion mean function
    """
    
    def __init__(
        self,
        X: torch.Tensor,  # Initial counts guess [n_genes, n_spots]
        y: torch.Tensor,  # Observed counts [n_spots, n_genes]
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
        X = X.to(torch.float32)
        n_inducing_genes = min(20, int(n_genes**0.5))
        
        if kernel is None: 
            kernel = gp.kernels.RBF(input_dim=X.shape[1], lengthscale=torch.ones(X.shape[1]))  # kernel input_dim is the number of features in X, which is the number of genes
        super().__init__(X, y, kernel, jitter=jitter, mean_function=self.forward_diffusion)
        
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
        self.laplacian = self.compute_laplacian().to(torch.float32).detach()
        #### for variational inference
        # Initialize counts in log space 
        self. X = self._robust_log_transform(X)
        self.X_loc = PyroParam(
            self. X, # [n_genes, n_spots]
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
        self.X_prior = self._robust_log_transform(self.X_prior)  # Log-transform prior counts
        
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
        """Memory-efficient log transform"""
        with torch.no_grad():  # Don't track gradients here to save memory
            upper = torch.quantile(x, clip_quantile)
            x_clipped = torch.clamp(x, max=upper)
        return torch.log1p(x_clipped)
    
    def compute_laplacian(self):
        """Compute graph Laplacian with diffusion coefficients"""
        self.neighbors.fill_diagonal_(0)
        degree = torch.diag(self.neighbors.sum(dim=1))
        laplacian = degree - self.neighbors
        return laplacian  # no gradients needed here

    # def forward_diffusion(self, X):
    #     """
    #     Physics-informed mean function using diffusion process
    #     X: [n_genes, n_spots] in original count space
    #     """
    #     print(X.shape)
    #     laplacian = self.D * self.laplacian  # Laplacian with diffusion coefficients [n_spots, n_spots]
    #     gene_D = self.gene_specific_D.unsqueeze(0)  # [1, n_genes]
    #     X = X.t()  # Transpose to [n_spots, n_genes] for diffusion
    #     ttl_counts = X.sum(dim=0, keepdim=True)  # [1, n_genes]

    #     for _ in range(self.diffusion_steps):
    #         update = laplacian @ X  # [n_spots, n_genes]
    #         update = gene_D * update  # Scale by gene-specific diffusion coefficients result = [n_spots, n_genes]
    #         X = X - update
    #         X = torch.clamp(X, min=0.0)
            
    #     # Preserve total counts per gene
    #     X = (X / X.sum(dim=0, keepdim=True)) * ttl_counts
    #     print(X.shape)
    #     return X
    
    def forward_diffusion(self, initial_heat, steps=1):
        """Simulate one step of heat diffusion"""
        heat = initial_heat.t().clone()
        laplacian = self.laplacian
        alpha = pyro.param('D')
        gene_specific_D = pyro.param('gene_specific_D').unsqueeze(0) if pyro.param('gene_specific_D').ndim==1 else pyro.param('gene_specific_D')
        for _ in range(steps):
            heat = heat - alpha * laplacian @ heat * gene_specific_D
        return heat

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
        with pyro.plate("genes", self.n_genes):
            X = pyro.sample(
                'X',
                dist.TransformedDistribution(
                    base_distribution = dist.MultivariateNormal(
                        self.X_prior, scale_tril=torch.eye(self.n_spots)
                        ),
                    transforms = [dist.transforms.ExpTransform()]
                )
            ) 
        N = X.size(0)
        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()  # [M, M]
        Kuu.view(-1)[:: M + 1] += self.jitter  # add jitter to the diagonal
        assert not torch.isnan(Kuu).any(), "Kuu contains NaNs"
        assert not torch.isinf(Kuu).any(), "Kuu contains Infs"
        Luu = torch.linalg.cholesky(Kuu)  # the Cholesky decomposition of Kuu = Luu @ Luu.T shape [M, M]
        Kuf = self.kernel(self.Xu, X)  # [M, N]
        W = torch.linalg.solve_triangular(Luu, Kuf, upper=False).t()  # W = inv(Luu).T @ Kuf = Kfu @ inv(Luu).T (an approximation of Kfu @ inv(Kuu))  shape [N, M]
        print('W shape:', W.shape)
        D = self.noise.expand(N)
        if self.approx == "FITC" or self.approx == "VFE":
            Kffdiag = self.kernel(self.X, diag=True)  # diagonal of Kff  shape [N]
            Qffdiag = W.pow(2).sum(dim=-1)  # shape [N]
            if self.approx == "FITC":
                D = D + Kffdiag - Qffdiag
            else:  # approx = "VFE"
                trace_term = (Kffdiag - Qffdiag).sum() / self.noise  # shape []
                trace_term = trace_term.clamp(min=0)
                print(f"Trace term shape: {trace_term.shape}, value: {trace_term.item()}")
        
        f_loc = self.mean_function(X)
        self.debug_tensor("f_loc", f_loc, print_stats=True)
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
                    dist.LowRankMultivariateNormal(f_loc, W, D).to_event(self.y.dim() - 1),  # num_data (num_genes) now become dimensions dependent on the number of GP (num_spots)
                obs=self.y,
            )
            
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
    

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()
        with pyro.plate("genes", self.n_genes):
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
