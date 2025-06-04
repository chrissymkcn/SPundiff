from concurrent.futures import ProcessPoolExecutor
import torch
from joblib import Parallel, delayed
import torch
from torch.distributions import constraints
from torch.nn import Parameter
from functools import partial

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.nn.module import PyroParam, pyro_method
import pyro.ops.stats as stats
import pyro.contrib.gp as gp

from undiff_model_torch_optim_global import undiff_global
from diffusion_sim import forward_diffusion, calculate_domain_parameters
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

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
        self, X, y, kernel, Xu, noise=None, mean_function=None, approx=None, jitter=1e-6
    ):
        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
                y, torch.Tensor
            ), "y needs to be a torch Tensor instead of a {}".format(type(y))
        assert isinstance(
            Xu, torch.Tensor
        ), "Xu needs to be a torch Tensor instead of a {}".format(type(Xu))

        super().__init__(X, y, kernel, mean_function, jitter)

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
        Luu = torch.linalg.cholesky(Kuu)
        Kuf = self.kernel(self.Xu, self.X)
        W = torch.linalg.solve_triangular(Luu, Kuf, upper=False).t()

        D = self.noise.expand(N)
        if self.approx == "FITC" or self.approx == "VFE":
            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=-1)
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


class undiff_prob(undiff_global):
    def __init__(self, adata, n_jobs=1, metric='euclidean', optimizer='adam', optim_params=None):
        super().__init__(adata, n_jobs, metric, optimizer, optim_params)

    def freeze_all(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.requires_grad = False

    # def gene_specific_adaptation(self, out_dist, in_dist, out_sum, in_sum, reg):
    #     # Get gene-specific distributions
    #     global_ot = torch.ones_like(self.global_ot, dtype=torch.float32)  # [n_spots, n_spots]    
    #     ## assume out_dist and in_dist are already normalized to sum to 1
    #     # Compute scaling factors
    #     row_scale = out_dist / global_ot.sum(1)
    #     col_scale = in_dist / global_ot.sum(0)
        
    #     # Compute adapted transport (no need for full matrix)
    #     to_target = (global_ot * row_scale.unsqueeze(1) * col_scale.unsqueeze(0)).sum(0)
    #     to_target = to_target / to_target.sum()
    #     to_return = to_target * out_sum + in_dist * in_sum
    #     return to_return  # [n_spots, 1]

    def compute_res_count(self, params):
        regs = params['regs']
        invalid_qts = params['invalid_qts']
        genes = self.gene_selected
        out_tiss_filt, in_tiss_filt, out_tiss_sum, in_tiss_sum = self.prep(invalid_qts)
        
        batch_size = 10
        res = []
        for i, gene in enumerate(genes):
            # Checkpointing for memory-efficient OT computation
            if i % batch_size == 0:
                print(f'Processing gene {i+1}/{len(genes)}: {gene}')
            # Use checkpoint for memory-efficient OT computation
            g_out, g_in = out_tiss_filt[:, i], in_tiss_filt[:, i]
            g_outsum, g_insum = out_tiss_sum[i], in_tiss_sum[i]
            # transported_in = self.gene_specific_adaptation(g_out, g_in, g_outsum, g_insum)
            transported_in = self.gene_specific_adaptation(g_out, g_in, g_outsum, g_insum, regs[i])
            res.append(transported_in)
        self.res_count = torch.stack(res, dim=1)
        

    @staticmethod
    def model_init(algo, i):
        if algo == KMeans:
            model = algo(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        elif algo == AgglomerativeClustering:
            model = algo(n_clusters=i)
        elif algo == SpectralClustering:
            model = algo(n_clusters=i, affinity='nearest_neighbors', n_init=10)
        elif algo == Birch:
            model = algo(n_clusters=i)
        elif algo == BayesianGaussianMixture:
            model = algo(n_components=i, weight_concentration_prior=1.0, max_iter=100, random_state=0)
        elif algo == GaussianMixture:
            model = algo(n_components=i, covariance_type='full', max_iter=100, random_state=0)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
        return model

    @staticmethod
    def find_optimal_ncls(data, methods, max_clusters=None, starting_clusters=3):
        if max_clusters is None:
            max_clusters = np.ceil(np.sqrt(data.shape[0]))  # Default to sqrt of number of samples
        max_clusters = int(max_clusters) + 5
        # import common sklearn clustering algorithms
        scs = {}
        # wcsses = {}
        max_clusters = np.ceil(np.sqrt(data.shape[0])).astype(int)
        cluster_range = range(starting_clusters, max_clusters + 1)

        for algo in methods:
            scs[f'{algo.__name__}'] = []
            # wcsses[f'{algo.__name__}'] = [] if algo == KMeans else None
            for i in cluster_range:
                model = undiff_prob.model_init(algo, i)
                predicted_labels = model.fit_predict(data)
                # if hasattr(model, 'inertia_'):
                #     wcsses[f'{algo.__name__}'].append(model.inertia_)
                scs[f'{algo.__name__}'].append(silhouette_score(data, predicted_labels))

        # calculate the optimal number of clusters by averaging the silhouette scores for each n_clusters
        df = pd.DataFrame(scs)
        df.index += starting_clusters
        optimal_clusters = df.mean(axis=1).sort_values(ascending=False).index[:3]  # top 3 cluster numbers based on average silhouette score
        algo = df.loc[optimal_clusters, :].mean(axis=0).idxmax()  # algo with the highest average silhouette score among the top 3 cluster numbers
        return optimal_clusters, algo

    @staticmethod
    def auto_cluster(data, starting_clusters=3, max_clusters=None):
        """
        Calculate silhouette scores for PCA-reduced data and return the optimal number of clusters.
        """
        # Standardize the data
        data = StandardScaler().fit_transform(data)
        from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
        from sklearn.metrics import silhouette_score
        methods = [KMeans, AgglomerativeClustering, SpectralClustering, Birch, GaussianMixture, BayesianGaussianMixture]
            
        # Reduce dimensionality using PCA
        pca_silhouette_scores = {}
        best_algos = []
        for n_pca in range(5, min(30, data.shape[0]), 5):
            pca = PCA(n_components=n_pca)
            data_pca = pca.fit_transform(data)
            top_ncl, algo = undiff_prob.find_optimal_ncls(data_pca, methods, max_clusters=max_clusters, starting_clusters=starting_clusters)
            pca_silhouette_scores[n_pca] = top_ncl
            best_algos.append(algo)
        
        # Convert to DataFrame for better visualization
        pca_silhouette_df = pd.DataFrame(pca_silhouette_scores).T  # dim: (n_pcas, top_choices=3)

        # Find the optimal number of clusters based on the mode of the silhouette scores
        optimal_clusters = pca_silhouette_df.mode().values.max()  # here we take the mode of top choices (most chosen number of clusters across all PCA num choices)
        # then take the maximum of the mode values (could use [0] instead, 
        # essentially the most chosen top 1 number of clusters, across all PCA numbers)

        # # find the optimal number of PCA components based on the mode of the silhouette scores
        # col = pca_silhouette_df.mode().T.idxmax()[0]
        # # print(col)
        # optimal_pca = pca_silhouette_df.loc[pca_silhouette_df[col] == optimal_clusters, col].index[0]
        # pca = PCA(n_components=optimal_pca).fit_transform(data)

        optimal_algo = np.unique(best_algos)[0]
        actual_algo = [algo for algo in methods if algo.__name__ == optimal_algo][0]
        model = undiff_prob.model_init(actual_algo, optimal_clusters)
        cluster_labels = model.fit_predict(data)
        return {
            'cluster_labels': cluster_labels,
            'pca_silhouette_df': pca_silhouette_df,
            'optimal_clusters': optimal_clusters,
            # 'pca': pca,
        }
    
    def run_one_round(self, qts_prior=0.8, n_genes=None, add_genes=[], optim_params=None):
        self.params.update(optim_params) if optim_params is not None else None
        self.prep_genes_params(add_genes=add_genes, first_n_genes=n_genes)
        self.set_states(qts_prior=qts_prior)
        self.freeze_all()  # no need to update parameters during this round
        self.run_ot({
            'cost_weight_s': self.cost_weight_s,
            'cost_weight_t': self.cost_weight_t,
            'invalid_qts': self.invalid_qts,
            'regs': self.regs,
            'global_reg': self.global_reg
        })

    def get_clustered_embeddings(self, n_genes=100, qts_prior=0.8):
        if self.res_count is None:
            self.run_one_round(n_genes=n_genes, qts_prior=qts_prior)
        res_count = self.res_count.detach().cpu().numpy()  # Convert to numpy for clustering
        res = undiff_prob.auto_cluster(res_count.T, starting_clusters=3, max_clusters=None)  # pass in transposed res_count to have genes as samples
        self.gene_clusters = {self.gene_selected[i]: lab for i,lab in enumerate(res['cluster_labels'])}
        # compute barycenters of each cluster
        unique_clusters = np.unique(res['cluster_labels'])
        gene_groups = {}
        z_prior = {}
        self.cluster_genes_dict = {}
        for i in unique_clusters:
            # get indices of genes in this cluster
            cluster_genes_idx = torch.tensor(np.where(res['cluster_labels'] == i)[0])
            self.cluster_genes_dict[i] = np.array(self.gene_selected)[np.where(res['cluster_labels'] == i)[0]]
            separated_chunk = self.res_count[:, cluster_genes_idx]
            print(type(separated_chunk))
            # twod_genes = []
            # for j in range(len(cluster_genes)):
                # gene_twod = self.grid_embedding(separated_chunk[:, j])
            #     gene_twod = gene_twod / gene_twod.sum()  # Normalize to sum to 1
            #     twod_genes.append(torch.tensor(gene_twod))
            # separated_chunk = torch.stack(twod_genes, dim=0)
            gene_groups[i] = separated_chunk.clone()  # do not embed here, just use expression vector directly
            z_prior[i] = torch.mean(separated_chunk, dim=1)
        return gene_groups, z_prior, self.sub_count

    # @staticmethod
    # def mean_function(X, coords, sub_count, in_tiss_mask, ttl_cnts):
    #     '''
    #     apply forward diffusion to the input data X
    #     '''
    #     domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes = calculate_domain_parameters(coords, divideby=1)
        
    #     res = []
    #     for i in range(X.shape[0]):
    #         x = X[i, :]
    #         y = sub_count[:, i]
    #         # x = x / x.sum()
    #         # y = y / y.sum()
    #         model = forward_diffusion(
    #             grid_sizes=grid_sizes, voxel_sizes=voxel_sizes, padding_sizes=padding_sizes,
    #             x=x, y=y, coords=coords, in_tiss_mask=in_tiss_mask, ttl_cnts=ttl_cnts,
    #             diffusivity=0.2, noise=0.01,
    #         )
    #         x_diffused = model.run()
    #         x_diffused = torch.tensor(x_diffused, dtype=torch.float32)
    #         res.append(x_diffused)
    #     X_u_diffused = torch.stack(res, dim=0)
        
    #     return X_u_diffused.T

    @staticmethod
    def mean_function(X, coords, sub_count, in_tiss_mask, ttl_cnts):
        domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes = calculate_domain_parameters(coords, divideby=1)

        def process(i):
            x = X[i, :]
            y = sub_count[:, i]
            model = forward_diffusion(
                grid_sizes=grid_sizes, voxel_sizes=voxel_sizes, padding_sizes=padding_sizes,
                x=x, y=y, coords=coords, in_tiss_mask=in_tiss_mask, ttl_cnts=ttl_cnts,
                diffusivity=0.2, noise=0.01,
            )
            return torch.tensor(model.run(), dtype=torch.float32)

        res = Parallel(n_jobs=-1)(delayed(process)(i) for i in range(X.shape[0]))
        return torch.stack(res, dim=0).T

    # @staticmethod
    # def process_one(i, X, sub_count, coords, in_tiss_mask, ttl_cnts, grid_sizes, voxel_sizes, padding_sizes):
    #     x = X[i, :]
    #     y = sub_count[:, i]
    #     model = forward_diffusion(
    #         grid_sizes=grid_sizes, voxel_sizes=voxel_sizes, padding_sizes=padding_sizes,
    #         x=x, y=y, coords=coords, in_tiss_mask=in_tiss_mask, ttl_cnts=ttl_cnts,
    #         diffusivity=0.2, noise=0.01,
    #     )
    #     x_diffused = model.run()
    #     return torch.tensor(x_diffused, dtype=torch.float32)
    
    # @staticmethod
    # def mean_function(X, coords, sub_count, in_tiss_mask, ttl_cnts):
    #     domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes = calculate_domain_parameters(coords, divideby=1)
    #     with ProcessPoolExecutor() as executor:
    #         futures = [executor.submit(undiff_prob.process_one, i, X, sub_count, coords, in_tiss_mask, ttl_cnts, grid_sizes, voxel_sizes, padding_sizes)
    #                 for i in range(X.shape[0])]
    #         res = [f.result() for f in futures]
    #     return torch.stack(res, dim=0).T


    def run_sgpr(self, n_genes=1000, qts_prior=0.8, train_steps=2000, figname='sgpr_loss_curve.png'):
        """
        Run Sparse GP Regression on the gene expression data.
        """
        gene_groups, z_prior, sub_count = self.get_clustered_embeddings(n_genes=n_genes, qts_prior=qts_prior)
        n_spots = gene_groups[0].shape[1]
        X_prior_mean = torch.cat([ts.T for ts in gene_groups.values()], dim=0)  # [n_genes, n_spots]
        X = Parameter(X_prior_mean.clone())
        n_inducing = np.sqrt(n_genes).astype(int)  # number of inducing points
        if len(z_prior) < n_inducing:
            more_to_sample = n_inducing - len(z_prior)
            additional = stats.resample(X_prior_mean.clone(), more_to_sample)
            Xu = torch.cat(
                [val.unsqueeze(0) for val in z_prior.values()] + 
                [additional], 
                dim=0)  # [n_inducing, n_spots]
        else:
            Xu = torch.cat(
                [val.unsqueeze(0) for val in z_prior.values()], 
                dim=0)[:n_inducing, :]  # [n_inducing, n_spots]
        y = torch.tensor(sub_count, dtype=torch.float32)  # [n_spots, n_genes]
        kernel = gp.kernels.RBF(input_dim=X.shape[1], lengthscale=torch.ones(X.shape[1]))  # kernel input_dim is the number of features in X, which is the number of genes

        coords = self.coords - self.coords.mean(dim=0)  # center the coordinates
        ttl_cnts = self.adata.obs['total_counts'].values
        ttl_cnts = torch.tensor(ttl_cnts, dtype=torch.float32)
        mean_func = partial(self.mean_function, coords=coords, sub_count=sub_count,
                            in_tiss_mask=self.in_tiss_mask, ttl_cnts=ttl_cnts)

        gplvm = SparseGPRegression(X, y, kernel, Xu, 
                                noise=torch.tensor(0.001), jitter=1e-5, approx='VFE', 
                                mean_function=mean_func,
                                )
        gplvm.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean, 0.1).to_event())
        gplvm.autoguide("X", dist.Normal)
        
        # note that training is expected to take a minute or so
        losses = gp.util.train(gplvm, num_steps=train_steps)

        # let's plot the loss curve after 4000 steps of training
        plt.figure(figsize=(10, 5))
        plt.title("Sparse GP Regression Loss Curve")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.grid()
        plt.plot(losses, label='Loss')
        plt.legend()
        plt.savefig(figname)

        # return the model 
        return gplvm, losses