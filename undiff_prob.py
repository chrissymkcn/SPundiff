import torch
from torch.nn import Parameter
from functools import partial
import pyro.contrib.gp as gp

from diffusion_sim import SparseGPRegression, EarlyStopping

from undiff_model_torch_optim import undiff
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class undiff_prob(undiff):
    def __init__(self, adata, n_jobs=1, metric='euclidean', optimizer='adam', optim_params=None):
        super().__init__(adata, n_jobs, metric, optimizer, optim_params)

    def freeze_all(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.requires_grad = False

    @staticmethod
    def clustering_model_init(algo, i):
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
                model = undiff_prob.clustering_model_init(algo, i)
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
        max_clusters = data.shape[0] // 2 if max_clusters is None else max_clusters
        for n_pca in range(5, min(30, data.shape[0]), 5):
            pca = PCA(n_components=n_pca)
            data_pca = pca.fit_transform(data)
            top_ncl, algo = undiff_prob.find_optimal_ncls(data_pca, methods, max_clusters=max_clusters, starting_clusters=starting_clusters)
            pca_silhouette_scores[n_pca] = top_ncl
            best_algos.append(algo)
        
        # Convert to DataFrame for better visualization
        pca_silhouette_df = pd.DataFrame(pca_silhouette_scores).T  # dim: (n_pcas, top_choices=3)
        # Find the optimal number of clusters based on the mode of the silhouette scores
        optimal_clusters = pca_silhouette_df.mode() # here we take the mode of top choices (most chosen number of clusters across all PCA num choices)
        optimal_clusters = optimal_clusters.iloc[0].values.max()  # get the first row of the mode values
        # then take the maximum of the mode values (could use [0] instead, 
        # essentially the most chosen top 1 number of clusters, across all PCA numbers)
        # # find the optimal number of PCA components based on the mode of the silhouette scores
        # col = pca_silhouette_df.mode().T.idxmax()[0]
        # # print(col)
        # optimal_pca = pca_silhouette_df.loc[pca_silhouette_df[col] == optimal_clusters, col].index[0]
        # pca = PCA(n_components=optimal_pca).fit_transform(data)
        optimal_algo = np.unique(best_algos)[0]
        actual_algo = [algo for algo in methods if algo.__name__ == optimal_algo][0]
        model = undiff_prob.clustering_model_init(actual_algo, optimal_clusters)
        cluster_labels = model.fit_predict(data)
        return {
            'cluster_labels': cluster_labels,
            'pca_silhouette_df': pca_silhouette_df,
            'optimal_clusters': optimal_clusters,
            # 'pca': pca,
        }
    
    def run_initialization(self, qts_prior=0.8, n_genes=None, add_genes=[]):
        self.prep_genes_params(add_genes=add_genes, first_n_genes=n_genes)
        self.set_states(qts_prior=qts_prior)
        self.freeze_all()  # no need to update parameters during this round
        # self.cost_matrix_calculation(cost_weight_s, cost_weight_s) # for computing scaled cost matrix
        self.cost_matrix_calculation(self.cost_weight_s, self.cost_weight_t) # for computing scaled cost matrix
        self.compute_shared_OT(self.global_reg) # for updating warmstart
        self.compute_res_count({
            'invalid_qts': self.invalid_qts,
            'regs': self.regs,
        })
        
    def gene_initialization(self, gene_expr_out, gene_expr_in, out_sum, in_sum, reg):
        """
        Improved gene-specific adaptation using global OT plan as structural prior
        
        Args:
            gene_expr_out: [n_spots] - source distribution for this gene
            gene_expr_in: [n_spots] - target distribution for this gene  
            reg: scalar - gene-specific regularization strength
            
        Returns:
            transported_in: [n_spots] - adapted transport for this gene
        """
        global_ot = self.global_ot  # [n_spots, n_spots]
        # Normalize gene-specific distributions
        p = gene_expr_out / (gene_expr_out.sum())  # Source
        q = gene_expr_in / (gene_expr_in.sum())    # Target
        # Compute scaling factors in log domain for stability
        # Gene-specific adaptations
        f = (p / (global_ot.sum(dim=1)))  # Source adaptation
        g = (q / (global_ot.sum(dim=0)))  # Target adaptation
        # Regularization - higher reg means stay closer to global plan
        reg_weight = torch.sigmoid(reg)  # Maps reg ∈ (-∞,∞) to (0,1)
        # Adapt the global plan
        adapted_ot = (
            global_ot
            * (f.unsqueeze(1) ** reg_weight) # Adapt source
            * (g.unsqueeze(0) ** reg_weight)     # Adapt target
        )
        # Compute transported mass
        to_target = adapted_ot.sum(dim=0)  # [n_spots]
        # Preserve original scale
        transported_in = to_target * out_sum + gene_expr_in * in_sum
        return transported_in

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
            transported_in = self.gene_initialization(g_out, g_in, g_outsum, g_insum, regs[i])
            # transported_in = self.compute_ot(g_out, g_in, g_outsum, g_insum, regs[i])
            res.append(transported_in)
        self.res_count = torch.stack(res, dim=1)
        
    def get_initialized_embeddings(self, n_genes=100, qts_prior=0.8, clustering=False):
        if self.res_count is None:
            self.run_initialization(n_genes=n_genes, qts_prior=qts_prior)
        if clustering:
            res = undiff_prob.auto_cluster(self.res_count.detach().cpu().numpy().T, starting_clusters=3, max_clusters=None)  # pass in transposed res_count to have genes as samples
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
        else:
            gene_groups = self.res_count.clone().T  # use the full gene expression matrix
            z_prior = {}
        return gene_groups, z_prior, self.sub_count

    def run_sgpr(self, n_genes=1000, qts_prior=0.8, train_steps=2000, 
                noise=0.001, sgpr_approx='VFE',
                figname='sgpr_loss_curve.png'):
        """
        Run Sparse GP Regression on the gene expression data.
        """
        gene_groups, z_prior, sub_count = self.get_initialized_embeddings(n_genes=n_genes, qts_prior=qts_prior, clustering=False)
        X = torch.cat([ts.T for ts in gene_groups.values()], dim=0) if isinstance(gene_groups, dict) else gene_groups
        y = torch.tensor(sub_count, dtype=torch.float32) if not isinstance(sub_count, torch.Tensor) else sub_count
        kernel = gp.kernels.RBF(input_dim=X.shape[1], lengthscale=torch.ones(X.shape[1]))  # kernel input_dim is the number of features in X, which is the number of genes
        coords = self.coords - self.coords.mean(dim=0)  # center the coordinates
        ttl_cnts = self.adata.obs['total_counts'].values
        ttl_cnts = torch.tensor(ttl_cnts, dtype=torch.float32)
        
        gplvm = SparseGPRegression(X, y, kernel, 
                                coords=coords, in_tiss_mask=self.in_tiss_mask, ttl_cnts=ttl_cnts,
                                noise=torch.tensor(noise), jitter=1e-5, approx=sgpr_approx, 
                                )
        
        from pyro.infer import SVI, TraceMeanField_ELBO
        from pyro.optim import ClippedAdam

        def train_model(model, num_epochs=1000, lr=0.01, patience=10, min_delta=1.0):
            # Setup optimizer and ELBO
            optimizer = ClippedAdam({"lr": lr})
            elbo = TraceMeanField_ELBO()
            svi = SVI(model.model, model.guide, optimizer, loss=elbo)

            # Early stopping monitor
            early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode='min')

            for epoch in range(num_epochs):
                loss = svi.step()
                print(f"Epoch {epoch}, ELBO loss: {loss:.4f}")

                if early_stopper(loss):
                    print("Early stopping triggered.")
                    break

            return model

        trained_model = train_model(gplvm, num_epochs=train_steps, lr=0.01, patience=10, min_delta=0.5)

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