import torch
from torch.nn import Parameter
from functools import partial
import pyro.contrib.gp as gp

from diffusion_sim import PhysicsInformedDiffusionModel, PhysicsInformedSparseGP
from simple_pisei import PhysicsInformedSpatialInverter
import pyro
import torch.nn.functional as F

from base import base
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scanpy as sc 
import squidpy as sq
import pickle as pkl
import os

class undiff(base):
    def __init__(self, adata, n_neighs=15):
        super().__init__(adata, n_neighs=n_neighs)
        self.image_features = None
        self.ecm_scores = None
        self.cell_count = None

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
                model = undiff.clustering_model_init(algo, i)
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
            top_ncl, algo = undiff.find_optimal_ncls(data_pca, methods, max_clusters=max_clusters, starting_clusters=starting_clusters)
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
        model = undiff.clustering_model_init(actual_algo, optimal_clusters)
        cluster_labels = model.fit_predict(data)
        return {
            'cluster_labels': cluster_labels,
            'pca_silhouette_df': pca_silhouette_df,
            'optimal_clusters': optimal_clusters,
            # 'pca': pca,
        }
    
    def run_initialization(self, qts_prior=0.8, derivative_threshold=0.02, n_genes=None, add_genes=[]):
        self.prep_genes_params(add_genes=add_genes, first_n_genes=n_genes)
        self.set_states(qts_prior=qts_prior, derivative_threshold=derivative_threshold)
        self.compute_X_init({
            'invalid_qts': self.invalid_qts,
        })
        
    def gene_initialization(self, gene_expr_out, gene_expr_in, out_sum, in_sum):
        """
        Distribute mass from out-of-tissue spots to in-tissue spots based on:
        1. Spatial proximity (from coord_cost matrix)
        2. Likelihood weights (from gene_expr_in values)
        
        Args:
            gene_expr_out: [n_spots] - source distribution (out-of-tissue)
            gene_expr_in: [n_spots] - target distribution weights (in-tissue)  
            out_sum: scalar - total mass to distribute from out-of-tissue
            in_sum: scalar - total mass already in-tissue
            
        Returns:
            transported_in: [n_spots] - adapted transport for this gene
        """
        # Normalize inputs
        p = gene_expr_out / (gene_expr_out.sum() + 1e-8)  # Source distribution (normalized)
        q = gene_expr_in / (gene_expr_in.sum() + 1e-8)    # Target weights (normalized)
        
        # Get spatial cost matrix (precomputed in base class)
        C = self.coord_cost  # [n_spots, n_spots]
        
        # Create combined cost matrix incorporating both spatial distance and target weights
        # We want spots that are both close and have high expression to receive more mass
        # So we take the inverse of distance (closer = higher value) and multiply by target weights
        with torch.no_grad():
            # Avoid division by zero for spatial cost
            spatial_affinity = 1.0 / (1.0 + C)  # [n_spots, n_spots], closer spots have higher affinity
            
            # Combine spatial affinity with target weights
            combined_cost = spatial_affinity * q.unsqueeze(0)  # [n_spots, n_spots]
            
            # Normalize rows to create a probability distribution for each source spot
            transport_plan = combined_cost / (combined_cost.sum(dim=1, keepdim=True) + 1e-8)
        
        # Distribute the source mass according to the transport plan
        transported_mass = torch.matmul(p.unsqueeze(0), transport_plan).squeeze(0)  # [n_spots]
        
        # Preserve original scale
        transported_in = transported_mass * out_sum + gene_expr_in * in_sum
        
        return transported_in
    
    def compute_X_init(self, params):
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
            transported_in = self.gene_initialization(g_out, g_in, g_outsum, g_insum)
            # transported_in = self.compute_ot(g_out, g_in, g_outsum, g_insum, regs[i])
            res.append(transported_in)
        self.X_init = torch.stack(res, dim=1)
        self.round_counts_to_integers()
        
    def get_initialized_embeddings(self, n_genes=100, qts_prior=0.8, derivative_threshold=0.02, clustering=False):
        self.run_initialization(n_genes=n_genes, qts_prior=qts_prior, derivative_threshold=derivative_threshold,)
        if clustering:
            res = undiff.auto_cluster(self.X_init.detach().cpu().numpy().T, starting_clusters=3, max_clusters=None)  # pass in transposed X_init to have genes as samples
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
                separated_chunk = self.X_init[:, cluster_genes_idx]
                print(type(separated_chunk))
                # twod_genes = []
                # for j in range(len(cluster_genes)):
                    # gene_twod = self.shifted_grid_embedding(separated_chunk[:, j])
                #     gene_twod = gene_twod / gene_twod.sum()  # Normalize to sum to 1
                #     twod_genes.append(torch.tensor(gene_twod))
                # separated_chunk = torch.stack(twod_genes, dim=0)
                gene_groups[i] = separated_chunk.clone()  # do not embed here, just use expression vector directly
                z_prior[i] = torch.mean(separated_chunk, dim=1)
            self.X_init = torch.cat([ts for ts in gene_groups.values()], dim=0).T
        return self.X_init, self.y_init
    
    def define_PISGP(self, X=None, y=None, 
                    n_genes=100, # Reduced from 1000
                    qts_prior=0.8, 
                    derivative_threshold=0.02,
                    **kwargs):
        """
        Memory-optimized version of define_PISGP
        """
        if X == None:
            self.get_initialized_embeddings(n_genes=n_genes, qts_prior=qts_prior, 
                                            derivative_threshold=derivative_threshold, clustering=False)
            X = self.X_init
            y = self.y_init
            
        X = X.T  # shape [n_genes, n_spots]
        
        # Convert sparse matrix to dense if needed
        if issparse(self.spatial_con):
            print("Converting sparse connectivity matrix to dense...")
            neighbor_graph = self.spatial_con.toarray()
        else:
            neighbor_graph = self.spatial_con
        
        # Convert to torch tensors with float32 dtype to save memory
        self.spatial_con = torch.tensor(neighbor_graph, dtype=torch.float32)
        self.ttl_cnts = torch.tensor(self.adata.obs['total_counts'].values, dtype=torch.float32)
        
        # Initialize prior mean
        X_prior = self.y_init.detach().clone() * self.in_tiss_mask.unsqueeze(1).float()
        X_prior = (X_prior / (X_prior.sum(dim=0, keepdim=True) + 1e-8)) * self.y_init.sum(dim=0, keepdim=True)
        X_prior = X_prior.T  # shape [n_genes, n_spots]
        # Default parameters optimized for memory efficiency
        params = {
            'D_out': 0.1,
            'D_in': 0.02,
            'noise': 0.1,
            'approx': 'VFE',
            'jitter': 1e-6,
            'diffusion_steps': 6,  # Reduced from 10
        }
        
        # Update default parameters with any additional kwargs
        params.update(kwargs)
        
        # Convert all tensors to float32
        X = X.detach().to(torch.float32)
        y = y.detach().to(torch.float32)
        coords = self.coords.to(torch.float32)
        in_tiss_mask = self.in_tiss_mask.detach().to(torch.float32)
        ttl_cnts = self.ttl_cnts.detach().to(torch.float32)
        X_prior = X_prior.to(torch.float32)
        
        self.model = PhysicsInformedSparseGP(
            X = X,  # shape [n_genes, n_spots]
            y = y,  # shape [n_spots, n_genes]
            X_prior = X_prior,  # shape [n_genes, n_spots]
            coords = coords,
            in_tiss_mask = in_tiss_mask,
            ttl_cnts = ttl_cnts,
            neighbors = self.spatial_con,
            kernel = None,
            **params
        )
        self.model_name = 'PISGP'
        return self.model
        
    def define_PID(self, X=None, y=None, 
                    n_genes=1000, 
                    qts_prior=0.8, 
                    derivative_threshold=0.02,
                    **kwargs):
        """
        Run Sparse GP Regression on the gene expression data.
        """
        if X == None:
            self.get_initialized_embeddings(n_genes=n_genes, qts_prior=qts_prior, 
                                            derivative_threshold=derivative_threshold, clustering=False)
            X = self.X_init
            y = self.y_init
            
        X = X.T  # shape [n_genes, n_spots]
        y = y.T  # shape [n_genes, n_spots]
        
        # Convert sparse matrix to dense if needed
        if issparse(self.spatial_con):
            print("Converting sparse connectivity matrix to dense...")
            neighbor_graph = self.spatial_con.toarray()
        else:
            neighbor_graph = self.spatial_con
        
        # Convert to torch tensors with float32 dtype to save memory
        self.spatial_con = torch.tensor(neighbor_graph, dtype=torch.float32)
        self.ttl_cnts = torch.tensor(self.adata.obs['total_counts'].values, dtype=torch.float32)
        
        # Initialize prior mean
        X_prior = self.y_init.detach().clone() * self.in_tiss_mask.unsqueeze(-1).float()  # shape [n_spots, n_genes]
        X_prior = (X_prior / (X_prior.sum(dim=0, keepdim=True) + 1e-8)) * self.y_init.sum(dim=0, keepdim=True)  # keep gene level total sums the same
        X_prior = X_prior.T  # shape [n_genes, n_spots]
                
        # Convert all tensors to float32
        X = X.detach().to(torch.float32)
        y = y.detach().to(torch.float32)
        coords = self.coords.to(torch.float32)
        in_tiss_mask = self.in_tiss_mask.detach().to(torch.float32)
        ttl_cnts = self.ttl_cnts.detach().to(torch.float32)
        X_prior = X_prior.to(torch.float32)
        cluster_labels = self.cluster_labels
        # Initialize model
        default_params = {
            'D_out': 0.1,  # Initial diffusion coefficient
            'D_in': 0.02,
            'diffusion_steps': 6,  # Number of diffusion steps
            'alpha': 0.8,
            'beta': 0.2,
        }
        # Update default parameters with any additional kwargs
        default_params.update(kwargs)
        self.kwargs = default_params
        self.model = PhysicsInformedDiffusionModel(
            X = X,
            y = y,
            X_prior = X_prior,
            coords = coords,
            in_tiss_mask = in_tiss_mask,
            ttl_cnts = ttl_cnts,
            neighbors = self.spatial_con,
            cluster_labels=cluster_labels,
            **default_params
        )
        self.model_name = 'PID'
        return self.model
        
    def train(self, learning_rate=0.01, n_epochs=1000):
        """Train the model using SVI"""
        pyro.clear_param_store()
        
        # Initialize optimizer with learning rate scheduling
        scheduler = pyro.optim.ExponentialLR({
            'optimizer': torch.optim.Adam,
            'optim_args': {'lr': learning_rate},
            'gamma': 0.995
        })
        
        # Setup SVI
        svi = pyro.infer.SVI(
            model = self.model.model,
            guide = self.model.guide,
            optim = scheduler,
            loss = pyro.infer.Trace_ELBO()
        )
        
        # Training loop
        losses = []
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for step in range(n_epochs):
            loss = svi.step()
            losses.append(loss)
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at step {step}")
                break
            
            # Print progress
            if step % 3 == 0:
                print(f"Step {step}: Loss = {loss:.4f}")
        
        return losses
    
    def save(self, path: str):
        """Save the trained model parameters"""
        pyro.get_param_store().save(f'{path}/model.pt')
        print(f"Model parameters saved to {path}/model.pt")
        
        self.adata.uns['undiff'] = {
            'gene_selected': self.gene_selected,
            'X_init': self.X_init.detach().cpu().numpy(),  # [n_spots, n_genes]
            'y_init': self.y_init.detach().cpu().numpy(),  # [n_spots, n_genes]
            'invalid_qts': self.invalid_qts.detach().cpu().numpy(),
            'image_features': self.image_features.detach().cpu().numpy() if self.image_features is not None else None,
            'ecm_scores': self.ecm_scores.detach().cpu().numpy() if self.ecm_scores is not None else None,
            'cell_count': self.cell_count.detach().cpu().numpy() if self.cell_count is not None else None,
            'model_name': self.model_name,
            'cluster_labels': self.cluster_labels if hasattr(self, 'cluster_labels') else None,
            'kwargs': self.kwargs,
        }
        self.adata.write_h5ad(f'{path}/adata.h5ad')
        
    @staticmethod
    def load(path: str):
        """Load a trained model from saved parameters"""
        # Load model parameters
        params = torch.load(f"{path}/model.pt", weights_only=False)
        pyro.get_param_store().set_state(params)
        
        adata = sc.read_h5ad(f'{path}/adata.h5ad')
        # Reconstruct the undiff model
        restorer = undiff(adata)
        restorer.gene_selected = adata.uns['undiff']['gene_selected']
        restorer.y_init = torch.tensor(adata.uns['undiff']['y_init'], dtype=torch.float32)
        restorer.X_init = torch.tensor(adata.uns['undiff']['X_init'], dtype=torch.float32)
        restorer.invalid_qts = adata.uns['undiff']['invalid_qts']
        restorer.cluster_labels = adata.uns['undiff']['cluster_labels'] if 'cluster_labels' in adata.uns['undiff'] else None
        restorer.kwargs = adata.uns['undiff']['kwargs']
        restorer.ttl_cnts = torch.tensor(restorer.adata.obs['total_counts'].values, dtype=torch.float32)
        restorer.model_name = adata.uns['undiff']['model_name']
        
        neighborgraph = torch.tensor(restorer.spatial_con.toarray())
        
        if adata.uns['undiff']['model_name'] == 'PISEI':
            restorer.image_features = torch.tensor(adata.uns['undiff']['image_features'], dtype=torch.float32) if adata.uns['undiff']['image_features'] is not None else None
            restorer.ecm_scores = torch.tensor(adata.uns['undiff']['ecm_scores'], dtype=torch.float32) if adata.uns['undiff']['ecm_scores'] is not None else None
            restorer.cell_count = torch.tensor(adata.uns['undiff']['cell_count'], dtype=torch.float32) if adata.uns['undiff']['cell_count'] is not None else None
            pca_emb = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32) if 'X_pca' in adata.obsm else None
            
        if adata.uns['undiff']['model_name'] in ['PID', 'PISGP']:
            X_prior = restorer.y_init * restorer.in_tiss_mask.unsqueeze(-1).float()
            X_prior = (X_prior / (X_prior.sum(dim=0, keepdim=True) + 1e-8)) * restorer.y_init.sum(dim=0, keepdim=True)
        
        if restorer.model_name == 'PID':
            restorer.model = PhysicsInformedDiffusionModel(
                X=restorer.X_init.T,  # Transpose to [n_genes, n_spots]
                y=restorer.y_init.T,  # Transpose to [n_genes, n_spots]
                X_prior=X_prior.T,  # Transpose to [n_genes, n_spots]
                coords=restorer.coords,
                in_tiss_mask=restorer.in_tiss_mask,
                ttl_cnts=restorer.ttl_cnts,
                neighbors=neighborgraph,
                **restorer.kwargs,  # Unpack additional parameters
            )
        elif restorer.model_name == 'PISGP':
            restorer.model = PhysicsInformedSparseGP(
                X=restorer.X_init.T,  # Transpose to [n_genes, n_spots]
                y=restorer.y_init,  # [n_spots, n_genes]
                X_prior=X_prior.T,  # Transpose to [n_genes, n_spots]
                coords=restorer.coords,
                in_tiss_mask=restorer.in_tiss_mask,
                ttl_cnts=restorer.ttl_cnts,
                neighbors=neighborgraph,
                kernel=None,  # Use default kernel
                **restorer.kwargs  # Unpack additional parameters
            )
        elif restorer.model_name == 'PISEI':
            restorer.model = PhysicsInformedSpatialInverter(
                X=restorer.X_init,  # Transpose to [n_genes, n_spots]
                y=restorer.y_init,  # [n_spots, n_genes]
                coords=restorer.coords,
                spatial_adjacency=neighborgraph,
                pca_emb=pca_emb,
                image_features=restorer.image_features,
                ecm_scores=restorer.ecm_scores,
                cell_count=restorer.cell_count,
                **restorer.kwargs  # Unpack additional parameters
                )
            
        return restorer
    
    def get_restored_counts(self, num_samples: int = 100) -> np.ndarray:
        """Get restored counts by sampling from posterior"""
        restored_samples = []
        
        # Sample from posterior
        for _ in range(num_samples):
            with torch.no_grad():
                restored = self.model.guide()
                restored_samples.append(restored)
        
        # Average samples
        restored_mean = torch.stack(restored_samples).mean(0)
        # restored_mean = (restored_mean / restored_mean.sum(dim=0, keepdim=True)) * self.X_init.sum(dim=0, keepdim=True)  # Rescale to match original total counts
        return restored_mean.numpy()
    
    def restore_adata(self, copy: bool = False, diffusion_steps: int = 6, num_samples=1000, sampling=False, param_key=None) -> sc.AnnData:
        """Create new AnnData object with restored counts"""
        if copy:
            new_adata = self.adata.copy()
        else:
            new_adata = self.adata
        if sampling:
            restored_counts = self.get_restored_counts(diffusion_steps=diffusion_steps, num_samples=num_samples)
        else:
            if param_key is not None:
                restored_counts = pyro.get_param_store().get_param(param_key)
                restored_counts = restored_counts.detach().cpu().numpy()
            else:
                print('Available parameters in param store:')
                print(pyro.get_param_store().get_all_param_names())
                raise ValueError("param_key must be provided if sampling is False")
        if restored_counts.shape[0] != self.adata.n_obs and restored_counts.shape[1] == self.adata.n_obs:
            restored_counts = restored_counts.T
        if restored_counts.shape[1] == self.adata.n_vars:
            new_adata.layers['restored'] = restored_counts
        else:
            new_adata = new_adata[:, self.gene_selected].copy()
            new_adata.layers['restored'] = restored_counts
        
        return new_adata if copy else None

    def prepare_tissue_features(self, seg_method='watershed', seg_dir=None):
        """
        Calculate image features and ECM/cytoskeletal scores for each spot
        """
        print("Extracting image features...")
        self._extract_image_features()
        
        print("Calculating ECM and cytoskeletal module scores...")
        self._calculate_module_scores()
        
        print("Performing cell segmentation...")
        self._perform_cell_segmentation(method=seg_method, savedir=seg_dir)
                
        return {
            'image_features': self.image_features,
            'ecm_scores': self.ecm_scores,
            'cell_count': self.cell_count,
        }
        
    def _extract_image_features(self, crop_size=1.5, scale=1.0):
        """Extract image features using squidpy"""
        img = self.adata.uns['spatial'][self.lib_id]['images'][self.img_key]
        img = sq.im.ImageContainer(img, layer="image")
        if self.adata.obsm['spatial'].max() > max(img.shape):
            self.adata.obsm['spatial'] = self.adata.obsm['spatial'] * self.adata.uns['spatial'][self.lib_id]['scalefactors'][f'tissue_{self.img_key}_scalef']
        # Calculate features with scaling and larger context
        sq.im.calculate_image_features(
            self.adata,
            img,
            features=["summary", "texture", "histogram"],
            key_added="image_features",
            mask_circle=True,
            spot_scale=crop_size,
            scale=scale,
            show_progress_bar=False,
        )
        
        # Extract feature matrix
        self.image_features = torch.tensor(self.adata.obsm['image_features'].values, dtype=torch.float32)        
        # Normalize features
        self.image_features = (self.image_features - self.image_features.mean(dim=0)) / (
            self.image_features.std(dim=0) + 1e-8
        )
        
        print(f"Extracted {self.image_features.shape[1]} image features per spot in .image_features")
        
    def _calculate_module_scores(self):
        """Calculate ECM and cytoskeletal module scores from gene expression"""
        
        # Define gene sets (you can customize these based on your organism)
        ecm_genes = [
            'COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL4A2', 'COL6A1', 'COL6A2',
            'FN1', 'LAMB1', 'LAMB2', 'LAMC1', 'VTN', 'DCN', 'LUM', 'BGN',
            'MMP2', 'MMP9', 'MMP14', 'TIMP1', 'TIMP2'
        ]
        
        cytoskeletal_genes = [
            'ACTB', 'ACTG1', 'ACTA2', 'TUBB', 'TUBB3', 'TUBA1A', 'TUBA1B',
            'VIM', 'KRT8', 'KRT18', 'KRT19', 'CTNNB1', 'CDH1', 'CDH2',
            'MYH9', 'MYH10', 'MYL9', 'CFL1', 'ARPC2'
        ]
        
        # Filter genes present in the dataset
        ecm_genes_present = [g for g in ecm_genes if g in self.adata.var_names]
        cyto_genes_present = [g for g in cytoskeletal_genes if g in self.adata.var_names]
        
        print(f"Found {len(ecm_genes_present)} ECM genes and {len(cyto_genes_present)} cytoskeletal genes")
        if len(ecm_genes_present) == 0 or len(cyto_genes_present) == 0:
            raise ValueError("No ECM or cytoskeletal genes found in the dataset. Please check your gene names.")
        
        # Calculate module scores using scanpy
        if len(ecm_genes_present) > 0:
            sc.tl.score_genes(self.adata, ecm_genes_present, score_name='ecm_score')
        else:
            self.adata.obs['ecm_score'] = 0.0
            
        if len(cyto_genes_present) > 0:
            sc.tl.score_genes(self.adata, cyto_genes_present, score_name='cytoskeletal_score')
        else:
            self.adata.obs['cytoskeletal_score'] = 0.0
        
        # Combine scores
        ecm_cyto_scores = np.column_stack([
            self.adata.obs['ecm_score'].values,
            self.adata.obs['cytoskeletal_score'].values
        ])
        
        self.ecm_scores = torch.tensor(ecm_cyto_scores, dtype=torch.float32)
        
        # Normalize scores
        self.ecm_scores = (self.ecm_scores - self.ecm_scores.mean(dim=0)) / (
            self.ecm_scores.std(dim=0) + 1e-8
        )
        print(f"Calculated ECM and cytoskeletal scores for {self.ecm_scores.shape[0]} spots in .ecm_scores")
    
    def _perform_cell_segmentation(self, method='watershed', savedir=None):
        """Perform cell segmentation to identify cell-containing spots"""
        # Get the image
        img = self.adata.uns['spatial'][self.lib_id]['images'][self.img_key]
        img = sq.im.ImageContainer(img, layer="image")
        # Use squidpy for segmentation
        if method == 'cellpose':
            from cellpose import models
            def cellpose_he(img, min_size=15, flow_threshold=0.4, channel_cellpose=0):
                model = models.CellposeModel(model_type="nuclei")
                res, _, _ = model.eval(
                    img,
                    channels=[channel_cellpose, 0],
                    diameter=None,
                    min_size=min_size,
                    invert=True,
                    flow_threshold=flow_threshold,
                )
                return res
            sq.im.segment(
                img=img,
                layer="image",
                channel=None,
                method=cellpose_he,
                flow_threshold=0.4,
                channel_cellpose=0,
                min_size=10,
                layer_added="segmentation_cellpose",
            )
            layer_added = "segmentation_cellpose"
        elif method == 'stardist':
            from csbdeep.utils import normalize
            from stardist.models import StarDist2D
            def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None):
                # axis_norm = (0,1)   # normalize channels independently
                axis_norm = (0, 1, 2)  # normalize channels jointly
                # Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.
                # this is the default normalizer noted in StarDist examples.
                img = normalize(img, 1, 99.8, axis=axis_norm)
                model = StarDist2D.from_pretrained("2D_versatile_he")
                labels, _ = model.predict_instances(
                    img, nms_thresh=nms_thresh, prob_thresh=prob_thresh
                )
                return labels
            sq.im.segment(
                img=img,
                layer="image",
                channel=None,
                method=stardist_2D_versatile_he,
                layer_added="segmentation_stardist",
                prob_thresh=0.3,
                nms_thresh=None,
            )
            layer_added = "segmentation_stardist"
        else:
            from skimage import exposure, filters, morphology, measure
            sq.im.process(img, layer="image", method="gray")
            sq.im.process(img, layer="image_gray", method=exposure.equalize_adapthist, layer_added="image_gray", apply_kwargs={"clip_limit": 0.02})
            sq.im.process(img, layer='image_gray', method=filters.threshold_local, apply_kwargs={"block_size": 21, "offset": 10}, layer_added='image_gray')
            sq.im.segment(img, layer="image_gray", method="watershed", geq=False, layer_added="segmentation_watershed")
            sq.im.process(img, layer='segmentation_watershed', method=morphology.remove_small_objects, apply_kwargs={"min_size": 10}, layer_added='segmentation_watershed')
            layer_added = "segmentation_watershed"
        if self.adata.obsm['spatial'].max() > max(img.shape):
            self.adata.obsm['spatial'] = self.adata.obsm['spatial'] * self.adata.uns['spatial'][self.lib_id]['scalefactors'][f'tissue_{self.img_key}_scalef']
        sq.im.calculate_image_features(
            self.adata,
            img,
            layer=layer_added,  # layer with segmentation results
            features="segmentation",
            key_added=f"segmentation_features",
            features_kwargs={
                "segmentation": {
                    "label_layer": layer_added,  # layer with segmentation results
                    "props": ["label", "area"],  # mean intensity is calculated for each specified channel
                    # "props": ["label", "area", "mean_intensity"],  # mean intensity is calculated for each specified channel
                    # label is the number of segments (i.e. nuclei) identified
                    # "channels": [0],  # not necessary to specify channels, but can calculate mean intensity for specific channels
                }
            },
            mask_circle=True,
        )
        if savedir is not None:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            img.save(f'{savedir}/{layer_added}')
            img.show(layer_added, channelwise=True, save=f'{savedir}/{layer_added}/seg.png')
            plt.close()
            
            ## some other diagnostic plots
            # histogram of segmented cell counts
            plt.hist(self.adata.obsm['segmentation_features']['segmentation_label'].values, bins=50)
            quantiles = self.adata.obsm['segmentation_features']['segmentation_label'].quantile([0.95])
            plt.axvline(quantiles[0.95], color='orange', linestyle='--', label='95% quantile')
            plt.legend()
            plt.savefig(f'{savedir}/{layer_added}/segmentation_histogram.png')
            plt.close()
            
            # plot correlation between label and area
            plt.scatter(
                self.adata.obsm['segmentation_features']['segmentation_label'],
                self.adata.obsm['segmentation_features']['segmentation_area_mean'],
                alpha=0.5,
            )
            plt.xlabel("Segmentation label")
            plt.ylabel("Segmentation area mean")
            plt.title("Correlation between segmentation label and area mean")
            plt.savefig(f'{savedir}/{layer_added}/segmentation_label_area_correlation.png')
            plt.close()
            
        # clamp segmentation labels to below 98th quantile to avoid outliers
        seg_labels = self.adata.obsm['segmentation_features']['segmentation_label']
        quantile = np.quantile(seg_labels, 0.98)
        self.adata.obsm['segmentation_features']['segmentation_label'] = np.clip(seg_labels, 0, quantile)
        # for out-tissue spots, set segmentation label to 0
        self.adata.obsm['segmentation_features']['segmentation_label'][self.adata.obs['in_tissue'] == 0] = 0
        self.cell_count = torch.tensor(self.adata.obsm['segmentation_features']['segmentation_label'].values, dtype=torch.float32)
        print(f"Identified cell counts for {self.cell_count.shape[0]} spots in .cell_count")

    def define_PISEI(self, 
                    n_genes=100,
                    qts_prior=0.8,
                    derivative_threshold=0.03,
                    prepare_features=True,
                    **kwargs):
        """
        Define simplified PISEI model with adaptive features
        """
        seg_kwargs = {k: v for k, v in kwargs.items() if k.startswith('seg_')}
        if isinstance(prepare_features, bool) and prepare_features:
            self.prepare_tissue_features(**seg_kwargs)
        elif isinstance(prepare_features, str):
            feat_path = os.path.join(prepare_features, 'img_features.pkl')
            if not os.path.exists(feat_path):
                res = self.prepare_tissue_features(**seg_kwargs)
                res = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in res.items()}
                for k, v in res.items(): print(f"{k}: {v.dtype if isinstance(v, np.ndarray) else v}")
                with open(feat_path, 'wb') as f:
                    pkl.dump(res, f)
            elif os.path.exists(prepare_features):
                with open(feat_path, 'rb') as f:
                    res = pkl.load(f)
                self.image_features = torch.tensor(res['image_features'], dtype=torch.float32)
                self.ecm_scores = torch.tensor(res['ecm_scores'], dtype=torch.float32)
                self.cell_count = torch.tensor(res['cell_count'], dtype=torch.float32)
                
        for feats in ['image_features', 'ecm_scores', 'cell_count']:
            if getattr(self, feats) is None:
                Warning(f"{feats} is not defined. Not using it in the model.")
            else:
                print(f"Using {feats} with shape {getattr(self, feats).shape} and dtype {getattr(self, feats).dtype}")
        # Initialize expression data
        if not hasattr(self, 'X_init') or self.X_init is None:
            self.get_initialized_embeddings(
                n_genes=n_genes,
                qts_prior=qts_prior,
                derivative_threshold=derivative_threshold,
                clustering=False
            )
        
        neighbor_graph = self.spatial_con.toarray() if issparse(self.spatial_con) else self.spatial_con
        spatial_adjacency = torch.tensor(neighbor_graph, dtype=torch.float32)
        if 'X_pca' not in self.adata.obsm:
            sc.pp.scale(self.adata, max_value=10)
            # note highly variable genes should have already been computed when selecting genes
            sc.pp.pca(self.adata, n_comps=50, svd_solver='arpack', mask_var="highly_variable", key_added='X_pca')
        pca_emb = torch.tensor(self.adata.obsm['X_pca'].copy(), dtype=torch.float32)
    
        default_params = {
            'diffusion_steps': 3,  # Number of diffusion steps
        }
        # Update default parameters with any additional kwargs
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('seg_')}
        default_params.update(kwargs)
        self.kwargs = default_params

        # Create model
        self.model = PhysicsInformedSpatialInverter(
            X=self.X_init,  # shape [n_spots, n_genes]
            y=self.y_init,  # shape [n_spots, n_genes]
            coords=self.coords,
            spatial_adjacency=spatial_adjacency,  # shape [n_spots, n_spots]
            pca_emb=pca_emb,  # shape [n_spots, n_spots]
            image_features=self.image_features,  # shape [n_spots, n_features]
            ecm_scores=self.ecm_scores,  # shape [n_spots, 2] (ECM and cytoskeletal scores)
            cell_mask=self.cell_count,  # shape [n_spots] (cell counts)
            **default_params
        )
        self.model_name = 'PISEI'
        return self.model