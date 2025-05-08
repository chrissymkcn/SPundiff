from scipy.sparse import isspmatrix_lil
import squidpy as sq
import numpy as np  # always need it
import ot  # ot
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim
from joblib import Parallel, delayed


class undiff():
    DEFAULT_OT_PARAMS = {
        'mass': 1,
        'nb_dummies': 1,
        'to_one': True,
        'method': 'entropic_partial',
        'sinkhorn_method': 'sinkhorn',
        'scale_cost': 'by_max',
        'reg': 0.1,
        'reg_m_kl': (0.1, float('inf')),
        'reg_m_l2': (5, float('inf')),
        'reg_m_entropy': (0.1, float('inf')),
        'reg_m_lag': 0.1,
        'check_marginals': False,
        'div': 'kl',
        'log': False
    }
    
    def __init__(self, adata, n_jobs=1, leiden_range=[0.5, 1.1], metric='euclidean', 
                img_key='hires', optimizer='L-BFGS-B'):
        self.adata = adata
        self.raw_count = adata.X 
        # self.grid_coords = self.spatial_to_grid()
        # self.grid_coords = self.grid_coords - self.grid_coords.min(axis=0)
        self.img_key = img_key
        self.lib_id = list(self.adata.uns['spatial'].keys())[0]
        self.y1_max, self.y2_max = self.adata.uns['spatial'][self.lib_id]['images'][self.img_key].shape[:2]
        self.grid_coords = self.scale_spatial()
        # self.in_tiss_mask = self.get_in_tissue_field()  # 2d mask
        self.in_tiss_mask = self.adata.obs['in_tissue'].values
        self.gene_selected = []
        self.coord_cost = self.compute_ot_cost(metric=metric)
        self.leiden_range = leiden_range
        self.original_moranI = {}  # do not define this here as user may want to use different genes
        self.original_image_align_score = None
        self.original_silhouette = {}
        # Initialize
        self.n_jobs = n_jobs
        self.last_z = None
        self.objective_values = []  # Global list to store objective values
        self.optimizer = optimizer
        # Initial guess for parameters ## TO DO 
        # cost_weights, invalid_percs, reg
        self.params_init = None
        self.bounds = None


    def spatial_to_grid(self, orig_basis='spatial'):
        df = self.adata.obsm[orig_basis]
        df = pd.DataFrame(df, columns=['x','y'])
        x = df['x'].value_counts().index[0]
        y = df['y'].value_counts().index[1]
        re_x, re_y = df[df['y']==y]['x'].diff().abs().min()//2, df[df['x']==x]['y'].diff().abs().min()
        spatial = self.adata.obsm['spatial'].copy()
        spatial[:, 0] = (spatial[:, 0]/re_x).round()
        spatial[:, 1] = (spatial[:, 1]/re_y).round()
        return spatial

    def twod_grid(self, values, shape=None):
        """
        Convert a set of coordinates and values to a 2D grid.

        Parameters:
            values: Values at each coordinate
        """
        if shape is None:
            shape = (self.y1_max, self.y2_max)
        grid = np.zeros(shape)
        for i, (x, y) in enumerate(self.grid_coords):
            grid[int(x), int(y)] = values[i]
        return grid

    def get_spot_coords(self):  
        """
        Get the indices of the spots in the field, use if using a transformed coord system.
        """
        N_x, N_y = self.y1_max, self.y2_max
        # print('Dimension of the field:', N_x, N_y)
        # Create tensors using np.arange
        first = np.arange(N_x, dtype=np.float32)  # Tensor from 0 to N_x-1
        second = np.arange(N_y, dtype=np.float32)  # Tensor from 0 to N_y-1
        # Create a meshgrid for combinations
        X, Y = np.meshgrid(first, second, indexing='ij')
        # Combine X and Y to get all combinations
        combinations = np.stack([X, Y])
        # reshape to [N_x * N_y, 2]
        spot_coords = combinations.reshape(-1, 2)
        # print('Final coord shape', spot_coords.shape)
        return spot_coords

    def get_gene_conc_field(self, gene):
        """
        Get the total concentration field of a gene in the spatial transcriptomics dataset.

        Parameters:
            adata: AnnData object
            gene: Gene name
        """
        expression = self.adata[:, gene].X.toarray().flatten()
        expression_grid = self.twod_grid(expression)
        return expression_grid
    
    def get_total_conc_field(self, cnt_col='total_counts'):
        """
        Get the total concentration field of a gene in the spatial transcriptomics dataset.

        Parameters:
            adata: AnnData object
            cnt_col: Column name for total counts in the adata object
        """
        if cnt_col not in self.adata.obs.columns:
            raise ValueError(f'{cnt_col} not found in adata.obs')
        expression = self.adata.obs[cnt_col].values
        expression_grid = self.twod_grid(expression)
        return expression_grid

    def get_in_tissue_field(self, in_tiss_ind='in_tissue'):
        """
        Get the in tissue mask from the adata object.

        Parameters:
            adata: AnnData object
            in_tiss_ind: Key in the obsm attribute of adata object
        """
        expression = self.adata.obs[in_tiss_ind].values
        expression_grid = self.twod_grid(expression)
        return expression_grid
            
    def objective(self, params, genes): 
        """
        Objective function: Minimize the difference between deduced out-tissue values and 0.

        Parameters:
            params: Array of parameters (drift and diffusion constants).

        Returns:
            error: Sum of squared errors for out-tissue spots.
        """
        n_genes = len(genes)
        cost_weights = params[:n_genes]
        invalid_percs = params[n_genes:2*n_genes]
        regs = params[2*n_genes:]
        
        # compute ot for each gene
        resdicts = self.parallel_compute_ot(genes, self.coord_cost, cost_weights, invalid_percs, regs, method='entropic_partial')
        for gene in resdicts:
            self.update_cnt(gene, resdicts[gene])
        self.last_z = self.adata.X.copy()
        
        # loss from image alignment score, higher is better
        image_align_score = self.eval_image_align_score()
        image_align_score_loss = sum([self.original_image_align_score[res] - image_align_score[res] for res in image_align_score]) / len(image_align_score)
        
        # loss from out-tissue spot expression
        restored_out_tiss = self.adata[self.adata.obs['in_tissue']==0].X.data.sum()
        restored_out_tiss_err = restored_out_tiss / self.adata.X.sum()  # percentage of total expression that is out-tissue
        
    
        # calculate moranI difference, higher is better, so we use the negative value
        updated_moranI = self.eval_moranI(genes)
        moranI_loss = sum([self.original_moranI[gene] - updated_moranI[gene] for gene in updated_moranI]) / len(updated_moranI) # want this to be negative

        # compute cost
        self.adata_processing()
        
        # loss from silhouette scores, higher is better
        silhouette_scores = self.eval_silhouette(leiden_range=self.leiden_range, leiden_step=0.1)
        sil_score_loss = sum([self.original_silhouette[res] - silhouette_scores[res] for res in silhouette_scores]) / len(silhouette_scores) # want this to be positive        
                
        weight_sil = 1
        weight_restored_out_tiss = 1
        weight_moranI = 1
        weight_image_align_score = 1
        wsum = weight_sil + weight_restored_out_tiss + weight_moranI + weight_image_align_score
        weight_sil /= wsum
        weight_restored_out_tiss /= wsum
        weight_moranI /= wsum
        weight_image_align_score /= wsum
        error = weight_sil * sil_score_loss + weight_restored_out_tiss * restored_out_tiss_err + weight_moranI * moranI_loss + weight_image_align_score * image_align_score_loss
        
        # Store the error value
        self.objective_values.append(error)
        return error
    
    def reset(self):
        self.objective_values = []
        n_genes = len(self.gene_selected)
        self.params_init = np.concatenate([
            np.ones(n_genes),   # cost_weights
            np.zeros(n_genes),  # invalid_percs
            np.ones(n_genes)    # regs (now gene-specific)
        ])
        self.bounds = (
            [(0, 10)] * n_genes +    # cost_weights
            [(0, 1)] * n_genes +     # invalid_percs
            [(0, 100)] * n_genes     # regs
        )
        self.last_z = None
        self.adata.X = self.raw_count.copy()

    def em_algorithm(self, add_genes=[], max_iter=1000, log_file="em_progress.log"):
        self.reset()

        # Minimize the objective function
        self.gene_selected = self.gene_selection()
        self.original_image_align_score = self.eval_image_align_score()
        self.original_silhouette = self.eval_silhouette(leiden_step=0.1)
        genes = self.gene_selected + add_genes
        self.original_moranI = self.eval_moranI(add_genes)        

        n_genes = len(genes)
        self.params_init = np.concatenate([
            np.ones(n_genes),   # cost_weights
            np.zeros(n_genes),  # invalid_percs
            np.ones(n_genes)    # regs (now gene-specific)
        ])
        self.bounds = (
            [(0, 10)] * n_genes +    # cost_weights
            [(0, 1)] * n_genes +     # invalid_percs
            [(0, 100)] * n_genes     # regs
        )
        
        # Open log file in append mode
        with open(log_file, 'a') as f:
            f.write(f"Starting EM optimization with {n_genes} genes\n")
            f.write(f"Initial params: {self.params_init}\n")
            f.write("Iteration\tLoss\tParams\n")
            
            # Define callback function
            def callback(xk):
                current_loss = self.objective_values[-1] if self.objective_values else np.nan
                f.write(f"{len(self.objective_values)}\t{current_loss:.6f}\t{xk}\n")
                f.flush()  # Ensure immediate write to disk
            
            # Run optimization
            result = minimize(
                self.objective,
                self.params_init,
                args=(genes,),
                method=self.optimizer,
                bounds=self.bounds,
                options={'maxiter': max_iter, 'disp': True},
                callback=callback  # Add callback for logging
            )
            
            # Final log entry
            f.write(f"\nOptimization completed with final loss: {result.fun:.6f}\n")
            f.write(f"Optimal params: {result.x}\n\n")
        
        return result
    
    # adopt functions above to the class
    def prep(self, gene, to_one=True):
        X_g = self.adata[:, gene].X.toarray().flatten()
        # # if use only out and in tissue spots (do not assume in tissue transport)
        # out_tiss = X_g[self.adata.obs['in_tissue']==0]
        # in_tiss = X_g[self.adata.obs['in_tissue']==1]
        
        # if use all spots, also consider moving among in-tissue spots
        out_tiss = X_g.copy()
        in_tiss = X_g.copy()
        if to_one:
            out_tiss = out_tiss / out_tiss.sum()
            in_tiss = in_tiss / in_tiss.sum()
        return out_tiss, in_tiss
    
    def compute_ot_cost(self, metric='euclidean', **kwargs):
        coords = self.grid_coords
        # out_tiss_coords = coords[self.in_tiss_mask.flatten() == 0]
        # in_tiss_coords = coords[self.in_tiss_mask.flatten() == 1]
        if metric == 'euclidean':
            # cost = ot.dist(out_tiss_coords, in_tiss_coords, metric="euclidean", **kwargs)
            cost = ot.dist(coords, coords, metric="euclidean", **kwargs)
        elif metric == 'manhattan':
            # cost = ot.dist(out_tiss_coords, in_tiss_coords, metric="cityblock", **kwargs)
            cost = ot.dist(coords, coords, metric="cityblock", **kwargs)
        elif metric == 'chebyshev':
            # cost = ot.dist(out_tiss_coords, in_tiss_coords, metric="chebyshev", **kwargs)
            cost = ot.dist(coords, coords, metric="chebyshev", **kwargs)
        elif metric == 'minkowski':
            # cost = ot.dist(out_tiss_coords, in_tiss_coords, metric="minkowski", **kwargs)
            cost = ot.dist(coords, coords, metric="minkowski", **kwargs)
        else:
            raise ValueError('metric not recognized')
        return cost
    
    def scaling(self, C, scale=None):
        if scale == 'by_max':
            C /= C.max()
        elif scale == 'by_minmax':
            C = (C - C.min()) / (C.max() - C.min())
        elif scale == 'by_mean':
            C = (C - C.mean()) / C.std()
            C = C - C.min()
        elif scale == 'log':
            C = np.log1p(C)
        elif scale == 'sqrt':
            C = np.sqrt(C)
        return C

    def parallel_compute_ot(self, genes, base_cost, cost_weights, invalid_percs, regs,
                            batch_size=20, **kwargs):
        """
        Simple parallel OT computation with automatic CPU core limiting
        
        Args:
            genes: List of genes to process
        """
        # Calculate safe number of jobs (leave 1 core free)
        n_jobs = min(self.n_jobs, len(genes))
        
        results = {}
        for i in range(0, len(genes), batch_size):
            batch_genes = genes[i:i+batch_size]
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(self.compute_ot)(
                    gene,
                    base_cost=base_cost,
                    cost_weight=cost_weights[i],
                    inval_perc=invalid_percs[i],
                    reg=regs[i],
                    **kwargs
                )
                for i, gene in enumerate(batch_genes)
            )
            results.update(dict(zip(batch_genes, batch_results)))
        return results
    
    def compute_ot(self, gene, base_cost, cost_weight, inval_perc, reg,
                   **kwargs):
        # Merge defaults with provided kwargs
        params = {**self.DEFAULT_OT_PARAMS, **kwargs} 
        out_tiss, in_tiss = self.prep(gene, to_one=params['to_one'])
        C = base_cost * cost_weight
        target_thresh = np.percentile(in_tiss, inval_perc * 100)
        in_tiss[in_tiss < target_thresh] = 0
        in_tiss = in_tiss / in_tiss.sum()
        X_g = self.adata[:, gene].X.toarray().flatten()
        mass = min(out_tiss.sum(), in_tiss.sum())
        if params['scale_cost']:
            C = self.scaling(C, params['scale_cost'])
        if params['method'] is None or params['method']=='emd':
            ot_emd = ot.emd(out_tiss, in_tiss, C, check_marginals=params['check_marginals'])
        elif params['method'] == 'sinkhorn':
            if sinkhorn_method not in ['sinkhorn', 'sinkhorn_log', 'greenkhorn', 'sinkhorn_stabilized', 'sinkhorn_epsilon_scaling']:
                sinkhorn_method = 'sinkhorn'
            ot_emd = ot.sinkhorn(out_tiss, in_tiss, C, reg=reg, method=sinkhorn_method)
        elif params['method'] == 'greenkhorn':
            ot_emd = ot.bregman.greenkhorn(out_tiss, in_tiss, C, reg=reg)
        elif params['method'] == 'screenkhorn':
            ot_emd = ot.bregman.screenkhorn(out_tiss, in_tiss, C, reg=reg)
        elif params['method'] == 'sinkhorn_epsilon_scaling':
            ot_emd = ot.bregman.sinkhorn_epsilon_scaling(out_tiss, in_tiss, C, reg=reg)
        elif params['method'] == 'sinkhorn_unbalanced':
            if div not in ['kl', 'entropy']: div = 'entropy'
            reg_m = params['reg_m_kl'] if div == 'kl' else params['reg_m_entropy']
            ot_emd = ot.sinkhorn_unbalanced(out_tiss, in_tiss, C, reg, reg_m=reg_m,
                                            method=sinkhorn_method, reg_type=div)
        elif params['method'] == 'lbfgsb_unbalanced':
            if div == 'kl': reg_m = params['reg_m_kl']
            elif div == 'l2': reg_m = params['reg_m_l2']
            elif div == 'entropy': reg_m = params['reg_m_entropy']
            else: 
                div = 'kl'
                reg_m = params['reg_m_kl']
            ot_emd = ot.unbalanced.lbfgsb_unbalanced(out_tiss, in_tiss, C, 
                                                    reg=reg, reg_m=reg_m, 
                                                    reg_div=div, log=params['log'])
        elif params['method'] == 'mm_unbalanced':
            if div not in ['kl', 'l2']: div = 'kl'
            reg_m = params['reg_m_kl'] if div == 'kl' else params['reg_m_l2']
            ot_emd = ot.unbalanced.mm_unbalanced(out_tiss, in_tiss, C, reg_m=reg_m, div=div)
        elif params['method'] == 'partial':
            ot_emd = ot.partial.partial_wasserstein(out_tiss, in_tiss, C, 
                                                    m=mass, nb_dummies=params['nb_dummies'], check_marginals=params['check_marginals'])
        elif params['method'] == 'entropic_partial':
            ot_emd = ot.partial.entropic_partial_wasserstein(out_tiss, in_tiss, C, reg, m=mass)
        elif params['method'] == 'lag_partial':
            ot_emd = ot.partial.partial_wasserstein_lagrange(out_tiss, in_tiss, C, reg_m=params['reg_m_lag'], nb_dummies=params['nb_dummies'])
        elif params['method'] == 'smooth_ot_dual':
            if div not in ['kl', 'l2']: div = 'l2'
            ot_emd = ot.smooth.smooth_ot_dual(out_tiss, in_tiss, C, reg, reg_type=div)
        elif params['method'] == 'smooth_ot_semi_dual':
            if div not in ['kl', 'l2']: div = 'l2'
            ot_emd = ot.smooth.smooth_ot_semi_dual(out_tiss, in_tiss, C, reg, reg_type=div)
        else:
            raise ValueError('method not recognized')
        #### TO BE CONTINUED
        to_target = ot_emd.sum(axis=0)  # Sum over axis 0 (rows) gives mass transported to target spots
        from_source = ot_emd.sum(axis=1)  # Sum over axis 1 (columns) gives mass transported from source spots
        if params['to_one']:
            Xsum = X_g.sum()
            in_tiss = to_target * Xsum  # Original + deduced
            to_return = {'in': in_tiss,
                        'ot': ot_emd}
        else:
            to_return = {'in': in_tiss,
                        'ot': ot_emd}
        return to_return

    def gene_selection(self, out_tiss_perc=0.5):
        self.adata.var['MT'] = self.adata.var_names.str.startswith("MT-")
        self.adata.var['mt'] = self.adata.var_names.str.startswith("mt-")
        if any(self.adata.var['MT']) or any(self.adata.var['mt']):
            self.adata.var['mt'] = self.adata.var['MT'] | self.adata.var['mt']
            sc.pp.calculate_qc_metrics(self.adata, qc_vars=["mt"], inplace=True)
        else:
            sc.pp.calculate_qc_metrics(self.adata, inplace=True)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.highly_variable_genes(self.adata, flavor="seurat", n_top_genes=2000)
        out = self.adata[self.adata.obs['in_tissue']==0]
        out.var['total_counts'] = out.X.toarray().sum(axis=0)
        for i in range(500, len(out.var_names), 200):
            union_genes = out.var.sort_values('total_counts', ascending=False).index[:i].values
            perc_out = out.var.loc[union_genes,'total_counts'].sum()/out.var['total_counts'].sum()
            if perc_out > out_tiss_perc:
                break
        all_hvgs = self.adata.var_names[self.adata.var['highly_variable']].values
        union_genes = np.concatenate([union_genes, all_hvgs])
        union_genes = union_genes.tolist()
        self.adata.X = self.raw_count.copy()
        return union_genes

    def update_cnt(self, gene, resdict):
        # Convert to LIL format for efficient modification
        if not isspmatrix_lil(self.adata.X):
            self.adata.X = self.adata.X.tolil()
        in_filt = self.adata.obs['in_tissue'] == 1
        out_filt = self.adata.obs['in_tissue'] == 0
        gene_idx = np.where(self.adata.var_names == gene)[0][0]
        if resdict['in'].shape[0] == self.adata.n_obs:
            self.adata.X[:, gene_idx] = resdict['in']
        else:
            self.adata.X[in_filt, gene_idx] = resdict['in']
            self.adata.X[out_filt, gene_idx] = resdict['out']
        self.adata.X = self.adata.X.tocsr()

    def adata_processing(self):
        # check if the data is integer
        if self.adata.X.data.dtype in [np.int32, np.int64, np.uint32, np.uint64, int]:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, flavor="seurat", n_top_genes=2000)
        sc.pp.scale(self.adata, max_value=10)
        sc.pp.pca(self.adata)
        sc.pp.neighbors(self.adata, n_pcs=30, n_neighbors=30)

    def eval_silhouette(self, leiden_step=0.1):
        self.adata_processing() if 'X_pca' not in self.adata.obsm.keys() else None
        X = self.adata.obsm['X_pca']
        leiden_range = self.leiden_range
        leiden_reses = np.arange(leiden_range[0], leiden_range[1], leiden_step)
        # round to 2 decimal places
        leiden_reses = np.round(leiden_reses, 2)
        sil_dict = {}
        for res in leiden_reses:
            sc.tl.leiden(self.adata, resolution=res, key_added=f'leiden_{res}')
            cluster = self.adata.obs[f'leiden_{res}']
            silhouette = silhouette_score(X, cluster)
            sil_dict[res] = silhouette
        return sil_dict

    def eval_moranI(self, add_genes=[]):
        genes = np.unique(self.gene_selected + add_genes)
        if 'spatial_connectivities' not in self.adata.obsp.keys():
            sq.gr.spatial_neighbors(self.adata)
        sq.gr.spatial_autocorr(
            self.adata,
            mode="moran",
            genes=genes,
            n_perms=100,
            n_jobs=1,
        )
        moranI = dict(zip(self.adata.uns['moranI'].index.values, self.adata.uns['moranI']['I']))
        return moranI
    
    def scale_spatial(self, lib_id=None):
        """
        Calculate SSIM-based improvement between original and corrected expression.
        """
        lib_id = self.lib_id if lib_id is None else lib_id
        return self.adata.obsm['spatial'] * self.adata.uns['spatial'][lib_id]['scalefactors'][f'tissue_{self.img_key}_scalef']

    def eval_image_align_score(self, lib_id=None):
        """
        Calculate SSIM-based improvement between original and corrected expression.
        """
        lib_id = self.lib_id if lib_id is None else lib_id
        img = self.adata.uns['spatial'][lib_id]['images'][self.img_key]
        # convert to grayscale 
        if img.shape[2] == 3:
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        # Reshape expression to match image dimensions
        new_exp_field = self.twod_grid(self.adata.X.toarray().sum(axis=1))
        # Calculate SSIM for original and corrected expression
        ssim_score = ssim(img, new_exp_field, data_range=new_exp_field.max() - new_exp_field.min())
        return ssim_score

