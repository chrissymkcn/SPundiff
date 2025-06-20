import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from kneed import KneeLocator
from diffusion_sim import coords_to_filled_grid, calculate_domain_parameters
import squidpy as sq
import torch
from scipy.sparse import issparse
import numpy as np  # always need it
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity

class base():
    def __init__(self, adata, n_neighs=15):
        self.adata = adata.copy()
        self.var_names = adata.var_names
        self.y_init = None
        self.X_init = None
        self.raw_count = adata.X.copy()  # keep a sparse copy of the count
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs)
        self.spatial_con = adata.obsp['spatial_connectivities']
        self.spatial_dist = adata.obsp['spatial_distances']
        self.img_key = 'hires'
        self.lib_id = list(self.adata.uns['spatial'].keys())[0]
        self.coords = self.scale_spatial()
        self.coords = self.coords - self.coords.min(dim=0)[0]
        self.coord_cost = self.compute_distance_cost(metric='euclidean', scaling=True)  # precompute the distance cost matrix
        self.y1_max, self.y2_max = self.adata.uns['spatial'][self.lib_id]['images'][self.img_key].shape[:2]
        self.in_tiss_mask = torch.tensor(adata.obs['in_tissue'].values, dtype=torch.float32, requires_grad=False)
        self.gene_selected = []


    def scale_spatial(self, lib_id=None, coords=None):
        if coords is None:
            coords = self.adata.obsm['spatial']
        lib_id = list(self.adata.uns['spatial'].keys())[0] if lib_id is None else lib_id
        img_key = [k for k in self.adata.uns['spatial'][lib_id]['scalefactors'].keys() if 'hires' in k]
        if len(img_key) > 0:
            img_key = img_key[0]
        else:
            img_key = [k for k in self.adata.uns['spatial'][lib_id]['scalefactors'].keys() if 'lowres' in k][0]
        scaled_spatial = coords * self.adata.uns['spatial'][lib_id]['scalefactors'][img_key]
        scaled_spatial = np.round(scaled_spatial)
        scaled_spatial = torch.tensor(scaled_spatial, dtype=torch.float32, requires_grad=False)
        return scaled_spatial
    
    
    def calc_diff(self, df):
        diffs_x = []
        diffs_y = []
        for i in df['y'].unique():
            diffs = df[df['y'] == i]['x'].sort_values().diff().unique()
            diffs_x += [x for x in diffs if not np.isnan(x) and x not in diffs_x and x > 0]
        for i in df['x'].unique():
            diffs = df[df['x'] == i]['y'].sort_values().diff().unique()
            diffs_y += [y for y in diffs if not np.isnan(y) and y not in diffs_y and y > 0]
        print(f"diffs_x: {diffs_x}")
        print(f"diffs_y: {diffs_y}")
        return diffs_x, diffs_y    

    def calc_grid(self, df, cols):
        diffs_x, diffs_y = self.calc_diff(df)
        for col in cols:
            other_col = 'y' if col == 'x' else 'x'
            diffs = diffs_y if col == 'x' else diffs_x
            for i in df[col].sort_values().unique():
                coords = df[df[col] == i].sort_values(by=other_col)
                for j in range(coords.shape[0]):
                    difflast = coords.iloc[j][other_col] - coords.iloc[j-1][other_col]
                    if difflast > np.min(diffs):
                        coords.iloc[j][other_col] = coords.iloc[j-1][other_col] + np.min(diffs)
                df.loc[coords.index, other_col] = coords[other_col]
        return df

    # write above logic in a function, as concise as possible, no plotting
    def spatial_to_grid(self):
        df = pd.DataFrame(self.interpolate_grid(), columns=['x', 'y'])
        xy_order = self.calc_grid(df.copy(), ['x', 'y'])
        yx_order = self.calc_grid(df.copy(), ['y', 'x'])
        # print(f"xy_order: {xy_order.iloc[:10,:]}, yx_order: {yx_order.iloc[:10,:]}")
        self.adata.obsm['grid_spatial_xy'] = np.column_stack([xy_order['x'], xy_order['y']])
        self.adata.obsm['grid_spatial_yx'] = np.column_stack([yx_order['x'], yx_order['y']])
        for key_added in ['grid_spatial_xy', 'grid_spatial_yx', 'spatial']:
            sq.gr.spatial_neighbors(self.adata, spatial_key=key_added, n_neighs=6, key_added=f'{key_added}')
        # calculate similarity between the orinigal adjacency matrix (from 'spatial') and adj matrices from the two grid_spatial embeddings
        spatial_adj = self.adata.obsp['spatial_distances']
        xy_adj = self.adata.obsp['grid_spatial_xy_distances']
        yx_adj = self.adata.obsp['grid_spatial_yx_distances']
        xy_diff = cosine_similarity(spatial_adj, xy_adj).mean()
        yx_diff = cosine_similarity(spatial_adj, yx_adj).mean()
        # print(f"xy_diff: {xy_diff}, yx_diff: {yx_diff}")    
        self.adata.obsm['grid_spatial'] = self.adata.obsm['grid_spatial_xy'].copy() if xy_diff > yx_diff else self.adata.obsm['grid_spatial_yx'].copy()
        for key in ['grid_spatial_xy', 'grid_spatial_yx']:
            del self.adata.obsm[key], self.adata.obsp[f'{key}_distances'], self.adata.obsp[f'{key}_connectivities'], self.adata.uns[f'{key}_neighbors']        
        return self.adata.obsm['grid_spatial'].copy()

    def get_coordinates(self, grid: bool = False):
        """
        Get spatial coordinates from the adata object.
        If grid is True, return the grid coordinates.
        Otherwise, return the original spatial coordinates.
        """
        if grid:
            if 'grid_spatial' not in self.adata.obsm:    
                coords = self.spatial_to_grid()
            else:
                coords = self.adata.obsm['grid_spatial'].copy()
            coords = torch.tensor(coords, dtype=torch.float32)
        else:
            coords = self.coords if isinstance(self.coords, torch.Tensor) else torch.tensor(self.coords, dtype=torch.float32)
        coords = torch.round(coords).to(torch.float32)
        return coords

    def shifted_grid_embedding(self, values, grid=False):
        grid_coords = self.get_coordinates(grid=grid)
        x_max, y_max = grid_coords[:, 0].max(), grid_coords[:, 1].max()
        twod_grid = np.zeros((int(y_max) + 1, int(x_max) + 1))
        for i in range(grid_coords.shape[0]):
            x, y = int(grid_coords[i, 0]), int(grid_coords[i, 1])
            twod_grid[y, x] = values[i]
        return twod_grid

    def coords_to_filled_grid(self, values): # for visualization
        domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes = calculate_domain_parameters(self.coords, divideby=1)
        #### Custom noise spatially dependent
        coords = self.coords - self.coords.mean(dim=0)  # center the coordinates
        grid = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=values,
            coords=coords,
        )[0]
        return grid

    def orig_coord_embedding(self, values):
        """
        Hyper-optimized square drawing with:
        - No extra columns
        - No Python loops
        - Full gradient flow
        - Minimal memory usage
        """
        image_shape = (self.y1_max, self.y2_max)
        coords = self.get_coordinates(grid=False)
        grid_coords = torch.round(coords).long()
        
        # Calculate square parameters
        scale_factor = self.adata.uns['spatial'][self.lib_id]['scalefactors'][f'tissue_{self.img_key}_scalef']
        radius = int(100 * scale_factor)
        size = 2 * radius + 1
        
        # Create base indices for a square [-radius, radius]
        y_idx = torch.arange(-radius, radius+1, device=values.device)
        x_idx = torch.arange(-radius, radius+1, device=values.device)
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing='ij')
        
        # Calculate all possible positions (vectorized)
        all_y = (grid_coords[:, 0].unsqueeze(1).unsqueeze(2) + yy.unsqueeze(0))
        all_x = (grid_coords[:, 1].unsqueeze(1).unsqueeze(2) + xx.unsqueeze(0))
        
        # Flatten and filter valid positions
        flat_y = all_y.flatten()
        flat_x = all_x.flatten()
        valid_mask = (flat_y >= 0) & (flat_y < image_shape[0]) & \
                    (flat_x >= 0) & (flat_x < image_shape[1])
        
        # Prepare values (repeated for each square pixel)
        repeated_values = values.repeat_interleave(size*size)[valid_mask]
        
        # Calculate linear indices for valid positions
        linear_indices = flat_x[valid_mask] * image_shape[1] + flat_y[valid_mask]
        
        # Scatter-add using valid indices only
        grid = torch.zeros(image_shape, dtype=values.dtype, device=values.device)
        grid.view(-1).scatter_add_(0, linear_indices, repeated_values)
        return grid
    
    def compute_distance_cost(self, metric='euclidean', scaling=None):
        coords = self.coords
        if metric == 'euclidean':
            cost = torch.cdist(coords, coords, p=2)
        elif metric == 'manhattan':
            cost = torch.cdist(coords, coords, p=1)
        elif metric == 'chebyshev':
            cost = torch.cdist(coords, coords, p=float('inf'))
        else:
            raise ValueError('Metric not recognized.')
        cost.requires_grad = True
        if scaling:
            cost = self.scale_distance_cost(cost, method='by_sum')
        return cost

    def scale_distance_cost(self, C, method=None):
        c_max = C.max(dim=1, keepdim=True)[0]
        c_min = C.min(dim=1, keepdim=True)[0]
        c_std = C.std(dim=1, keepdim=True)
        c_mean = C.mean(dim=1, keepdim=True)
        c_sum = C.sum(axis=1, keepdims=True)
        if method == 'by_max':
            scaled_cost = C / c_max  # normalized by each row (source)
        elif method == 'by_minmax':
            scaled_cost = (C - c_min) / (c_max - c_min)
        elif method == 'log':
            scaled_cost = torch.log1p(C)
        elif method == 'by_sum':
            scaled_cost = C / c_sum
        elif method == 'by_ttlcnt_range':
            ttl_cnts = self.y_init.sum(axis=1).detach().clone()
            ttl_cnts = ttl_cnts / ttl_cnts.sum()
            ttl_range = [ttl_cnts.min(), ttl_cnts.max()]
            scaled_cost = self.scale_to_range(ttl_range, C)
        else:
            raise(ValueError)
        return scaled_cost

    def impute_qt_vectorized(self, n_steps=20, max_qt=1.0, window_length=5, polyorder=2, derivative_threshold=0.03):
        """
        Vectorized implementation that processes all genes simultaneously.
        Applies Savitzky-Golay smoothing to Moran's I curves and selects turning point quantile.

        Returns:
            best_cutoffs: Tensor of optimal cutoffs for each gene
        """
        gene_indices = self.gene_indices
        X = self.y_init.detach().clone() * self.in_tiss_mask.unsqueeze(1)
        quantiles = torch.linspace(0.05, max_qt, steps=n_steps, dtype=X.dtype)
        quantiles = torch.round(quantiles, decimals=2)

        all_morans = {qt.item(): [float('nan')] * len(gene_indices) for qt in quantiles}
        best_cutoffs = torch.zeros(len(gene_indices))
        best_qts = torch.zeros(len(gene_indices))

        for qt in quantiles:
            cutoffs = torch.quantile(X, qt, dim=0)
            masks = X > cutoffs.unsqueeze(0)
            for gene_idx in range(len(gene_indices)):
                mask = masks[:, gene_idx]
                if mask.sum() < 10:
                    continue
                z = X[mask, gene_idx]
                coords = self.coords[mask]
                W = 1 / (1 + torch.cdist(coords, coords, p=2))
                N = z.shape[0]
                z_mean = z.mean()
                z_centered = z - z_mean
                S0 = W.sum()
                Wz = torch.matmul(W, z_centered)
                current_moran = (N / S0) * (torch.sum(z_centered * Wz) / (torch.sum(z_centered ** 2) + 1e-8))
                all_morans[qt.item()][gene_idx] = current_moran.item()

        for gene_idx in range(len(gene_indices)):
            valid_pairs = [(qt.item(), all_morans[qt.item()][gene_idx]) for qt in quantiles if not np.isnan(all_morans[qt.item()][gene_idx])]
            if not valid_pairs:
                best_qts[gene_idx] = max_qt
                best_cutoffs[gene_idx] = torch.quantile(X[:, gene_idx], max_qt)
                continue

            x = np.array([p[0] for p in valid_pairs])
            y = np.array([p[1] for p in valid_pairs])

            if len(y) >= window_length:
                y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
            else:
                y_smooth = y

            dy = np.gradient(y_smooth)
            turning_indices = np.where(dy > derivative_threshold)[0]
            if len(turning_indices) > 0:
                best_qt = x[turning_indices[0]]
            else:
                best_qt = x[np.argmax(y_smooth)]

            best_qts[gene_idx] = best_qt
            best_cutoffs[gene_idx] = torch.quantile(X[:, gene_idx], best_qt)

        return best_cutoffs

    def plot_moran_curves(self, save_dir=None, n_samples=10, n_steps=20, max_qt=1.0, window_length=5, polyorder=2, derivative_threshold=0.03):
        """
        Plot Moran's I curves for randomly selected genes with smoothing and turning point detection.
        """
        gene_indices = self.gene_indices
        X = self.y_init.detach().clone() * self.in_tiss_mask.unsqueeze(1)
        quantiles = torch.linspace(0.05, max_qt, steps=n_steps, dtype=X.dtype)
        quantiles = torch.round(quantiles, decimals=2)

        selected_genes = np.random.choice(len(gene_indices), size=n_samples, replace=False)

        plt.figure(figsize=(15, 10))
        for i, gene_idx in enumerate(selected_genes):
            gene_name = self.gene_selected[gene_idx]
            morans = []
            for qt in quantiles:
                moran_i = float('nan')
                cutoffs = torch.quantile(X, qt, dim=0)
                mask = X[:, gene_idx] > cutoffs[gene_idx]
                if mask.sum() >= 10:
                    z = X[mask, gene_idx]
                    coords = self.coords[mask]
                    W = 1 / (1 + torch.cdist(coords, coords, p=2))
                    N = z.shape[0]
                    z_mean = z.mean()
                    z_centered = z - z_mean
                    S0 = W.sum()
                    Wz = torch.matmul(W, z_centered)
                    moran_i = (N / S0) * (torch.sum(z_centered * Wz) / (torch.sum(z_centered ** 2) + 1e-8))
                morans.append(moran_i)

            x = np.array([qt.item() for qt in quantiles])
            y = np.array(morans)
            valid_mask = ~np.isnan(y)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            if len(y_valid) >= window_length:
                y_smooth = savgol_filter(y_valid, window_length=window_length, polyorder=polyorder)
            else:
                y_smooth = y_valid

            dy = np.gradient(y_smooth)
            turning_indices = np.where(dy > derivative_threshold)[0]
            turning_qt = x_valid[turning_indices[0]] if len(turning_indices) > 0 else x_valid[np.argmax(y_smooth)]

            plt.subplot(n_samples // 5 + 1, 5, i + 1)
            plt.plot(x_valid, y_valid, label='Raw Moran\'s I')
            plt.plot(x_valid, y_smooth, label='Smoothed', linestyle='--')
            plt.axvline(turning_qt, color='red', linestyle=':', label='Turning Quantile')
            plt.title(f'Gene {gene_name}')
            plt.xlabel('Quantile')
            plt.ylabel('Moran\'s I')
            plt.legend()
            plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/{n_samples}_genes_turning_points.png', bbox_inches='tight')
        else:
            plt.show()

    def set_states(self, qts_prior, derivative_threshold=0.02):
        n_genes = len(self.gene_selected)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.invalid_qts = self.impute_qt_vectorized(max_qt=qts_prior, derivative_threshold=derivative_threshold)  # Shape [N], the quantile cutoff is an estimation such that expression above it is highly likely to be the source
        self.invalid_qts.to(device)        

    def prep_genes_params(self, add_genes=[], first_n_genes=None):
        raw_count = self.raw_count.toarray() if issparse(self.raw_count) else self.raw_count
        self.gene_selected = self.gene_selection()
        first_n_genes = first_n_genes if first_n_genes is not None else len(self.gene_selected)
        self.gene_selected = self.gene_selected[:first_n_genes] + [g for g in add_genes if g not in self.gene_selected]
        self.gene_selected = np.array(self.gene_selected)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gene_indices = torch.tensor(
            [self.adata.var_names.get_loc(g) for g in self.gene_selected],
            dtype=torch.long, device=device
        )  # gene indices indicate gene indices in original data
        self.y_init = torch.tensor(raw_count[:, self.gene_indices], dtype=torch.float32, device=device)
        self.var_ttcnt = torch.tensor(self.adata.var.loc[self.gene_selected, 'total_counts'].values, dtype=torch.float32, device=device)


    def gene_selection(self, out_tiss_perc=0.8, min_count=100):
        if 'highly_variable' not in self.adata.var.columns:
            self.adata.var['MT'] = self.adata.var_names.str.startswith("MT-")
            self.adata.var['mt'] = self.adata.var_names.str.startswith("mt-")
            if any(self.adata.var['MT']) or any(self.adata.var['mt']):
                self.adata.var['mt'] = self.adata.var['MT'] | self.adata.var['mt']
                sc.pp.calculate_qc_metrics(self.adata, qc_vars=["mt"], inplace=True)
            else:
                sc.pp.calculate_qc_metrics(self.adata, inplace=True)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, flavor="seurat", n_top_genes=5000)
        
        out = self.adata[self.adata.obs['in_tissue']==0]
        out.var['total_counts'] = out.X.toarray().sum(axis=0)
        
        for i in range(100, len(out.var_names), 20):
            union_genes = out.var.sort_values('total_counts', ascending=False).index[:i].values
            perc_out = out.var.loc[union_genes,'total_counts'].sum()/out.var['total_counts'].sum()
            if perc_out > out_tiss_perc:
                break
        
        all_hvgs = self.adata.var_names[self.adata.var['highly_variable']].values
        union_genes = np.concatenate([union_genes, all_hvgs])
        union_genes = np.unique(union_genes).tolist()
        
        # Remove mitochondrial and DEPRECATED genes
        to_remove = ['MT-', 'mt-', 'DEPRECATED','Rik']
        union_genes = [x for x in union_genes if not any(sub in x for sub in to_remove)]
        union_genes = [x for x in union_genes if self.adata.var.loc[x, 'total_counts'] > min_count]
        # rank genes by expression counts
        union_genes = self.adata.var.loc[union_genes, 'total_counts'].sort_values(ascending=False).index.tolist()
        # Sort genes by expression similarity
        # union_genes = self.sort_genes_by_similarity(union_genes)        
        return union_genes
        
    def soft_thresholding(self, gene_expr_in, gene_expr_out, quantiles):
        """向量化实现，同时处理所有 genes 的阈值滤波"""
        # cutoffs = self.smooth_quantile_batch(gene_expr_in, quantiles)
        cutoffs = quantiles
        
        # 2. 创建广播用掩码 [n_spots, 1]
        in_tiss_mask_2d = self.in_tiss_mask.unsqueeze(1).bool()  # shape [n_spots, 1]
        
        # 3. 并行计算以下过滤条件（利用广播机制避免大矩阵）==================================
        
        # -- in_tiss_filt条件（梯度可导处理）--
        # (a) 超过阈值的内组织 spots（保留原始值）
        mask_high_expression = (gene_expr_in > cutoffs) & in_tiss_mask_2d  # shape [n_spots, n_genes]
        
        # (b) 低于阈值的内组织 spots（应用软衰减）
        scaled_diff = (gene_expr_in - cutoffs) / (gene_expr_in.max(0)[0] - gene_expr_in.min(0)[0])
        soft_mask = torch.sigmoid(100 * (scaled_diff - 0.5))  # 陡峭但能保持梯度的阈值过渡
        
        in_tiss_filt = torch.where(
            mask_high_expression,
            gene_expr_in,
            gene_expr_in * soft_mask
        )
        
        # -- out_tiss_filt条件（梯度可导处理）--
        # (a) 内组织中低于阈值的部分
        # (b) 全部外组织 spots
        mask_out = (~mask_high_expression) | (~in_tiss_mask_2d)  # shape [n_spots, n_genes]
        
        # (c) 为数值稳定性添加ε
        eps = 1e-20
        out_tiss_filt = torch.where(
            mask_out,
            gene_expr_out,  # 这里使用原始输出的表达量（只读操作）
            torch.tensor(eps, device=gene_expr_out.device)
        )
        
        # 4. 保持梯度连接的数值稳定性
        return torch.clamp(in_tiss_filt, min=eps), torch.clamp(out_tiss_filt, min=eps)

    def prep(self, invalid_qts):
        """
        Prepares the input data for the OT computation.
        Including tissue mask applying to in_tiss, soft thresholding and normalization.
        """
        X_g = self.y_init.detach().clone()
        ttcnt = self.y_init.sum(dim=0)
        out_tiss = X_g.clone()
        in_tiss = X_g.clone()
        in_tiss_filt, out_tiss_filt = self.soft_thresholding(in_tiss, out_tiss, invalid_qts)            
        out_tiss_sum, in_tiss_sum = out_tiss_filt.sum(dim=0), in_tiss_filt.sum(dim=0)
        s = out_tiss_sum + in_tiss_sum
        sum_scale = s / ttcnt
        out_tiss_filt = out_tiss_filt / sum_scale
        in_tiss_filt = in_tiss_filt / sum_scale
        out_tiss_sum, in_tiss_sum = out_tiss_filt.sum(dim=0), in_tiss_filt.sum(dim=0)

        out_tiss_filt = out_tiss_filt / out_tiss_sum
        in_tiss_filt = in_tiss_filt / in_tiss_sum
        
        return out_tiss_filt, in_tiss_filt, out_tiss_sum, in_tiss_sum
        

    def round_counts_to_integers(self, preserve_sum=True, return_copy=False):
        """
        Round corrected expression counts to integers.
        
        Parameters
        ----------
        preserve_sum : bool, default=True
            If True, preserve the column-wise sums during rounding
        return_copy : bool, default=False
            If True, return a copy of the rounded counts without modifying self.X_init
            
        Returns
        -------
        rounded_counts : torch.Tensor
            Rounded count matrix (returned only if return_copy=True)
        """
        if self.X_init is None:
            raise ValueError("No corrected counts available. Run optimization first.")
            
        # Work with a copy to avoid modifying the original
        rounded_counts = self.X_init.clone()
        
        if preserve_sum:
            # Approach 1: Preserve column sums using largest remainder method
            # Get original column sums
            original_sums = rounded_counts.sum(dim=0)
            
            # Simple rounding
            simple_rounded = torch.round(rounded_counts)
            
            # Calculate column sums after simple rounding
            rounded_sums = simple_rounded.sum(dim=0)
            
            # Calculate deficit or excess for each column
            count_diff = original_sums - rounded_sums
            
            for j in range(rounded_counts.shape[1]):
                if count_diff[j] == 0:
                    continue
                    
                # Get the fractional parts
                frac_parts = rounded_counts[:, j] - simple_rounded[:, j]
                
                # We need to adjust based on deficit or excess
                adjustment = int(count_diff[j].item())
                
                if adjustment > 0:  # Need to round up more values
                    # Find cells with highest fractional parts that were rounded down
                    rounded_down = (frac_parts > 0) & (frac_parts < 0.5)
                    candidates = torch.where(rounded_down)[0]
                    
                    # If not enough candidates, also consider those that were rounded up
                    if len(candidates) < adjustment:
                        rounded_up = frac_parts >= 0.5
                        more_candidates = torch.where(rounded_up)[0]
                        candidates = torch.cat([candidates, more_candidates])
                    
                    # Sort by fractional part (descending)
                    if len(candidates) > 0:
                        sorted_indices = candidates[torch.argsort(-frac_parts[candidates])]
                        # Adjust by rounding up the top candidates
                        for i in range(min(adjustment, len(sorted_indices))):
                            simple_rounded[sorted_indices[i], j] += 1
                            
                elif adjustment < 0:  # Need to round down more values
                    # Find cells with lowest fractional parts that were rounded up
                    rounded_up = frac_parts >= 0.5
                    candidates = torch.where(rounded_up)[0]
                    
                    # If not enough candidates, also consider those that were rounded down
                    if len(candidates) < abs(adjustment):
                        rounded_down = frac_parts < 0.5
                        more_candidates = torch.where(rounded_down)[0]
                        candidates = torch.cat([candidates, more_candidates])
                    
                    # Sort by fractional part (ascending)
                    if len(candidates) > 0:
                        sorted_indices = candidates[torch.argsort(frac_parts[candidates])]
                        # Adjust by rounding down the bottom candidates
                        for i in range(min(abs(adjustment), len(sorted_indices))):
                            simple_rounded[sorted_indices[i], j] -= 1
            
            rounded_counts = simple_rounded
        else:
            # Simple rounding without preserving sums
            rounded_counts = torch.round(rounded_counts)
        
        # Ensure no negative values
        rounded_counts = torch.clamp(rounded_counts, min=0)
        
        # Either return the copy or modify in-place
        if return_copy:
            return rounded_counts
        else:
            self.X_init = rounded_counts
            print(f"Rounded counts stored in X_init. Original range: [{self.X_init.min():.2f}, {self.X_init.max():.2f}]")
            return None
        
