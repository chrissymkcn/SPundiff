import random   
import scanpy as sc
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score, silhouette_score
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

def preprocess(adata_ot, selected_genes=None):
    sc.pp.normalize_total(adata_ot, target_sum=1e4)
    sc.pp.log1p(adata_ot)
    sc.pp.highly_variable_genes(adata_ot, flavor="seurat", n_top_genes=2000)
    if selected_genes is not None:
        adata_ot = adata_ot[:, selected_genes]
    adata_ot.raw = adata_ot
    sc.pp.scale(adata_ot, max_value=10)
    sc.pp.pca(adata_ot)
    sc.pp.neighbors(adata_ot)
    sc.tl.umap(adata_ot)
    return adata_ot

def eval_clustering(adata_ot, embed='counts', leiden_range=[0.3, 1.3], size=1.3):
    if embed == 'X':
        X = adata_ot.X
    elif embed in 'pca':    
        X = adata_ot.obsm['X_pca']
    elif embed == 'umap':
        X = adata_ot.obsm['X_umap']
    elif embed == 'spatial':
        X = adata_ot.obsm['spatial']
    else:
        raise ValueError('embed not recognized')
    plt.rcParams["figure.figsize"] = (4, 4)
    sild = {}
    for res in np.arange(leiden_range[0], leiden_range[1], 0.1):
        res = np.round(res, 1)
        sc.tl.leiden(adata_ot, resolution=res, key_added=f'leiden_{res}')
        if adata_ot.obs[f'leiden_{res}'].nunique() == 1:
            continue
        print(sc.pl.spatial(adata_ot, color=f'leiden_{res}', 
                            title=f"resolution={res}", size=size))
        cluster = adata_ot.obs[f'leiden_{res}'].values
        silhouette = silhouette_score(X, cluster)
        sild[res] = silhouette
    return adata_ot, sild


#### Cluster quality metrics
def calculate_spatial_coherence(adata, cluster_key):
    """Calculate spatial coherence of clusters"""
    # Get spatial neighbors
    if 'spatial_connectivities' not in adata.obsp:
        import squidpy as sq
        sq.gr.spatial_neighbors(adata)
    
    spatial_conn = adata.obsp['spatial_connectivities'].toarray()
    labels = adata.obs[cluster_key].values
    
    # Count same-cluster vs different-cluster neighbors
    same_cluster = 0
    total_connections = 0
    
    for i in range(spatial_conn.shape[0]):
        for j in range(i+1, spatial_conn.shape[1]):
            if spatial_conn[i, j] > 0:  # They are spatial neighbors
                total_connections += 1
                if labels[i] == labels[j]:
                    same_cluster += 1
    
    return same_cluster / max(1, total_connections)

def calculate_spatial_expression_consistency(adata, cluster_key):
    """
    Calculate how consistent clusters are between
    expression space and spatial neighborhood space
    """
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    
    # Get expression-based clusters
    expr_clusters = adata.obs[cluster_key].values
    
    # Create spatial-only clustering
    if 'spatial_neighbors' not in adata.uns:
        import squidpy as sq
        sq.gr.spatial_neighbors(adata, n_neighs=15)
    
    import scanpy as sc
    sc.tl.leiden(adata, resolution=0.8, key_added='spatial_clusters',
                 neighbors_key='spatial_neighbors')
    
    spatial_clusters = adata.obs['spatial_clusters'].values
    
    # Calculate consistency using ARI
    return adjusted_rand_score(expr_clusters, spatial_clusters)

def assess_cluster_quality(adata, resolutions=None, optimal_res=None):
    """
    Comprehensively assess cluster quality at selected resolutions
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    resolutions : list, default=None
        List of resolutions to evaluate
        If None, will use [0.2, 0.4, 0.8, 1.2]
    optimal_res : float, default=None
        Optimal resolution (will be included in evaluation)
        
    Returns:
    --------
    dict
        Dictionary containing various cluster quality metrics
    """
    
    # Default resolutions if none provided
    if resolutions is None:
        resolutions = [0.2, 0.4, 0.8, 1.0, 1.2]
    
    # Add optimal resolution if provided
    if optimal_res is not None and optimal_res not in resolutions:
        resolutions.append(optimal_res)
        resolutions.sort()
    
    # Make sure we have neighbors computed
    if 'neighbors' not in adata.uns or 'X_pca' not in adata.obsm.keys():
        adata = preprocess(adata)
    
    results = {}
    
    # For each resolution
    for res in resolutions:
        # Run leiden clustering
        res_key = f'leiden_{res}'
        if res_key not in adata.obs:
            sc.tl.leiden(adata, resolution=res, key_added=res_key)
        
        # Get cluster labels
        labels = adata.obs[res_key].astype(int).values
        n_clusters = len(np.unique(labels))
        
        # Skip if only 1 cluster
        if n_clusters <= 1:
            continue
            
        # Calculate spatial coherence
        spatial_coherence = calculate_spatial_coherence(adata, res_key)
        
        # Calculate standard clustering metrics
        try:
            silhouette_avg = silhouette_score(
                adata.obsm['X_pca'], labels, sample_size=min(5000, adata.n_obs))
            
            silhouette_vals = silhouette_samples(
                adata.obsm['X_pca'], labels, sample_size=min(5000, adata.n_obs))
        except:
            silhouette_avg = np.nan
            silhouette_vals = np.nan
        
        try:
            calinski = calinski_harabasz_score(adata.obsm['X_pca'], labels)
            davies = davies_bouldin_score(adata.obsm['X_pca'], labels)
        except:
            calinski = np.nan
            davies = np.nan
            
        try:
            # Calculate per-cluster silhouette scores
            cluster_silhouettes = {}
            for i in range(n_clusters):
                cluster_silhouettes[f"cluster_{i}"] = np.mean(silhouette_vals[labels==i])
        except:
            cluster_silhouettes = {}
        
        # Calculate spatial vs expression consistency
        spatial_expr_consistency = calculate_spatial_expression_consistency(adata, res_key)
        
        # Store results
        results[res] = {
            'n_clusters': n_clusters,
            'spatial_coherence': spatial_coherence,
            'silhouette_avg': silhouette_avg,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'cluster_silhouettes': cluster_silhouettes,
            'spatial_expr_consistency': spatial_expr_consistency
        }
    
    return results


#### Cluster stability evaluation by subsampling
def cluster_stability_under_subsampling(adata, 
                                        n_iterations=20, 
                                        subsample_fraction=0.8,
                                        resolution=1.0,
                                        spatial_constraint=True,
                                        random_seed=42):
    """
    Evaluate cluster stability by performing multiple rounds of spatial subsampling
    and measuring cluster assignment consistency.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with spatial coordinates in adata.obsm['spatial']
    n_iterations : int, default=20
        Number of subsampling iterations
    subsample_fraction : float, default=0.8
        Fraction of cells to include in each subsample
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering
    spatial_constraint : bool, default=True
        Whether to subsample in a spatially coherent way
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    float
        Stability score between 0 and 1, where higher values indicate more stable clustering
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Original clustering as reference
    if f'leiden_{resolution:.1f}' not in adata.obs.columns:
        sc.pp.neighbors(adata, use_rep='X' if 'X_pca' not in adata.obsm else 'X_pca')
        sc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_{resolution:.1f}')
    reference_labels = adata.obs[f'leiden_{resolution:.1f}'].astype(int).values
    
    # Store pairwise ARI scores
    stability_scores = []
    
    # Store cluster assignments for all iterations
    all_cluster_assignments = np.zeros((adata.n_obs, n_iterations), dtype=int)
    
    for i in range(n_iterations):
        # Create a subsample mask
        n_cells = adata.n_obs
        n_subsample = int(subsample_fraction * n_cells)
        
        if spatial_constraint:
            # Spatial-aware subsampling by randomly selecting seed points
            # and including their neighbors
            spatial_coords = adata.obsm['spatial']
            
            # Start with random seeds (5-10% of total desired points)
            n_seeds = int(0.1 * n_subsample)
            seed_indices = np.random.choice(n_cells, size=n_seeds, replace=False)
            
            # Compute distances to all seeds
            from scipy.spatial.distance import cdist
            distances = cdist(spatial_coords, spatial_coords[seed_indices])
            
            # For each point, find distance to nearest seed
            min_distances = np.min(distances, axis=1)
            
            # Select points with probability inversely proportional to distance
            # This creates spatially coherent clusters around seeds
            p = 1 / (1 + min_distances)
            p = p / np.sum(p)
            
            # Sample remaining points based on this probability
            remaining_indices = np.random.choice(
                n_cells, 
                size=n_subsample-n_seeds, 
                replace=False, 
                p=p
            )
            
            subsample_indices = np.concatenate([seed_indices, remaining_indices])
        else:
            # Simple random subsampling
            subsample_indices = np.random.choice(n_cells, size=n_subsample, replace=False)
        
        # Create subsample view
        subsample_mask = np.zeros(n_cells, dtype=bool)
        subsample_mask[subsample_indices] = True
        adata_sub = adata[subsample_mask].copy()
        
        # Clustering on subsample
        if f'leiden_{resolution:.1f}' not in adata_sub.obs.columns:
            sc.pp.neighbors(adata_sub, use_rep='X' if 'X_pca' not in adata_sub.obsm else 'X_pca')
            sc.tl.leiden(adata_sub, resolution=resolution, key_added=f'leiden_{resolution:.1f}')
        
        # Map labels back to original data
        subsample_labels = -np.ones(n_cells, dtype=int)
        subsample_labels[subsample_indices] = adata_sub.obs[f'leiden_{resolution:.1f}'].astype(int).values
        
        # Store cluster assignments
        all_cluster_assignments[subsample_indices, i] = subsample_labels[subsample_indices]
        
        # Calculate ARI between this subsample and reference (only for overlapping cells)
        overlap_mask = subsample_mask
        if np.sum(overlap_mask) > 0:
            ari = adjusted_rand_score(
                reference_labels[overlap_mask],
                subsample_labels[overlap_mask]
            )
            stability_scores.append(ari)
    
    # Calculate co-clustering probabilities for each pair of cells
    n_overlaps = np.zeros((n_cells, n_cells), dtype=int)
    n_coclusters = np.zeros((n_cells, n_cells), dtype=int)
    
    for i in range(n_iterations):
        # Get current iteration's clustering
        current_labels = all_cluster_assignments[:, i]
        
        # Find cells that were included in this iteration
        included = current_labels >= 0
        included_indices = np.where(included)[0]
        
        # Update co-clustering matrix for included cells
        for idx1 in range(len(included_indices)):
            i1 = included_indices[idx1]
            for idx2 in range(idx1+1, len(included_indices)):
                i2 = included_indices[idx2]
                n_overlaps[i1, i2] += 1
                n_overlaps[i2, i1] += 1
                
                # Check if they're in the same cluster
                if current_labels[i1] == current_labels[i2]:
                    n_coclusters[i1, i2] += 1
                    n_coclusters[i2, i1] += 1
    
    # Calculate consistency for each cell pair that appears together
    valid_pairs = n_overlaps > 0
    consistency_matrix = np.zeros((n_cells, n_cells))
    consistency_matrix[valid_pairs] = n_coclusters[valid_pairs] / n_overlaps[valid_pairs]
    
    # Calculate overall stability score
    mean_stability = np.mean(stability_scores)
    mean_consistency = np.sum(consistency_matrix) / np.sum(valid_pairs)
    
    # Combine the two metrics (ARI-based and co-clustering based)
    overall_stability = 0.5 * mean_stability + 0.5 * mean_consistency
    
    # Store results in adata for visualization
    adata.uns['cluster_stability'] = {
        'overall_score': overall_stability,
        'ari_scores': stability_scores,
        'consistency_score': mean_consistency
    }
    
    # Calculate per-spot stability
    spot_stability = np.zeros(n_cells)
    for i in range(n_cells):
        valid_comparisons = valid_pairs[i]
        if np.any(valid_comparisons):
            spot_stability[i] = np.mean(consistency_matrix[i, valid_comparisons])
    
    # Add spot-level stability scores to adata
    adata.obs['cluster_stability'] = spot_stability
    
    return overall_stability

def assess_cluster_subsampling_stability(adata, adata_corrected):
    # Before correction
    stability_before = cluster_stability_under_subsampling(adata)
    adata.obs['stability_before'] = adata.obs['cluster_stability'].copy()

    # After correction (with your corrected AnnData object)
    stability_after = cluster_stability_under_subsampling(adata_corrected)
    adata_corrected.obs['stability_after'] = adata_corrected.obs['cluster_stability'].copy()

    print(f"Clustering stability before correction: {stability_before:.3f}")
    print(f"Clustering stability after correction: {stability_after:.3f}")
    print(f"Stability improvement: {stability_after - stability_before:.3f} ({(stability_after/stability_before - 1)*100:.1f}%)")

    # Visualize stability on the tissue
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.spatial(adata, color='stability_before', ax=ax1, title='Cluster stability before correction')
    sc.pl.spatial(adata_corrected, color='stability_after', ax=ax2, title='Cluster stability after correction')
                                         
                                            
#### Clustering stability across resolutions
def cluster_stability_across_resolutions(adata, 
                                           resolution_range=None, 
                                           n_resolutions=10,
                                           use_rep='X_pca',
                                           plot=True):
    """
    Evaluate clustering stability across different Leiden resolution parameters.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    resolution_range : tuple, default=None
        (min_resolution, max_resolution) range to test
        If None, will use (0.1, 2.5)
    n_resolutions : int, default=10
        Number of resolution values to test
    use_rep : str, default='X_pca'
        Representation to use for clustering
    plot : bool, default=True
        Whether to generate plots
        
    Returns:
    --------
    dict
        Dictionary containing stability metrics and optimal resolution
    """    
    # Generate resolution values to test
    if resolution_range is None:
        resolution_range = (0.1, 1.5)
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
    
    # Store results
    results = {
        'resolutions': resolutions,
        'n_clusters': [],
        'silhouette': [],
        'ari_matrix': np.zeros((n_resolutions, n_resolutions)),
        'cluster_labels': []
    }

    # Ensure neighbors are computed
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep)

    # Run clustering at each resolution
    for i, res in enumerate(resolutions):
        print(f"Computing clustering at resolution {res:.2f}")
        if not f'leiden_{res:.1f}' in adata.obs.columns:
            sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res:.1f}')
        labels = adata.obs[f'leiden_{res:.1f}'].astype(int).values
        results['cluster_labels'].append(labels)
        results['n_clusters'].append(len(np.unique(labels)))
        
        # Calculate silhouette score if more than one cluster
        if results['n_clusters'][-1] > 1:
            try:
                # Use PCA space for silhouette to make it computationally feasible
                s_score = silhouette_score(
                    adata.obsm[use_rep], 
                    labels, 
                    sample_size=min(5000, adata.n_obs)
                )
                results['silhouette'].append(s_score)
            except:
                results['silhouette'].append(np.nan)
        else:
            results['silhouette'].append(np.nan)
    
    # Calculate pairwise ARI between all resolutions
    for i in range(n_resolutions):
        for j in range(n_resolutions):
            results['ari_matrix'][i, j] = adjusted_rand_score(
                results['cluster_labels'][i],
                results['cluster_labels'][j]
            )
    
    # Calculate stability metric: average ARI with adjacent resolutions
    stability_scores = []
    for i in range(n_resolutions):
        adjacent_scores = []
        # Compare with resolution to the left
        if i > 0:
            adjacent_scores.append(results['ari_matrix'][i, i-1])
        # Compare with resolution to the right
        if i < n_resolutions - 1:
            adjacent_scores.append(results['ari_matrix'][i, i+1])
        stability_scores.append(np.mean(adjacent_scores))
    
    results['stability_scores'] = stability_scores
    
    # Calculate overall metrics
    results['mean_stability'] = np.mean(stability_scores)
    
    # Find "stability plateaus" - regions where clustering is consistent
    stability_diffs = np.abs(np.diff(stability_scores))
    plateau_indices = np.where(stability_diffs < np.percentile(stability_diffs, 25))[0]
    plateau_groups = []
    current_group = [plateau_indices[0]]
    
    for i in range(1, len(plateau_indices)):
        if plateau_indices[i] == plateau_indices[i-1] + 1:
            current_group.append(plateau_indices[i])
        else:
            plateau_groups.append(current_group)
            current_group = [plateau_indices[i]]
    
    if current_group:
        plateau_groups.append(current_group)
    
    # Find longest plateau
    if plateau_groups:
        longest_plateau = max(plateau_groups, key=len)
        plateau_resolutions = resolutions[longest_plateau]
        results['stable_resolution_range'] = (plateau_resolutions.min(), plateau_resolutions.max())
        
        # Choose middle of longest plateau as optimal resolution
        mid_idx = longest_plateau[len(longest_plateau) // 2]
        results['optimal_resolution'] = resolutions[mid_idx]
    else:
        # If no clear plateau, choose resolution with highest stability
        best_idx = np.argmax(stability_scores)
        results['optimal_resolution'] = resolutions[best_idx]
    
    # Generate plots
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot number of clusters vs resolution
        axs[0, 0].plot(resolutions, results['n_clusters'], '-o')
        axs[0, 0].set_xlabel('Resolution')
        axs[0, 0].set_ylabel('Number of clusters')
        axs[0, 0].set_title('Number of clusters vs Resolution')
        
        # Plot silhouette score vs resolution
        axs[0, 1].plot(resolutions, results['silhouette'], '-o')
        axs[0, 1].set_xlabel('Resolution')
        axs[0, 1].set_ylabel('Silhouette score')
        axs[0, 1].set_title('Silhouette score vs Resolution')
        
        # Plot stability score vs resolution
        axs[1, 0].plot(resolutions, stability_scores, '-o')
        axs[1, 0].axvline(results['optimal_resolution'], color='red', linestyle='--', 
                          label=f"Optimal: {results['optimal_resolution']:.2f}")
        axs[1, 0].set_xlabel('Resolution')
        axs[1, 0].set_ylabel('Stability score')
        axs[1, 0].set_title('Stability score vs Resolution')
        axs[1, 0].legend()
        
        # Plot ARI heatmap
        im = axs[1, 1].imshow(results['ari_matrix'], cmap='viridis')
        axs[1, 1].set_title('ARI between resolutions')
        axs[1, 1].set_xlabel('Resolution index')
        axs[1, 1].set_ylabel('Resolution index')
        plt.colorbar(im, ax=axs[1, 1])
        
        plt.tight_layout()
        
    return results

def assess_cluster_diffres_stability(adata_orig, adata_corrected, title_prefix=""):
    """Compare clustering stability before and after correction"""
    print("Evaluating original data clustering stability...")
    orig_results = cluster_stability_across_resolutions(
        adata_orig, 
        plot=False
    )
    
    print("Evaluating corrected data clustering stability...")
    corr_results = cluster_stability_across_resolutions(
        adata_corrected, 
        plot=False
    )
    
    # Compare key metrics
    metrics = {
        'Mean stability': (orig_results['mean_stability'], corr_results['mean_stability']),
        'Optimal resolution': (orig_results['optimal_resolution'], corr_results['optimal_resolution']),
        'Max silhouette': (np.nanmax(orig_results['silhouette']), np.nanmax(corr_results['silhouette'])),
        'Resolution stability range': (
            f"{orig_results.get('stable_resolution_range', (None, None))[0]:.2f}-{orig_results.get('stable_resolution_range', (None, None))[1]:.2f}", 
            f"{corr_results.get('stable_resolution_range', (None, None))[0]:.2f}-{corr_results.get('stable_resolution_range', (None, None))[1]:.2f}"
        )
    }
    
    # Print comparison table
    print(f"\n{title_prefix} Clustering Stability Comparison:")
    print(f"{'Metric':<25} {'Original':>15} {'Corrected':>15} {'Change':>15}")
    print("-" * 70)
    
    for metric, (orig_val, corr_val) in metrics.items():
        if isinstance(orig_val, (int, float)) and isinstance(corr_val, (int, float)):
            change = corr_val - orig_val
            change_pct = (corr_val / orig_val - 1) * 100 if orig_val != 0 else float('inf')
            change_str = f"{change:.3f} ({change_pct:+.1f}%)"
            print(f"{metric:<25} {orig_val:>15.3f} {corr_val:>15.3f} {change_str:>15}")
        else:
            print(f"{metric:<25} {orig_val:>15} {corr_val:>15}")
    
    # Create comparison plots
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot number of clusters
    axs[0, 0].plot(orig_results['resolutions'], orig_results['n_clusters'], '-o', label='Original')
    axs[0, 0].plot(corr_results['resolutions'], corr_results['n_clusters'], '-o', label='Corrected')
    axs[0, 0].set_xlabel('Resolution')
    axs[0, 0].set_ylabel('Number of clusters')
    axs[0, 0].set_title('Number of clusters vs Resolution')
    axs[0, 0].legend()
    
    # Plot silhouette scores
    axs[0, 1].plot(orig_results['resolutions'], orig_results['silhouette'], '-o', label='Original')
    axs[0, 1].plot(corr_results['resolutions'], corr_results['silhouette'], '-o', label='Corrected')
    axs[0, 1].set_xlabel('Resolution')
    axs[0, 1].set_ylabel('Silhouette score')
    axs[0, 1].set_title('Silhouette score vs Resolution')
    axs[0, 1].legend()
    
    # Plot stability scores
    axs[1, 0].plot(orig_results['resolutions'], orig_results['stability_scores'], '-o', label='Original')
    axs[1, 0].plot(corr_results['resolutions'], corr_results['stability_scores'], '-o', label='Corrected')
    axs[1, 0].axvline(orig_results['optimal_resolution'], color='blue', linestyle='--', 
                      label=f"Orig optimal: {orig_results['optimal_resolution']:.2f}")
    axs[1, 0].axvline(corr_results['optimal_resolution'], color='orange', linestyle='--', 
                      label=f"Corr optimal: {corr_results['optimal_resolution']:.2f}")
    axs[1, 0].set_xlabel('Resolution')
    axs[1, 0].set_ylabel('Stability score')
    axs[1, 0].set_title('Stability score vs Resolution')
    axs[1, 0].legend()
    
    # Plot ARI difference (corrected - original)
    ari_diff = corr_results['ari_matrix'] - orig_results['ari_matrix']
    im = axs[1, 1].imshow(ari_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axs[1, 1].set_title('ARI difference (corrected - original)')
    axs[1, 1].set_xlabel('Resolution index')
    axs[1, 1].set_ylabel('Resolution index')
    plt.colorbar(im, ax=axs[1, 1])
    
    plt.tight_layout()
    plt.suptitle(f"{title_prefix} Clustering Stability Before vs After Correction", y=1.02, fontsize=16)
    
    return {
        'original': orig_results,
        'corrected': corr_results,
        'comparison': metrics
    }


#### Boundary consistency evaluation
def boundary_consistency_across_resolutions(adata, 
                                           resolution_range=None,
                                           n_resolutions=10,
                                           image_key='hires',
                                           use_tissue_features=True,
                                           return_visualizations=False):
    """
    Evaluate boundary consistency of spatial clustering across Leiden resolutions
    with integration of tissue histology features.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with spatial information
    resolution_range : tuple, default=None
        (min_resolution, max_resolution) range to test
        If None, will use (0.1, 2.0)
    n_resolutions : int, default=10
        Number of resolution values to test
    image_key : str, default='hires'
        Key in adata.uns['spatial'][lib_id]['images'] for the tissue image
    use_tissue_features : bool, default=True
        Whether to incorporate tissue image features in boundary evaluation
    return_visualizations : bool, default=False
        Whether to return visualization elements
        
    Returns:
    --------
    dict
        Dictionary containing boundary consistency metrics and visualizations
    """
    import scanpy as sc
    import squidpy as sq
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from skimage import filters, feature, segmentation, color, measure
    from scipy.spatial import distance
    
    # Generate resolution values to test
    if resolution_range is None:
        resolution_range = (0.1, 2.0)
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
    
    # Ensure neighbors are computed
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata)
    
    # Extract tissue image if available and requested
    tissue_image = None
    edge_mask = None
    if use_tissue_features and 'spatial' in adata.uns:
        lib_id = list(adata.uns['spatial'].keys())[0]
        if image_key in adata.uns['spatial'][lib_id]['images']:
            tissue_image = adata.uns['spatial'][lib_id]['images'][image_key]
            
            # Extract edges from tissue image
            if tissue_image.ndim == 3:  # RGB image
                gray_img = color.rgb2gray(tissue_image)
            else:
                gray_img = tissue_image
                
            # Detect edges in tissue image
            edge_mask = feature.canny(gray_img, sigma=2)
    
    # Run clustering at each resolution and extract boundaries
    cluster_labels = []
    boundary_masks = []
    boundary_points = []
    
    for i, res in enumerate(resolutions):
        print(f"Computing clustering at resolution {res:.2f}")
        key = f'leiden_{res:.1f}'
        if key not in adata.obs.columns:
            sc.tl.leiden(adata, resolution=res, key_added=key)
        
        # Store cluster labels
        labels = adata.obs[key].astype(int).values
        cluster_labels.append(labels)
        
        # Generate spatial map of clusters to get boundaries
        if 'spatial' in adata.obsm:
            # Create a labeled image based on spatial coordinates and clusters
            cluster_map = sq.im.cluster_map(
                adata, 
                cluster_key=key, 
                library_id=lib_id if 'lib_id' in locals() else None
            )
            
            # Extract boundary mask
            boundaries = segmentation.find_boundaries(cluster_map)
            boundary_masks.append(boundaries)
            
            # Extract boundary coordinates
            y_coords, x_coords = np.where(boundaries)
            boundary_points.append(np.column_stack([y_coords, x_coords]))
    
    # Calculate pairwise boundary consistency scores
    n_res = len(resolutions)
    boundary_overlap_matrix = np.zeros((n_res, n_res))
    hausdorff_distance_matrix = np.zeros((n_res, n_res))
    
    for i in range(n_res):
        for j in range(n_res):
            if i == j:
                boundary_overlap_matrix[i, j] = 1.0
                hausdorff_distance_matrix[i, j] = 0.0
                continue
                
            # Skip if either resolution produced no boundaries
            if len(boundary_points[i]) == 0 or len(boundary_points[j]) == 0:
                boundary_overlap_matrix[i, j] = np.nan
                hausdorff_distance_matrix[i, j] = np.nan
                continue
            
            # Calculate IoU of boundary masks
            if len(boundary_masks) > 0:
                intersection = np.logical_and(boundary_masks[i], boundary_masks[j]).sum()
                union = np.logical_or(boundary_masks[i], boundary_masks[j]).sum()
                boundary_overlap_matrix[i, j] = intersection / union if union > 0 else 0
            
            # Calculate Hausdorff distance between boundary points
            try:
                # Sample points if too many (for performance)
                pts_i = boundary_points[i]
                pts_j = boundary_points[j]
                
                if len(pts_i) > 1000:
                    idx = np.random.choice(len(pts_i), 1000, replace=False)
                    pts_i = pts_i[idx]
                    
                if len(pts_j) > 1000:
                    idx = np.random.choice(len(pts_j), 1000, replace=False)
                    pts_j = pts_j[idx]
                
                hausdorff_distance_matrix[i, j] = distance.directed_hausdorff(pts_i, pts_j)[0]
            except:
                hausdorff_distance_matrix[i, j] = np.nan
    
    # Calculate image-based consistency if tissue image is available
    image_alignment_scores = []
    if edge_mask is not None:
        for i in range(n_res):
            if len(boundary_masks) > i:
                # Calculate overlap between cluster boundaries and tissue edges
                intersection = np.logical_and(boundary_masks[i], edge_mask).sum()
                union = np.logical_or(boundary_masks[i], edge_mask).sum()
                alignment_score = intersection / union if union > 0 else 0
                image_alignment_scores.append(alignment_score)
            else:
                image_alignment_scores.append(np.nan)
    
    # Calculate stability metrics
    # 1. Adjacent resolution consistency (for each resolution, compare with neighbors)
    adjacent_consistency = []
    for i in range(n_res):
        adjacent_scores = []
        if i > 0:  # Compare with previous resolution
            adjacent_scores.append(boundary_overlap_matrix[i, i-1])
        if i < n_res - 1:  # Compare with next resolution
            adjacent_scores.append(boundary_overlap_matrix[i, i+1])
        adjacent_consistency.append(np.mean(adjacent_scores) if adjacent_scores else np.nan)
    
    # 2. Find plateaus of consistency
    consistency_diffs = np.abs(np.diff(adjacent_consistency))
    stable_indices = np.where(consistency_diffs < np.percentile(consistency_diffs, 25))[0]
    
    # Identify longest plateau
    plateau_groups = []
    if len(stable_indices) > 0:
        current_group = [stable_indices[0]]
        for i in range(1, len(stable_indices)):
            if stable_indices[i] == stable_indices[i-1] + 1:
                current_group.append(stable_indices[i])
            else:
                plateau_groups.append(current_group)
                current_group = [stable_indices[i]]
        if current_group:
            plateau_groups.append(current_group)
    
    # Results compilation
    results = {
        'resolutions': resolutions,
        'cluster_labels': cluster_labels,
        'boundary_overlap_matrix': boundary_overlap_matrix,
        'hausdorff_distance_matrix': hausdorff_distance_matrix,
        'adjacent_consistency': adjacent_consistency,
        'plateau_groups': plateau_groups,
        'image_alignment_scores': image_alignment_scores if edge_mask is not None else None
    }
    
    # Find optimal resolution based on combined boundary stability and image alignment
    if edge_mask is not None:
        # Combine boundary consistency and image alignment
        combined_scores = []
        for i in range(n_res):
            if np.isnan(adjacent_consistency[i]) or np.isnan(image_alignment_scores[i]):
                combined_scores.append(np.nan)
            else:
                combined_scores.append(0.7 * adjacent_consistency[i] + 0.3 * image_alignment_scores[i])
        
        best_idx = np.nanargmax(combined_scores)
        results['optimal_resolution'] = resolutions[best_idx]
        results['combined_scores'] = combined_scores
    else:
        best_idx = np.nanargmax(adjacent_consistency)
        results['optimal_resolution'] = resolutions[best_idx]
    
    # Create visualization if requested
    if return_visualizations:
        # Set up visualization objects
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Boundary consistency across resolutions
        axs[0, 0].plot(resolutions, adjacent_consistency, '-o')
        axs[0, 0].set_xlabel('Resolution')
        axs[0, 0].set_ylabel('Boundary consistency')
        axs[0, 0].set_title('Boundary consistency across resolutions')
        axs[0, 0].axvline(results['optimal_resolution'], color='red', linestyle='--', 
                         label=f"Optimal: {results['optimal_resolution']:.2f}")
        axs[0, 0].legend()
        
        # Plot 2: Image alignment scores if available
        if edge_mask is not None:
            axs[0, 1].plot(resolutions, image_alignment_scores, '-o')
            axs[0, 1].set_xlabel('Resolution')
            axs[0, 1].set_ylabel('Image alignment score')
            axs[0, 1].set_title('Cluster boundaries alignment with tissue features')
        else:
            axs[0, 1].text(0.5, 0.5, "No tissue image features available", 
                          ha='center', va='center')
        
        # Plot 3: Boundary overlap heatmap
        im = axs[1, 0].imshow(boundary_overlap_matrix, cmap='viridis')
        axs[1, 0].set_title('Boundary overlap between resolutions')
        axs[1, 0].set_xlabel('Resolution index')
        axs[1, 0].set_ylabel('Resolution index')
        plt.colorbar(im, ax=axs[1, 0])
        
        # Plot 4: Combined visualization
        # Find optimal resolution for visualization
        opt_idx = np.where(resolutions == results['optimal_resolution'])[0][0]
        
        # If we have boundary masks, show the optimal one
        if len(boundary_masks) > opt_idx:
            if edge_mask is not None:
                # Overlay cluster boundaries on tissue edges
                overlay = np.zeros((*edge_mask.shape, 3))
                overlay[edge_mask, 0] = 1.0  # Tissue edges in red
                overlay[boundary_masks[opt_idx], 1] = 1.0  # Cluster boundaries in green
                overlay[(edge_mask & boundary_masks[opt_idx]), :2] = 1.0  # Overlap in yellow
                
                axs[1, 1].imshow(overlay)
                axs[1, 1].set_title(f"Boundary alignment at res={results['optimal_resolution']:.2f}")
            else:
                axs[1, 1].imshow(boundary_masks[opt_idx])
                axs[1, 1].set_title(f"Cluster boundaries at res={results['optimal_resolution']:.2f}")
        else:
            axs[1, 1].text(0.5, 0.5, "Boundary visualization not available", 
                          ha='center', va='center')
        
        plt.tight_layout()
        results['visualization'] = fig
    
    return results

def assess_boundary_consistency(adata_orig, adata_corrected, **kwargs):
    """
    Compare boundary consistency metrics between original and corrected data
    
    Parameters:
    -----------
    adata_orig : AnnData
        Original data before diffusion correction
    adata_corrected : AnnData
        Data after diffusion correction
    **kwargs : dict
        Additional arguments to pass to boundary_consistency_across_resolutions
        
    Returns:
    --------
    dict
        Comparison results and visualizations
    """
    
    # Evaluate boundary consistency for both datasets
    print("Evaluating boundary consistency for original data...")
    orig_results = boundary_consistency_across_resolutions(
        adata_orig, return_visualizations=False, **kwargs)
    
    print("Evaluating boundary consistency for corrected data...")
    corr_results = boundary_consistency_across_resolutions(
        adata_corrected, return_visualizations=False, **kwargs)
    
    # Compare key metrics
    metrics = {
        'Mean boundary consistency': (
            np.nanmean(orig_results['adjacent_consistency']), 
            np.nanmean(corr_results['adjacent_consistency'])
        ),
        'Optimal resolution': (
            orig_results['optimal_resolution'], 
            corr_results['optimal_resolution']
        )
    }
    
    # Add image alignment comparison if available
    if orig_results['image_alignment_scores'] is not None and corr_results['image_alignment_scores'] is not None:
        metrics['Mean image alignment'] = (
            np.nanmean(orig_results['image_alignment_scores']),
            np.nanmean(corr_results['image_alignment_scores'])
        )
    
    # Print comparison table
    print("\nBoundary Consistency Comparison:")
    print(f"{'Metric':<25} {'Original':>15} {'Corrected':>15} {'Change':>15}")
    print("-" * 70)
    
    for metric, (orig_val, corr_val) in metrics.items():
        if isinstance(orig_val, (int, float)) and isinstance(corr_val, (int, float)):
            change = corr_val - orig_val
            change_pct = (corr_val / orig_val - 1) * 100 if orig_val != 0 else float('inf')
            change_str = f"{change:.3f} ({change_pct:+.1f}%)"
            print(f"{metric:<25} {orig_val:>15.3f} {corr_val:>15.3f} {change_str:>15}")
        else:
            print(f"{metric:<25} {orig_val:>15} {corr_val:>15}")
    
    # Create comparison plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Boundary consistency comparison
    axs[0, 0].plot(orig_results['resolutions'], orig_results['adjacent_consistency'], '-o', label='Original')
    axs[0, 0].plot(corr_results['resolutions'], corr_results['adjacent_consistency'], '-o', label='Corrected')
    axs[0, 0].axvline(orig_results['optimal_resolution'], color='blue', linestyle='--',
                     label=f"Orig optimal: {orig_results['optimal_resolution']:.2f}")
    axs[0, 0].axvline(corr_results['optimal_resolution'], color='orange', linestyle='--',
                     label=f"Corr optimal: {corr_results['optimal_resolution']:.2f}")
    axs[0, 0].set_xlabel('Resolution')
    axs[0, 0].set_ylabel('Boundary consistency')
    axs[0, 0].set_title('Boundary consistency across resolutions')
    axs[0, 0].legend()
    
    # Plot 2: Image alignment comparison if available
    if orig_results['image_alignment_scores'] is not None and corr_results['image_alignment_scores'] is not None:
        axs[0, 1].plot(orig_results['resolutions'], orig_results['image_alignment_scores'], '-o', label='Original')
        axs[0, 1].plot(corr_results['resolutions'], corr_results['image_alignment_scores'], '-o', label='Corrected')
        axs[0, 1].set_xlabel('Resolution')
        axs[0, 1].set_ylabel('Image alignment score')
        axs[0, 1].set_title('Cluster boundaries alignment with tissue features')
        axs[0, 1].legend()
    else:
        axs[0, 1].text(0.5, 0.5, "No tissue image features available", 
                      ha='center', va='center')
    
    # Plot 3: Boundary overlap difference
    overlap_diff = corr_results['boundary_overlap_matrix'] - orig_results['boundary_overlap_matrix']
    im = axs[1, 0].imshow(overlap_diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axs[1, 0].set_title('Boundary overlap difference (corrected - original)')
    axs[1, 0].set_xlabel('Resolution index')
    axs[1, 0].set_ylabel('Resolution index')
    plt.colorbar(im, ax=axs[1, 0])
    
    # Plot 4: Visualization at optimal resolution for both datasets
    lib_id = list(adata_orig.uns['spatial'].keys())[0] if 'spatial' in adata_orig.uns else None
    
    if lib_id is not None:
        # Get the optimal resolutions
        opt_res_orig = orig_results['optimal_resolution']
        opt_res_corr = corr_results['optimal_resolution']
        
        # Create cluster map at optimal resolution
        import squidpy as sq
        
        # Create keys for optimal clustering
        sc.tl.leiden(adata_orig, resolution=opt_res_orig, key_added='optimal_clusters_orig')
        sc.tl.leiden(adata_corrected, resolution=opt_res_corr, key_added='optimal_clusters_corr')
        
        # Plot spatial cluster maps side by side
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, width_ratios=[1, 1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        sc.pl.spatial(adata_orig, color='optimal_clusters_orig', ax=ax1, show=False, 
                      title=f'Original (res={opt_res_orig:.2f})')
        sc.pl.spatial(adata_corrected, color='optimal_clusters_corr', ax=ax2, show=False,
                      title=f'Corrected (res={opt_res_corr:.2f})')
    
    plt.tight_layout()
    
    return {
        'original': orig_results,
        'corrected': corr_results,
        'comparison': metrics,
        'visualization': fig
    }