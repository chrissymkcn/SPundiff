# credit: https://github.com/scverse/scanpy/issues/1643
import anndata2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import numpy as np

anndata2ri.activate()
pandas2ri.activate()

def run(adata, layer=None, **kwargs):
    if layer:
        mat = adata.layers[layer]
    else:
        mat = adata.X

    # Set names for the input matrix
    cell_names = adata.obs_names
    gene_names = adata.var_names
    r.assign('mat', mat.T)
    r.assign('cell_names', cell_names)
    r.assign('gene_names', gene_names)
    r('colnames(mat) <- cell_names')
    r('rownames(mat) <- gene_names')

    seurat = importr('Seurat')
    r('seurat_obj <- CreateSeuratObject(mat)')

    # Run
    for k, v in kwargs.items():
        r.assign(k, v)
    kwargs_str = ', '.join([f'{k}={k}' for k in kwargs.keys()])
    r(f'seurat_obj <- SCTransform(seurat_obj, vst.flavor="v2", {kwargs_str})')

    # Prevent partial SCT output because of default min.genes messing up layer addition
    r('diffDash <- setdiff(rownames(seurat_obj), rownames(mat))')
    r('diffDash <- gsub("-", "_", diffDash)')
    r('diffScore <- setdiff(rownames(mat), rownames(seurat_obj))')
    filtout_genes = list(r.setdiff(r('diffScore'), r('diffDash')))
    filtout_indicator = np.in1d(adata.var_names, filtout_genes)
    adata = adata[:, ~filtout_indicator]

    # Extract the SCT data and add it as a new layer in the original anndata object
    sct_data = np.asarray(r['as.matrix'](r('seurat_obj@assays$SCT@data')))
    adata.layers['SCT_data'] = sct_data.T
    sct_data = np.asarray(r['as.matrix'](r('seurat_obj@assays$SCT@counts')))
    adata.layers['SCT_counts'] = sct_data.T
    adata.uns['SCT'] = r('seurat_obj@assays$SCT@scale.data')
    adata.uns['SCT_var_features'] = r('seurat_obj@assays$SCT@var.features')
    return adata


# convert factor to character
i <- sapply(srt@meta.data, is.factor)
srt@meta.data[i] <- lapply(srt@meta.data[i], as.character)
DefaultAssay(srt) <- 'RNA' # set default assay
SaveH5Seurat(srt, filename = 'srt.h5seurat', overwrite = TRUE)
Convert('srt.h5seurat','srt.h5ad', assay='RNA', overwrite = TRUE)
