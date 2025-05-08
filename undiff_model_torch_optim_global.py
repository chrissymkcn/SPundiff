from undiff_model_torch_optim import undiff
import numpy as np
import torch
import torch.optim as optim
import time

class undiff_global(undiff):
    def __init__(self, adata, n_jobs=1, metric='euclidean', img_key='hires', optimizer='adam', optim_params=None):
        super().__init__(adata, n_jobs, metric, img_key, optimizer, optim_params)

    def smooth_quantile_batch(self, x, q):
        """Differentiable approximation of quantile"""
        sorted_x, _ = torch.sort(x, dim=0)
        n = x.shape[0]
        index = (n - 1) * q  # q ∈ [0,1]

        # Linear interpolation between adjacent values
        lower = torch.floor(index).long()
        upper = lower + 1
        weight = index - lower

        # For out-of-bound cases
        upper = torch.clamp(upper, 0, n-1)
        x_qvals = (1 - weight) * sorted_x[lower, torch.arange(sorted_x.size(1))] + weight * sorted_x[upper, torch.arange(sorted_x.size(1))] 
        return x_qvals


    def gene_specific_adaptation(self, out_dist, in_dist, out_sum, in_sum):
        # Get gene-specific distributions
        global_ot = self.global_ot  # [n_spots, n_spots]    
        ## assume out_dist and in_dist are already normalized to sum to 1
        # Compute scaling factors
        row_scale = out_dist / global_ot.sum(1)
        col_scale = in_dist / global_ot.sum(0)
        
        # Compute adapted transport (no need for full matrix)
        to_target = (global_ot * row_scale.unsqueeze(1) * col_scale.unsqueeze(0)).sum(0)
        to_target = to_target / to_target.sum()
        to_return = to_target * out_sum + in_dist * in_sum
        return to_return  # [n_spots, 1]

    def gene_specific_adaptation_reg(self, gene_expr_out, gene_expr_in, out_sum, in_sum, reg):
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
        eps = 1e-10
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
            # transported_in = self.gene_specific_adaptation(g_out, g_in, g_outsum, g_insum)
            transported_in = self.gene_specific_adaptation_reg(g_out, g_in, g_outsum, g_insum, regs[i])
            res.append(transported_in)
        self.res_count = torch.stack(res, dim=1)
 
    def run_one_round_ot(self, params):
        cost_weight_s = params['cost_weight_s']
        cost_weight_t = params['cost_weight_t']
        invalid_qts = params['invalid_qts']
        regs = params['regs']
        global_reg = params['global_reg']

        # Process genes with checkpointing
        # self.cost_matrix_calculation(cost_weight_s, cost_weight_s) # for computing scaled cost matrix
        self.cost_matrix_calculation(cost_weight_s, cost_weight_t) # for computing scaled cost matrix
        self.compute_shared_OT(global_reg) # for updating warmstart
        # Use checkpoint for memory-efficient OT computation
        self.compute_ot_batch(invalid_qts, regs)
        
    def run_one_round(self, first_n_genes=None, add_genes=[], optim_params=None):
        self.params.update(optim_params) if optim_params is not None else None
        self.prep_genes_params(add_genes=add_genes, first_n_genes=first_n_genes)
        self.set_states(qts_prior=0.3)
        # test case 
        resdict = self.run_one_round_ot({
            'cost_weight_s': self.cost_weight_s,
            'cost_weight_t': self.cost_weight_t,
            'invalid_qts': self.invalid_qts,
            'regs': self.regs,
            'global_reg': self.global_reg
        })
        return resdict
        
