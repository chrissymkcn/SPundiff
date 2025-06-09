from undiff_model_torch_optim import undiff
import torch
import pyro.distributions as dist

class undiff_global(undiff):
    def __init__(self, adata, n_jobs=1, metric='euclidean',optimizer='adam', optim_params=None):
        super().__init__(adata, n_jobs, metric, optimizer, optim_params)

    def gene_specific_adaptation(self, gene_expr_out, gene_expr_in, out_sum, in_sum, reg):
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
            transported_in = self.gene_specific_adaptation(g_out, g_in, g_outsum, g_insum, regs[i])
            res.append(transported_in)
        self.res_count = torch.stack(res, dim=1)

    def run_ot(self, params):
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
        self.compute_res_count({
            'regs': regs,
            'invalid_qts': invalid_qts
        })
        
    def run_one_round(self, n_genes=None, add_genes=[], optim_params=None):
        self.params.update(optim_params) if optim_params is not None else None
        self.prep_genes_params(add_genes=add_genes, first_n_genes=n_genes)
        self.set_states(qts_prior=0.3)
        self.run_ot({
            'cost_weight_s': self.cost_weight_s,
            'cost_weight_t': self.cost_weight_t,
            'invalid_qts': self.invalid_qts,
            'regs': self.regs,
            'global_reg': self.global_reg
        })
        
