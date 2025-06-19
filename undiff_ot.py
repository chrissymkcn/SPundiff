from .base import base
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import ot
import kornia.losses as kornia_losses
import kornia.filters as kornia_filters
import time
from scipy.sparse import issparse
import kornia.losses as kornia_losses
import kornia.filters as kornia_filters
import torch.optim as optim
import time

class DynamicLossBalancer:
    def __init__(self, num_losses=2):
        # Initialize learnable log variances
        self.log_vars = torch.nn.Parameter(torch.zeros(num_losses))
        
    def __call__(self, losses):
        # Convert to precision (inverse variance)
        precision = torch.exp(-self.log_vars)
        
        # Weighted loss + regularization
        weighted_losses = precision * losses + self.log_vars
        return torch.sum(weighted_losses)

class undiff_ot(base):
    DEFAULT_OT_PARAMS = {
        # OT parameters
        'mass': 1,
        'nb_dummies': 1,
        'to_one': True,
        'ot_num_iter': 2000,
        'ot_stopThr': 1e-9,
        'ot_tau': 1e3,
        'ot_method': 'sinkhorn',
        'sinkhorn_method': 'sinkhorn',
        'scale_cost': 'by_max',
        # 'reg': 10,
        'reg_m_kl': (0.1, float('inf')),
        'reg_m_l2': (5, float('inf')),
        'reg_m_entropy': (0.1, float('inf')),
        'reg_m_lag': 0.1,
        'check_marginals': False,
        'div': 'kl',
        # Optimization parameters
        'train_lr': 1e-3,
        'train_n_epochs': 10,
        'train_tol': 1e-9,
    }
    def __init__(self, adata, metric='euclidean',
                    optimizer='adam', optim_params=None):
        super().__init__(adata, metric=metric, 
                         optimizer=optimizer, optim_params=optim_params)
        # Initialize
        self.objective_values = []  # Global list to store objective values
        self.optimizer = optimizer
        self.params = self.DEFAULT_OT_PARAMS.copy()
        if optim_params is not None:
            self.params.update(optim_params)
    
    def calc_loss(self):
        # loss from image alignment score, higher is better
        image_losses = self.eval_image_alignment_loss()
        print('Image alignment loss:', image_losses, image_losses['ssim'].requires_grad)
        
        # # loss from out-tissue spot expression
        out_tiss_mask = self.in_tiss_mask.float() * torch.tensor([-1], dtype=torch.float32) + 1.0
        restored_out_tiss = out_tiss_mask @ self.res_count  # want this to be 0
        restored_out_tiss_err = restored_out_tiss.sum()
        print('restored_out_tiss loss:', restored_out_tiss_err, restored_out_tiss_err.requires_grad)

        # calculate moranI difference, higher is better, so we use the negative value
        moranI_loss = self.eval_moranI()
        print('MoranI loss:', moranI_loss, moranI_loss.requires_grad)

        # smoothness_loss = self.eval_spatial_smoothness_penalty(penalty_weight=0.1, eps=1e-8)
        # print('smoothness loss:', smoothness_loss, smoothness_loss.requires_grad)
        
        weights = {
            # 'ssim': 1,       # Strongest negative correlation
            # 'psnr': 1,
            # 'ms_ssim': 1,
            # 'ncc': 1,
            # 'mi': -1,
            # 'gms': 1,
            # 'attention_weighted': 1,
            'moranI': 1,
            'restored_out_tiss': 1,
            # 'smoothness': 1,
        }
        error = {k: v * weights[k] for k, v in image_losses.items() if k in weights} # use weights to control which losses to use
        error['moranI'] = moranI_loss * weights['moranI']
        error['restored_out_tiss'] = restored_out_tiss_err * weights['restored_out_tiss']
        # error['smoothness'] = smoothness_loss * weights['smoothness']
        return error
    
    @staticmethod    
    def compute_gradients(loss, params):
        return torch.autograd.grad(loss, params, retain_graph=True)
    
    @staticmethod
    def balance_weights(loss_dict):
        # Step 1: Compute inverse of absolute loss magnitudes
        raw_weights = {k: 1.0 / abs(v.detach()) for k, v in loss_dict.items()}
        # Step 2: Normalize to sum=1
        total = sum(raw_weights.values())
        return {k: v / total for k, v in raw_weights.items()}

    def compute_res_count(self, params):
        genes = self.gene_selected
        regs = params['regs']
        invalid_qts = params['invalid_qts']
        out_tiss_filt, in_tiss_filt, out_tiss_sum, in_tiss_sum = self.prep(invalid_qts)
        res = []
        batch_size = 10
        for i, gene in enumerate(genes):
            if i % batch_size == 0:
                print(f'Processing gene {i+1}/{len(genes)}: {gene}')
            # Use checkpoint for memory-efficient OT computation
            g_out, g_in = out_tiss_filt[:, i], in_tiss_filt[:, i]
            g_outsum, g_insum = out_tiss_sum[i], in_tiss_sum[i]
            res = self.compute_ot(out_tiss_filt=g_out,
                                        in_tiss_filt=g_in,
                                        out_tiss_sum=g_outsum,
                                        in_tiss_sum=g_insum,
                                        reg=regs[i])
            # print('OT result:', resdict)
            res.append(res)
            torch.cuda.empty_cache()
        self.res_count = torch.stack(res, dim=1)            

    def objective(self, params):
        """
        Objective function with gradient checkpointing to reduce memory usage.
        """
        cost_weight_s = params['cost_weight_s']
        cost_weight_t = params['cost_weight_t']
        global_reg = params['global_reg']
        
        # Process genes with checkpointing
        self.scaled_cost_matrix_calculation(cost_weight_s, cost_weight_t) # for computing scaled cost matrix
        self.compute_shared_OT(global_reg) # for updating warmstart
        self.compute_res_count(params)
        self.last_z = self.res_count.detach().clone()
        
        # Compute losses (this part doesn't need checkpointing as it's not the memory bottleneck)
        loss_dict = self.calc_loss()
        # loss_w = undiff.balance_weights(loss_dict)
        loss_w = {k:1 for k, v in loss_dict.items()}
        loss_d = {k:loss_w[k] * v for k, v in loss_dict.items()}
        print('loss_d', loss_d)
        loss = sum(loss_d.values())
        
        # # Store the error value
        # self.objective_values.append(loss)

        return torch.stack(list(loss_d.values()))

    def set_states(self, qts_prior):
        super(unddiff, self).set_states(qts_prior=qts_prior)
        self.objective_values = []
        self.last_z = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cost_weight_s = torch.ones(
                self.coords.shape[0], 
                dtype=torch.float32, 
                requires_grad=True,
                device=device
            )
        self.cost_weight_t = torch.ones(
                self.coords.shape[0],
                dtype=torch.float32,
                requires_grad=True,
                device=device
            )
        self.regs = torch.tensor([6.0] * self.sub_count.shape[1], requires_grad=True, dtype=torch.float32, device=device)
        self.global_reg = torch.tensor(1.0, requires_grad=True, dtype=torch.float32, device=device)
        
    def optimization(self, first_n_genes=None, add_genes=[], qts_prior=0.4, optim_params=None, optim_name=None):
        """
        Optimize the objective function using the given optimizer.
        """
        self.params.update(optim_params) if optim_params is not None else None
        params = self.params
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.prep_genes_params(add_genes=add_genes, first_n_genes=first_n_genes)
        self.set_states(qts_prior=qts_prior)
        
        # Define optimizer
        lr = params['train_lr']
        optim_name = optim_name if optim_name is not None else self.optimizer
        if optim_name == 'adam':
            optimizer = optim.Adam(
                params=[
                    {'params': self.cost_weight_s, 'lr': lr},
                    {'params': self.cost_weight_t, 'lr': lr},
                    {'params': self.invalid_qts, 'lr': lr},
                    {'params': self.regs, 'lr': lr},
                    {'params': self.global_reg, 'lr': lr}
                ],
            )
        elif optim_name == 'lbfgs':
            optimizer = optim.LBFGS(
                params=[
                    {'params': self.invalid_qts, 'lr': lr},
                    {'params': self.regs, 'lr': lr},
                    {'params': self.global_reg, 'lr': lr},
                    {'params': self.cost_weight_s, 'lr': lr},
                    {'params': self.cost_weight_t, 'lr': lr},
                ],
            )
        
        # Training loop
        for epoch in range(self.params['train_n_epochs']):
            print(f'Epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss = self.objective(
                {
                    'invalid_qts': self.invalid_qts,
                    'regs': self.regs,
                    'global_reg': self.global_reg,
                    'cost_weight_s': self.cost_weight_s,
                    'cost_weight_t': self.cost_weight_t,
                }
            )
            balancer = DynamicLossBalancer()
            dw_loss = balancer(loss)
            print(f"Balanced Loss: {dw_loss.item()}")
            time_start = time.time()
            dw_loss.backward()
            print(f"Backward pass time cost: {time.time() - time_start}s")
            assert self.cost_weight_s.grad is not None, "Gradients not reaching cost_weight_s!"
            assert self.cost_weight_t.grad is not None, "Gradients not reaching cost_weight_t!"
            assert self.regs.grad is not None, "Gradients not reaching regs!"
            assert self.invalid_qts.grad is not None, "Gradients not reaching invalid_qts!"
            assert self.global_reg.grad is not None, "Gradients not reaching global_reg!"
            print('Gradient calculated, magnitudes for cost_weight_s, cost_weight_t, regs, invalid_qts, global_reg:')
            print(self.cost_weight_s.grad.norm(),
                self.cost_weight_t.grad.norm(),
                self.regs.grad.norm(), 
                self.invalid_qts.grad.norm(), 
                self.global_reg.grad.norm())
            if epoch > 0:
                grad_norm = torch.cat([
                                    p.grad.flatten() 
                                    for group in optimizer.param_groups 
                                    for p in group['params']
                                    if p.grad is not None  # Handle None gradients
                                ]).norm()
                if grad_norm < self.params['train_tol']:
                    break
            start = time.time()
            optimizer.step()
            print(f"Optimizer Step time cost: {time.time() - start}s")
            
        res = {
            'cost_weight_s': self.cost_weight_s.detach().cpu().numpy(),
            'cost_weight_t': self.cost_weight_t.detach().cpu().numpy(),
            'invalid_qts': self.invalid_qts.detach().cpu().numpy(),
            'regs': self.regs.detach().cpu().numpy(),
            'global_reg': self.global_reg.detach().cpu().numpy(),
            'res_count': self.res_count.detach().cpu().numpy(),
        }
        return res

    def gene_sim_scale_cost(self, base_distance, diffusion_coefficients):
        """
        base_distance: (nsource, ntarget) basic distance matrix
        diffusion_coefficients: (nspots,) gene-specific diffusion parameters
        """
        # Compute distance between genes (assuming correlated genes diffuse similarly)
        if 'X_pca' not in self.adata.obsm.keys():
            sc.tl.pca(self.adata)            
        gene_expression = self.adata.obsm['X_pca'].toarray() if issparse(self.adata.obsm['X_pca']) else self.adata.obsm['X_pca']
        gene_expression = torch.tensor(gene_expression.copy())
        # Compute expression similarity between spots
        spot_sim = torch.cdist(gene_expression, gene_expression) 
        spot_sim = self.scale_to_range([base_distance.min(), base_distance.max()], spot_sim)
        # Modulate distance by expression similarity and gene diffusion properties
        return base_distance * diffusion_coefficients * spot_sim

    def scale_to_range(self, new_range, data):
        ttl_min = new_range[0]
        ttl_max = new_range[1]
        r = ttl_max - ttl_min
        cost_min = data.min()
        cost_max = data.max()
        scaled = ((data - cost_min) / (cost_max - cost_min)) * r + ttl_min
        return scaled

    def ssim_loss(self, pred_img, target_img, window_size=11):
        return kornia_losses.ssim_loss(
            pred_img, 
            target_img, 
            window_size=window_size, 
            reduction='mean'
        )
    
    def psnr_loss(self, pred_img, target_img, max_val=1.0):
        return kornia_losses.psnr_loss(pred_img, target_img, max_val=max_val)
    
    def charbonnier_loss(self, pred_img, target_img):
        return kornia_losses.charbonnier_loss(pred_img, target_img)

    def multiscale_gradient_loss(self, pred_img, target_img, scales=3):
        loss = 0.0
        for _ in range(scales):
            loss += self.gms_loss(pred_img, target_img)
            pred_img = kornia_filters.gaussian_blur2d(pred_img, (3,3), (1.5,1.5))
            target_img = kornia_filters.gaussian_blur2d(target_img, (3,3), (1.5,1.5))
        return loss / scales
        
    def attention_weighted_loss(self, pred_img, target_img):
        saliency_map = kornia_filters.laplacian(target_img, kernel_size=5)
        return torch.mean(saliency_map * (pred_img - target_img)**2)

    def ms_ssim_loss(self, pred, target):
        criterion = kornia_losses.MS_SSIMLoss()
        return criterion(pred, target)
    
    def gms_loss(self, pred_img, target_img):
        pred_grad = kornia_filters.spatial_gradient(pred_img)
        target_grad = kornia_filters.spatial_gradient(target_img)
        return torch.abs(pred_grad - target_grad).mean()
    
    def mutual_information_loss(self, pred_img, target_img, bins=32):
        hist_2d = torch.histogramdd(
            torch.stack([pred_img.flatten(), target_img.flatten()], dim=1),
            bins=bins
        )[0].float()
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(dim=1)
        py = pxy.sum(dim=0)
        mi = torch.sum(pxy * (torch.log(pxy + 1e-10) - torch.log(px.unsqueeze(1) + 1e-10) - torch.log(py.unsqueeze(0) + 1e-10)))
        return -mi  # Maximize MI by minimizing negative MI
    
    def ncc_loss(self, pred_img, target_img):
        pred_norm = (pred_img - pred_img.mean()) / (pred_img.std() + 1e-10)
        target_norm = (target_img - target_img.mean()) / (target_img.std() + 1e-10)
        return -torch.mean(pred_norm * target_norm)  # Maximize correlation
    
    def histogram_loss(self, pred_img, target_img, bins=32):
        pred_hist = torch.histc(pred_img, bins=bins)
        target_hist = torch.histc(target_img, bins=bins)
        return torch.abs(pred_hist - target_hist).mean()
    
    def tv_loss(self, pred_img):
        return kornia_losses.total_variation(pred_img)
    
    def jsd_loss(self, pred_img, target_img):
        return kornia_losses.js_div_loss_2d(
            pred_img, 
            target_img, 
        )
    
    def kl_loss(self, pred_img, target_img):
        kl_loss = kornia_losses.kl_div_loss_2d(
            pred_img, 
            target_img, 
        )
        return kl_loss.mean()
        
    def eval_image_alignment_loss(self):
        tissue_img = self.get_image_embedding()
        pred_img = self.get_n_cnt_embedding(cnt_slot='res_count')
        # check if the image has white background
        background_mean = tissue_img[tissue_img > 0.1].mean()  # if background is dark, average will be low
        if background_mean > 0.5:
            tissue_img = 1 - tissue_img  # Invert the image if white background
        # Add batch and channel dims (B, C, H, W)
        pred_img = pred_img.unsqueeze(0).unsqueeze(0)
        tissue_img = tissue_img.unsqueeze(0).unsqueeze(0)

        # Compute individual losses
        losses = {
            'ssim': self.ssim_loss(pred_img, tissue_img),
            'gms': self.gms_loss(pred_img, tissue_img),
            # 'mi': self.mutual_information_loss(pred_img, tissue_img),
            'ncc': self.ncc_loss(pred_img, tissue_img),
            # 'histogram': self.histogram_loss(pred_img, tissue_img),
            # 'tv': self.tv_loss(pred_img)
        }
        losses.update({
            'psnr': self.psnr_loss(pred_img, tissue_img),
            'multiscale_grad': self.multiscale_gradient_loss(pred_img, tissue_img),
            # 'charbonnier': self.charbonnier_loss(pred_img, tissue_img),
            'attention_weighted': self.attention_weighted_loss(pred_img, tissue_img),
            'ms_ssim': self.ms_ssim_loss(pred_img, tissue_img),
            # 'jsd': self.jsd_loss(pred_img, tissue_img),
            # 'kl': self.kl_loss(pred_img, tissue_img),
        })
        
        return losses

    def smooth_quantile_batch(self, x, q):
        """向量化的分位数计算 (行→基因方向)"""
        sorted_x, _ = torch.sort(x, dim=0)  # 按列排序
        n_samples = x.shape[0]
        
        # 索引位置计算 [n_genes]
        index = (n_samples - 1) * q.to(x.device)  # 确保q在相同设备上
        lower_idx = torch.clamp(torch.floor(index).long(), min=0, max=n_samples-1)
        upper_idx = torch.clamp(lower_idx + 1, max=n_samples-1)
        
        # 通过gather获取具体值 [n_genes]
        lower_values = sorted_x[lower_idx, torch.arange(x.size(1), device=x.device)]
        upper_values = sorted_x[upper_idx, torch.arange(x.size(1), device=x.device)]
        
        return (upper_idx.float() - index) * lower_values + (index - lower_idx.float()) * upper_values

    def run_selected_ot(self, out_tiss, in_tiss_filt, scaled_cost, reg=None, mass=None, warmstart=None, log=False):
        def update_ab(ot_emd):
            log_r = ot_emd[1]
            self.alpha_prev = log_r['u'].detach().clone()
            self.beta_prev = log_r['v'].detach().clone()
        params = self.params
        log = log
        if params['ot_method'] is None or params['ot_method']=='emd':
            ot_emd = ot.emd(out_tiss, in_tiss_filt, scaled_cost, check_marginals=params['check_marginals'])
        elif params['ot_method'] == 'sinkhorn':
            sinkhorn_method = params['sinkhorn_method']
            if sinkhorn_method not in ['sinkhorn', 'sinkhorn_log', 'greenkhorn', 'sinkhorn_stabilized', 'sinkhorn_epsilon_scaling']:
                sinkhorn_method = 'sinkhorn'
            ot_emd = ot.sinkhorn(out_tiss, in_tiss_filt, scaled_cost, reg=reg, method=sinkhorn_method,
                                        warmstart=warmstart, 
                                        log=log)
            if log:
                update_ab(ot_emd)
                ot_emd = ot_emd[0]
        elif params['ot_method'] == 'greenkhorn':
            ot_emd = ot.bregman.greenkhorn(out_tiss, in_tiss_filt, scaled_cost, reg=reg)
        elif params['ot_method'] == 'screenkhorn':
            ot_emd = ot.bregman.screenkhorn(out_tiss, in_tiss_filt, scaled_cost, reg=reg, log=log)
            if log:
                update_ab(ot_emd)
                ot_emd = ot_emd[0]
        elif params['ot_method'] == 'sinkhorn_epsilon_scaling':
            ot_emd = ot.bregman.sinkhorn_epsilon_scaling(out_tiss, in_tiss_filt, scaled_cost, reg=reg)
        elif params['ot_method'] == 'sinkhorn_stabilized':
            # print('Calculating Sinkhorn stabilized')
            ot_emd = ot.bregman.sinkhorn_stabilized(out_tiss, in_tiss_filt, 
                                                    scaled_cost, reg=reg, 
                                                    stopThr=params['ot_stopThr'],
                                                    numItermax=params['ot_num_iter'], 
                                                    tau=params['ot_tau'], log=log, 
                                                    # warmstart=warmstart
                                                    )
            if log:
                update_ab(ot_emd)
                ot_emd = ot_emd[0]
            # print('Sinkhorn stabilized calculated')
        elif params['ot_method'] == 'sinkhorn_unbalanced':
            if div not in ['kl', 'entropy']: div = 'entropy'
            reg_m = params['reg_m_kl'] if div == 'kl' else params['reg_m_entropy']
            ot_emd = ot.sinkhorn_unbalanced(out_tiss, in_tiss_filt, scaled_cost, reg, reg_m=reg_m,
                                            method=sinkhorn_method, reg_type=div)
        elif params['ot_method'] == 'lbfgsb_unbalanced':
            if div == 'kl': reg_m = params['reg_m_kl']
            elif div == 'l2': reg_m = params['reg_m_l2']
            elif div == 'entropy': reg_m = params['reg_m_entropy']
            else: 
                div = 'kl'
                reg_m = params['reg_m_kl']
            ot_emd = ot.unbalanced.lbfgsb_unbalanced(out_tiss, in_tiss_filt, scaled_cost, 
                                                    reg=reg, reg_m=reg_m, 
                                                    reg_div=div, log=log)
        elif params['ot_method'] == 'mm_unbalanced':
            if div not in ['kl', 'l2']: div = 'kl'
            reg_m = params['reg_m_kl'] if div == 'kl' else params['reg_m_l2']
            ot_emd = ot.unbalanced.mm_unbalanced(out_tiss, in_tiss_filt, scaled_cost, reg_m=reg_m, div=div)
        elif params['ot_method'] == 'partial':
            ot_emd = ot.partial.partial_wasserstein(out_tiss, in_tiss_filt, scaled_cost, 
                                                    m=mass, nb_dummies=params['nb_dummies'], check_marginals=params['check_marginals'])
        elif params['ot_method'] == 'entropic_partial':
            ot_emd = ot.partial.entropic_partial_wasserstein(out_tiss, in_tiss_filt, scaled_cost, reg, m=mass)
        else:
            raise ValueError('method not recognized')
        return ot_emd
    
    def sinkhorn_stabilized_torch(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        M: torch.Tensor,
        reg: float,
        numItermax: int = 1000,
        tau: float = 1e3,
        stopThr: float = 1e-9,
        warmstart: tuple = None,
        log: bool = False,
        warn: bool = True,
    ) -> torch.Tensor:
        """
        Autograd-compatible Sinkhorn with full gradient tracking.
        Maintains numerical stability while preserving computation graph.
        """
        
        device, dtype = M.device, M.dtype
        dim_a, dim_b = M.shape
        # Initialize shared alpha/beta if they don't exist
        if self.alpha_prev is None or self.beta_prev is None:
            self.alpha_prev = torch.zeros(dim_a, device=device, dtype=dtype)
            self.beta_prev = torch.zeros(dim_b, device=device, dtype=dtype)
        
        if warmstart is None and self.alpha_prev is not None and self.beta_prev is not None:
            warmstart = (self.alpha_prev, self.beta_prev)
        
        # Initialize and normalize distributions with gradient tracking
        a = a / a.sum().clamp(min=1e-15)
        b = b / b.sum().clamp(min=1e-15)

        # Initialize dual variables
        if warmstart is None:
            alpha = torch.zeros_like(a)
            beta = torch.zeros_like(b)
        else:
            alpha, beta = warmstart
        
        u = torch.ones_like(a) / dim_a
        v = torch.ones_like(b) / dim_b

        # Main loop with gradient tracking
        for ii in range(numItermax):
            # Compute kernel matrix with gradient tracking
            K = torch.exp(-(M - alpha.unsqueeze(1) - beta.unsqueeze(0)) / reg + 1e-15)
            
            # Sinkhorn updates with gradient tracking
            v = b / (K.T @ u + 1e-15)
            u = a / (K @ v + 1e-15)

            # Stabilization mechanism with gradient tracking
            max_u = u.max()  # Detach only for condition check
            max_v = v.max()
            mask = (max_u > tau) | (max_v > tau)
            mask_float = mask.float()

            alpha = alpha + mask_float * reg * torch.log(u + 1e-15)
            beta = beta + mask_float * reg * torch.log(v + 1e-15)
            
            u = (1 - mask_float) * u + mask_float * torch.ones_like(u)/dim_a
            v = (1 - mask_float) * v + mask_float * torch.ones_like(v)/dim_b
            
        # Final transport matrix with gradient tracking
        K = torch.exp(-(M - alpha.unsqueeze(1) - beta.unsqueeze(0)) / reg + 1e-15)
        Gamma = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        self.alpha_prev = alpha.detach().clone()
        self.beta_prev = beta.detach().clone()
        
        # # Optional convergence check (doesn't affect gradients)
        # with torch.no_grad():
        #     final_err = 0.5 * (
        #         torch.norm(Gamma.sum(1) - a, p=1) + 
        #         torch.norm(Gamma.sum(0) - b, p=1)
        #     )
        #     if warn and final_err > stopThr:
        #         warnings.warn(f"Final error {final_err.item():.2e} > threshold {stopThr}")

        return Gamma

    def compute_shared_OT(self, global_reg):
        params = self.params
        reg = global_reg
        # use total expression counts to compute the global ot
        ttl_cnts = self.sub_count.sum(dim=1)  # selected genes
        eps = 1e-10
        total_source = ttl_cnts * (self.in_tiss_mask == 0).float()
        total_source = (total_source + eps) / total_source.sum()
        total_target = ttl_cnts * (self.in_tiss_mask == 1).float()
        total_target = (total_target + eps) / total_target.sum()
        scaled_cost = self.scaled_cost
        # run OT to get the warmstart vectors
        self.global_ot = self.run_selected_ot(
            total_source,
            total_target, 
            scaled_cost,
            reg=reg,
            warmstart=None,
            log=True
        )

    def scaled_cost_matrix_calculation(self, cost_weight_s, cost_weight_t):
        base_cost = self.coord_cost
        params = self.params
        diff_coeff = torch.outer(cost_weight_s, cost_weight_t)
        adjusted_cost = self.gene_sim_scale_cost(base_cost, diff_coeff)
        self.scaled_cost = self.scaling(adjusted_cost, params['scale_cost'])

    def compute_ot(self, out_tiss_filt, in_tiss_filt, out_tiss_sum, in_tiss_sum, reg):
        params = self.params
        scaled_cost = self.scaled_cost
        # print("Running OT...")
        time_start = time.time()
        warmstart = (self.alpha_prev, self.beta_prev) if self.alpha_prev is not None and self.beta_prev is not None else None
        # warmstart = None
        if warmstart is not None:
            print("Using warmstart")
        ot_emd = self.run_selected_ot(out_tiss=out_tiss_filt, 
                                    in_tiss_filt=in_tiss_filt, 
                                    scaled_cost=scaled_cost, 
                                    warmstart=warmstart,
                                    reg=reg,
                                    log=False
                                    )
        print(f"OT completed in {time.time() - time_start:.2f} seconds")
        to_target = ot_emd.sum(axis=0)  # Sum over axis 0 (rows) gives mass transported to target spots
        # from_source = ot_emd.sum(axis=1)  # Sum over axis 1 (columns) gives mass transported from source spots
        transported_in = to_target * out_tiss_sum + in_tiss_filt * in_tiss_sum    
        return transported_in.unsqueeze(1)  # Add a new dimension for broadcasting

    def adata_processing(self):
        # check if the data is integer
        if self.adata.X.data.dtype in [np.int32, torch.int64, torch.uint32, torch.uint64, int]:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, flavor="seurat", n_top_genes=2000)
        sc.pp.scale(self.adata, max_value=10)
        sc.pp.pca(self.adata)
        sc.pp.neighbors(self.adata, n_pcs=30, n_neighbors=30)

    def get_image_embedding(self, lib_id=None):
        lib_id = self.lib_id if lib_id is None else lib_id
        img = self.adata.uns['spatial'][lib_id]['images'][self.img_key]
        img = torch.tensor(img, dtype=torch.float32, requires_grad=True)
        # convert to grayscale 
        if img.shape[2] == 3:
            img = torch.matmul(img[...,:3], torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32))
        img_normed = (img - img.min()) / (img.max() - img.min())
        return img_normed
        
    def get_n_cnt_embedding(self, cnt_slot='res_count'):
        # Reshape expression to match image dimensions
        # gene_indices = self.gene_indices
        if cnt_slot == 'res_count':
            z = self.res_count
        elif cnt_slot == 'sub_count':
            z = self.sub_count
        else:
            raise ValueError("Invalid count slot. Choose 'res_count' or 'sub_count'.")
        z_sum = z.sum(dim=1, keepdim=True)  # [n_spots, 1]
        z_field = self.orig_coord_embedding(z_sum)  # [height, width]
        z_field_normed = (z_field - z_field.min()) / (z_field.max() - z_field.min())
        return z_field_normed
        
    def eval_moranI(
        self,
        normalize_weights: bool = True,
    ) -> torch.Tensor:
        """
        Compute Moran's I for each gene in a differentiable way using PyTorch.
        
        Args:
            adata: AnnData object with precomputed spatial graph
            expression_tensor: Tensor of shape [n_cells, n_genes]
            connectivity_key: Key in adata.obsp containing spatial weights
            normalize_weights: Whether to row-normalize the weight matrix
            
        Returns:
            morans_i: Tensor of Moran's I scores for each gene [n_genes]
        """
        # Get spatial weights matrix and convert to torch tensor
        expression_tensor = self.res_count
        device = expression_tensor.device
        W = self.spatial_con.toarray() if issparse(self.spatial_con) else self.spatial_con
        W = torch.tensor(W, dtype=torch.float32, device=expression_tensor.device)
        
        if normalize_weights:
            W = F.normalize(W, p=1, dim=1)  # Row normalize

        n = W.shape[0]
        S0 = W.sum()
        
        # Center the expression values (z-score)
        z = expression_tensor - expression_tensor.mean(dim=0, keepdim=True)
        
        # Compute denominator (sum of squared deviations)
        denom = (z ** 2).sum(dim=0)  # [n_genes]
        
        # Compute numerator (weighted covariance)
        # Wz = W @ z is more memory efficient than einsum for large matrices
        Wz = torch.mm(W, z)  # [n_cells, n_genes]
        numer = (z * Wz).sum(dim=0)  # [n_genes]
        
        # Moran's I calculation
        morans_i = (n / S0) * (numer / denom)
        
        return -morans_i.mean()

    def eval_spatial_smoothness_penalty(self, penalty_weight=0.1, eps=1e-8):
        """
        Computes a spatial smoothness penalty based on expression differences between neighbors.
        
        Uses:
        - self.res_count: [n_spots, n_genes] tensor of expression values
        - self.spatial_con: [n_spots, n_spots] binary adjacency matrix
        
        Returns:
        - Scalar penalty term weighted by penalty_weight
        """
        # Convert sparse matrix to dense if needed
        if isinstance(self.spatial_con, torch.Tensor):
            W = self.spatial_con
        else:
            W = torch.tensor(self.spatial_con.toarray(), 
                            dtype=torch.float32,
                            device=self.res_count.device)
        
        # Normalize adjacency matrix by number of neighbors
        neighbor_counts = W.sum(dim=1, keepdim=True)  # [n_spots, 1]
        norm_W = W / (neighbor_counts + eps)  # Row-normalized
        
        # Compute expression differences between neighbors
        # [n_spots, 1, n_genes] - [1, n_spots, n_genes] = [n_spots, n_spots, n_genes]
        expr_diff = self.res_count.unsqueeze(1) - self.res_count.unsqueeze(0)
        
        # Mask non-neighbor differences and compute mean absolute difference
        neighbor_diffs = expr_diff * norm_W.unsqueeze(-1)  # [n_spots, n_spots, n_genes]
        penalty = torch.mean(torch.abs(neighbor_diffs))  # L1 penalty
        
        return penalty_weight * penalty
    
    def get_undiff_adata(self, genes=None):
        tissmask = self.adata.obs['in_tissue'].astype(bool)
        genes = self.gene_selected + [g for g in genes if g not in self.gene_selected] if genes is not None else self.gene_selected
        adata_ot = self.adata[tissmask, genes].copy()
        adata_ot.X = self.res_count.detach().numpy()
        return adata_ot
    