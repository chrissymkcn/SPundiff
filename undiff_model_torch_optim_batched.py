from undiff_model_torch_optim import undiff
import numpy as np
import torch.optim as optim
import torch
import torch.func as fc
from torch.cuda.amp import autocast

class undiff_batch(undiff):
    def __init__(self, adata, n_jobs=1, metric='euclidean', 
                img_key='hires', optimizer='adam', optim_params=None, batch_size=10):
        super().__init__(adata, n_jobs=n_jobs, metric=metric,
                        img_key=img_key, optimizer=optimizer, optim_params=optim_params)
        self.batch_size = batch_size
        self.transport_buffer = None  # Initialize in set_states()
        torch.cuda.set_per_process_memory_fraction(0.7)  # Limit GPU usage

    def set_states(self):
        super().set_states()
        self.transport_buffer = torch.empty_like(self.sub_count)        
        
    def _prepare_batch(self, genes):
        """Prepare batched inputs for OT"""
        gene_indices = torch.tensor(
            [self.adata.var_names.get_loc(g) for g in genes],
            device=self.sub_count.device
        )
        X_g = self.sub_count[:, gene_indices]
        
        # Batched prep
        out_tiss = torch.where(X_g < 1e-10, torch.tensor(1e-10, device=X_g.device), X_g)
        in_tiss = out_tiss.clone() * self.in_tiss_mask.unsqueeze(1)
        
        # Normalize
        out_sum = out_tiss.sum(dim=0, keepdim=True) + 1e-10
        in_sum = in_tiss.sum(dim=0, keepdim=True) + 1e-10
        return out_tiss/out_sum, in_tiss/in_sum

    def _build_cost_batch(self, cost_weights):
        """Build batched cost matrix with spot weights"""
        params = self.params
        adj_cost = self.coord_cost * (cost_weights.unsqueeze(0) + cost_weights.unsqueeze(1))
        return self.scaling(adj_cost, params['scale_cost'])  # Add batch dim

    @torch.compile(mode='reduce-overhead')  # Uses TorchInductor
    def batched_compute_ot(self, genes, cost_weights, invalid_qts, regs):
        """Fully parallelized OT computation using vmap"""
        if len(genes) <= 2:
            return self.sequential_ot(genes, cost_weights, invalid_qts, regs)        
        
        # Prepare batch
        out_tiss, in_tiss = self._prepare_batch(genes)
        cost_batch = self._build_cost_batch(cost_weights)
        
        # Apply soft thresholding
        print("in_tiss shape:", in_tiss.shape, in_tiss.requires_grad)
        print('out_tiss shape:', out_tiss.shape, out_tiss.requires_grad)
        cutoffs = torch.quantile(in_tiss, invalid_qts)
        mask = torch.sigmoid(50 * (in_tiss - cutoffs) / 
            (in_tiss.max(dim=0)[0] - in_tiss.min(dim=0)[0] + 1e-10))
        print("Mask shape:", mask.shape)
        in_tiss_filt = torch.where(in_tiss > cutoffs, in_tiss, in_tiss * mask)
        print("in_tiss_filt shape and gradient:", in_tiss_filt.shape, in_tiss_filt.requires_grad)
        # Vectorized Sinkhorn
        def sinkhorn_wrapper(o, i, r):
            return self.sinkhorn_stabilized_torch(o, i, 
                            cost_batch, reg=r, 
                            stopThr=1e-9,
                            numItermax=3000, 
                            tau=1e4
                            )
        vmap_ot = fc.vmap(
            sinkhorn_wrapper,
            in_dims=(0,0,0),  # Batch dims:
                            # - out_tiss: dim 0 (genes)
                            # - in_tiss_filt: dim 0 (genes)
                            # - regs: dim 0 (already per-gene)
            out_dims=0          # Output stacked along dim 0 (genes)
        )
        
        # 5. Compute OT - input shapes:
        # out_tiss: [4992, 10]
        # in_tiss_filt: [4992, 10]
        # regs: [10]
        ot_emd = vmap_ot(out_tiss.T, in_tiss_filt.T, regs)  # Output: [10, 4992, 4992]
        # Format results
        masses = self.sub_count[:, [self.adata.var_names.get_loc(g) for g in genes]].sum(dim=0)
        masses.dtype = torch.float32 if masses.dtype in [torch.int16, torch.int32, torch.int64] else masses.dtype
        transported = (masses @ ot_emd.sum(dim=1)).T  # summed over axis 1 to get transportation in
        return transported


    def sequential_ot(self, genes, cost_weights, invalid_qts, regs):
        transported = []
        for g, q, r in zip(genes, invalid_qts, regs):
            res = self.compute_ot(g, cost_weights, q, r)
            transported.append(res['in'])
        return torch.stack(transported).T
    
    def objective(self, optim_params=None):
        """Parallel objective function"""
        genes = self.gene_selected
        self.params.update(optim_params) if optim_params is not None else None
        params = self.params
        cost_weights = params['cost_weights']
        invalid_qts = params['invalid_qts']
        regs = params['reg']
        
        # Process genes in small
        batch_size = self.batch_size
        num_batches = (len(genes) // batch_size) + (1 if len(genes) % batch_size != 0 else 0)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(genes))
            batch_genes = genes[start_idx:end_idx]
            
            # Get corresponding parameters for this batch
            batch_invalid_qts = invalid_qts[start_idx:end_idx]
            batch_regs = regs[start_idx:end_idx]
            
            # Batched OT computation for this chunk
            with autocast():
                transported = self.batched_compute_ot(batch_genes, cost_weights, 
                                                    batch_invalid_qts, batch_regs)
            
            # Update counts for this batch
            gene_indices = torch.tensor(
                [self.adata.var_names.get_loc(g) for g in batch_genes],
                device=self.sub_count.device
            )
            mask = torch.zeros_like(self.sub_count, dtype=torch.bool)
            mask[:, gene_indices] = True
            self.sub_count = torch.where(mask, transported, self.sub_count)
            
        # Update counts in batch
        gene_indices = torch.tensor(
            [self.adata.var_names.get_loc(g) for g in genes],
            device=self.sub_count.device
        )
        mask = torch.zeros_like(self.sub_count, dtype=torch.bool)
        mask[:, gene_indices] = True
        self.sub_count = torch.where(mask, transported, self.sub_count)
        
        # Compute losses (unchanged)
        self.last_z = self.sub_count.clone()
        
        loss_dict = self.calc_loss()   
        weights = undiff_batch.balance_weights(loss_dict)
        loss = sum(
            loss_dict[k] * weights[k] for k in loss_dict.keys()
        )
        return loss

    # CUDA Graph optimization
    def _capture_cuda_graph(self, optimizer):
        """Capture repetitive computation in CUDA Graph"""
        static_loss = torch.zeros(1, device='cuda')
        static_params = {k: v.detach() for k, v in self._get_params().items()}
        
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(1):
                optimizer.zero_grad(set_to_none=True)
                sub_count = self.sub_count.clone()
                loss = self.objective(static_params)
                loss.backward()
                optimizer.step()
                sub_count = sub_count
        torch.cuda.current_stream().wait_stream(s)
        
        # Capture
        g = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            static_loss = self.objective(static_params)
            static_loss.backward()
            optimizer.step()
        
        return g, static_loss, static_params

    def optimization(self, first_n_genes=None, add_genes=[], optim_params=None, optim_name=None):
        """Optimization with CUDA Graph support"""
        self.params.update(optim_params) if optim_params is not None else None
        params = self.params
        self.gene_selected = self.gene_selection()
        first_n_genes = first_n_genes if first_n_genes is not None else len(self.gene_selected)
        self.gene_selected = np.unique(self.gene_selected[:first_n_genes] + add_genes)
        self.set_states()
        
        lr = params['train_lr']
        optim_name = optim_name if optim_name is not None else self.optimizer
        if optim_name == 'adam':
            optimizer = optim.Adam(
                params=[
                    {'params': self.cost_weights, 'lr': lr},
                    {'params': self.invalid_qts, 'lr': lr},
                    {'params': self.reg, 'lr': lr}
                ],
            )
        elif optim_name == 'lbfgs':
            optimizer = optim.LBFGS(
                params=[
                    {'params': self.cost_weights, 'lr': lr},
                    {'params': self.invalid_qts, 'lr': lr},
                    {'params': self.reg, 'lr': lr}
                ],
            )

        # Use CUDA Graph if available
        if torch.cuda.is_available():
            g, static_loss, static_params = self._capture_cuda_graph(optimizer)
            
            for epoch in range(self.params['train_n_epochs']):
                # Replay graph
                g.replay()
                
                # Check convergence
                if static_loss.item() < self.params['train_tol']:
                    print(f"Converged at epoch {epoch}")
                    break
        else:
            # Fallback to regular training
            for epoch in range(self.params['train_n_epochs']):
                optimizer.zero_grad()
                sub_count = self.sub_count.clone()
                loss = self.objective({
                    'cost_weights': self.cost_weights,
                    'invalid_qts': self.invalid_qts,
                    'reg': self.reg
                })
                loss.backward()
                
                if torch.cat([p.grad.flatten() for p in optimizer.param_groups[0]['params']]).norm() < self.params['train_tol']:
                    break
                    
                optimizer.step()
                self.sub_count = sub_count

        return self._get_results()