import os,glob
import torch
import dgl
from torch.utils.data import Dataset

class ExprGraphDataset(Dataset):
    def __init__(self,graph_dir,mask_rate=0.15):
        self.files=sorted(glob.glob(os.path.join(graph_dir,"*.bin")))
        self.mask_rate=mask_rate

    def __len__(self):
        return len(self.files)

    def mask_expr(self,g):
        x=g.nodes["gene"].data["expr"]
        n=x.shape[0]
        m=max(1,int(n*self.mask_rate))
        idx=torch.randperm(n)[:m]

        target=x.clone()
        x_masked=x.clone()

        n_mask=int(0.8*m)
        n_rand=int(0.1*m)

        if n_mask>0:
            x_masked[idx[:n_mask]]=0.0
        if n_rand>0:
            ridx=torch.randint(0,n,(n_rand,))
            x_masked[idx[n_mask:n_mask+n_rand]]=x[ridx]

        mask=torch.zeros(n,dtype=torch.bool)
        mask[idx]=True

        g.nodes["gene"].data["expr_masked"]=x_masked
        g.nodes["gene"].data["target"]=target
        g.nodes["gene"].data["mask"]=mask
        return g

    def __getitem__(self,idx):
        g,_=dgl.load_graphs(self.files[idx])
        g=g[0]
        return self.mask_expr(g)
