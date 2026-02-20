import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax

# ---------------- Graph Transformer Layer ----------------

class GraphTransformerLayer(nn.Module):
    def __init__(self,dim,heads=8,drop=0.1):
        super().__init__()
        assert dim%heads==0
        self.dim=dim; self.h=heads; self.dk=dim//heads
        self.Wq=nn.Linear(dim,dim)
        self.Wk=nn.Linear(dim,dim)
        self.Wv=nn.Linear(dim,dim)
        self.proj=nn.Linear(dim,dim)
        self.drop=nn.Dropout(drop)
        self.norm=nn.LayerNorm(dim)

    def forward(self,g,h_src,h_dst,etype):
        s,_,d=g.to_canonical_etype(etype)
        with g.local_scope():
            q=self.Wq(h_dst).view(-1,self.h,self.dk)
            k=self.Wk(h_src).view(-1,self.h,self.dk)
            v=self.Wv(h_src).view(-1,self.h,self.dk)
            g.nodes[d].data["q"]=q
            g.nodes[s].data["k"]=k
            g.nodes[s].data["v"]=v
            g.apply_edges(fn.u_dot_v("k","q","score"),etype=etype)
            score=g.edges[etype].data["score"]/(self.dk**0.5)
            attn=edge_softmax(g,score,etype=etype)
            g.edges[etype].data["a"]=attn
            g.update_all(fn.u_mul_e("v","a","m"),fn.sum("m","out"),etype=etype)
            out=g.nodes[d].data["out"].reshape(-1,self.dim)
            return self.norm(h_dst+self.drop(self.proj(out)))

# ---------------- Encoder (Backbone) ----------------

class TGTEncoder(nn.Module):
    def __init__(self,num_genes,dim=512,heads=8,layers=2,drop=0.1):
        super().__init__()
        self.token_emb=nn.Embedding(num_genes,256)
        self.expr_mlp=nn.Sequential(
            nn.Linear(1,256),nn.GELU(),nn.Dropout(drop),
            nn.Linear(256,256)
        )
        self.gene_proj=nn.Linear(512,dim)
        self.path_proj=nn.Linear(1,dim)
        self.virt_proj=nn.Linear(1,dim)

        self.blocks=nn.ModuleList([
            nn.ModuleDict({
                "g2g":GraphTransformerLayer(dim,heads,drop),
                "g2p":GraphTransformerLayer(dim,heads,drop),
                "p2g":GraphTransformerLayer(dim,heads,drop),
                "p2p":GraphTransformerLayer(dim,heads,drop),
                "v2p":GraphTransformerLayer(dim,heads,drop),
                "p2v":GraphTransformerLayer(dim,heads,drop),
            }) for _ in range(layers)
        ])

    def encode(self,g):
        gd=g.nodes["gene"].data
        x=gd["expr_masked"] if "expr_masked" in gd else gd["expr"]
        tok=gd["token"]
        h_gene=self.gene_proj(torch.cat([self.expr_mlp(x),self.token_emb(tok)],1))

        h_path=self.path_proj(g.nodes["pathway"].data["feat"].view(-1,1))
        h_virt=self.virt_proj(g.nodes["virtual"].data["feat"].view(-1,1))

        h={"gene":h_gene,"pathway":h_path,"virtual":h_virt}

        for blk in self.blocks:
            if g.num_edges("g2g")>0:
                h["gene"]=blk["g2g"](g,h["gene"],h["gene"],"g2g")
            if g.num_edges("g2p")>0:
                h["pathway"]=blk["g2p"](g,h["gene"],h["pathway"],"g2p")
            if g.num_edges("p2g")>0:
                h["gene"]=blk["p2g"](g,h["pathway"],h["gene"],"p2g")
            if g.num_edges("p2p")>0:
                h["pathway"]=blk["p2p"](g,h["pathway"],h["pathway"],"p2p")
            if g.num_edges("v2p")>0:
                h["pathway"]=blk["v2p"](g,h["virtual"],h["pathway"],"v2p")
            if g.num_edges("p2v")>0:
                h["virtual"]=blk["p2v"](g,h["pathway"],h["virtual"],"p2v")
        return h

# ---------------- Pretrain Head ----------------

class ExprRecoverHead(nn.Module):
    def __init__(self,dim,drop=0.1):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(dim,dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim,1)
        )
    def forward(self,h):
        return self.mlp(h["gene"])

class TGTPretrain(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.head=ExprRecoverHead(encoder.gene_proj.out_features)
    def forward(self,g):
        h=self.encoder.encode(g)
        return self.head(h)

# ---------------- Finetune Head ----------------

class GraphClsHead(nn.Module):
    def __init__(self,dim,num_classes,drop=0.1):
        super().__init__()
        self.cls=nn.Sequential(
            nn.Linear(dim,512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512,num_classes)
        )
    def forward(self,h):
        return self.cls(h["virtual"])

class TGTFinetune(nn.Module):
    def __init__(self,encoder,num_classes):
        super().__init__()
        self.encoder=encoder
        self.head=GraphClsHead(encoder.gene_proj.out_features,num_classes)
    def forward(self,g):
        h=self.encoder.encode(g)
        return self.head(h)
