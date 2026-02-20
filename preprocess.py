import json
import pandas as pd
import torch
import dgl
import os,glob
import pandas as pd
import dgl
from tqdm import tqdm


def build_dgl_hetero(expr_df,sample_col,ppi_csv,pathway_csv,
                     uniprot_json,
                     ppi_u="Interactor1",ppi_v="Interactor2",
                     ppi_score="combined_score",score_thr=700,
                     path_genes_col="genes",min_path_genes=3):
    with open(uniprot_json) as f:
        uid2idx=json.load(f)

    genes=[g for g in expr_df.index.astype(str) if g in uid2idx]
    gid=[uid2idx[g] for g in genes]
    local2global={i:gid[i] for i in range(len(gid))}
    global2local={gid[i]:i for i in range(len(gid))}

    x=torch.tensor(expr_df.loc[genes,[sample_col]].values,dtype=torch.float32)

    ppi=pd.read_csv(ppi_csv)
    # ppi=ppi[ppi[ppi_score]>=score_thr]
    g2g_s=[];g2g_d=[]
    for u,v in zip(ppi[ppi_u].astype(str),ppi[ppi_v].astype(str)):
        if u in uid2idx and v in uid2idx:
            if uid2idx[u] in global2local and uid2idx[v] in global2local:
                i=global2local[uid2idx[u]]
                j=global2local[uid2idx[v]]
                g2g_s+= [i,j]; g2g_d+= [j,i]
    g2g_s=torch.tensor(g2g_s,dtype=torch.int64)
    g2g_d=torch.tensor(g2g_d,dtype=torch.int64)

    pw=pd.read_csv(pathway_csv)
    g2p_s=[];g2p_d=[];p2g_s=[];p2g_d=[]
    pid=0;pid2idx={}
    for _,r in pw.iterrows():
        gs=[uid2idx[g] for g in str(r[path_genes_col]).split(",") if g in uid2idx and uid2idx[g] in global2local]
        if len(gs)<min_path_genes: continue
        idx=[global2local[g] for g in gs]
        pid2idx[pid]=idx
        for gi in idx:
            g2p_s.append(gi); g2p_d.append(pid)
            p2g_s.append(pid); p2g_d.append(gi)
        pid+=1
    g2p_s=torch.tensor(g2p_s,dtype=torch.int64)
    g2p_d=torch.tensor(g2p_d,dtype=torch.int64)
    p2g_s=torch.tensor(p2g_s,dtype=torch.int64)
    p2g_d=torch.tensor(p2g_d,dtype=torch.int64)

    g=dgl.heterograph({
        ("gene","g2g","gene"):(g2g_s,g2g_d),
        ("gene","g2p","pathway"):(g2p_s,g2p_d),
        ("pathway","p2g","gene"):(p2g_s,p2g_d),
        ("virtual","v2p","pathway"):(torch.zeros(pid,dtype=torch.int64),torch.arange(pid)),
        ("pathway","p2v","virtual"):(torch.arange(pid),torch.zeros(pid,dtype=torch.int64))
    },num_nodes_dict={"gene":len(genes),"pathway":pid,"virtual":1})

    g.nodes["gene"].data["expr"]=x
    g.nodes["gene"].data["token"]=torch.tensor(gid,dtype=torch.int64)

    if pid>0:
        pfeat=torch.stack([x[idx].mean(0) for idx in pid2idx.values()])
        vfeat=pfeat.mean(0,keepdim=True)
    else:
        pfeat=torch.zeros((0,1)); vfeat=torch.zeros((1,1))
    g.nodes["pathway"].data["feat"]=pfeat
    g.nodes["virtual"].data["feat"]=vfeat
    return g



EXPR_DIR="dataset/pretrain/human_gse_gpl_uniprot_scaled"
PPI_CSV="dataset/interaction/reactome.csv"
PATHWAY_CSV="dataset/pathways/reactome.csv"
UNIPROT_JSON="dataset/uniprot_id2idx.json"
OUT_DIR="dataset/pretrain/dgl_graphs"

os.makedirs(OUT_DIR,exist_ok=True)

def main():
    expr_files=sorted(glob.glob(os.path.join(EXPR_DIR,"*.csv")))
    for f in tqdm(expr_files,desc="preprocess"):
        df=pd.read_csv(f,index_col=0)
        base=os.path.splitext(os.path.basename(f))[0]
        for col in df.columns:
            g=build_dgl_hetero(
                expr_df=df,
                sample_col=col,
                ppi_csv=PPI_CSV,
                pathway_csv=PATHWAY_CSV,
                uniprot_json=UNIPROT_JSON
            )
            out=os.path.join(OUT_DIR,f"{base}__{col}.bin")
            dgl.save_graphs(out,g)

if __name__=="__main__":
    main()
