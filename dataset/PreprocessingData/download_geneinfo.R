rm(list=ls())
library(org.Hs.eg.db)
library(xtable)

# Extract various mappings from org.Hs.eg.db
eg2symbol = toTable(org.Hs.egSYMBOL)
eg2name = toTable(org.Hs.egGENENAME)
eg2alias = toTable(org.Hs.egALIAS2EG)
eg2ensembl = toTable(org.Hs.egENSEMBL)  # Ensembl IDs
eg2uniprot = toTable(org.Hs.egUNIPROT)  # UniProt IDs

# List of gene symbols or IDs
eg2alis_list = lapply(split(eg2alias, eg2alias$gene_id), function(x) {
  paste0(x[,2], collapse = ";")
})

GeneList = mappedLkeys(org.Hs.egSYMBOL)

# Match GeneList to symbols and gene_ids
if (GeneList[1] %in% eg2symbol$symbol) {
  symbols = GeneList
  geneIds = eg2symbol[match(symbols, eg2symbol$symbol), 'gene_id']
} else {
  geneIds = GeneList
  symbols = eg2symbol[match(geneIds, eg2symbol$gene_id), 'symbol']
}

# Get gene names, aliases, Ensembl, and UniProt
geneNames = eg2name[match(geneIds, eg2name$gene_id), 'gene_name']
geneAlias = sapply(geneIds, function(x) {
  ifelse(is.null(eg2alis_list[[x]]), "no_alias", eg2alis_list[[x]])
})

# Ensembl and UniProt IDs
ensemblIds = eg2ensembl[match(geneIds, eg2ensembl$gene_id), 'ensembl_id']
uniprotIds = eg2uniprot[match(geneIds, eg2uniprot$gene_id), 'uniprot_id']

# Create a function to generate clickable links
createLink <- function(base, val) {
  sprintf('<a href="%s" class="btn btn-link" target="_blank">%s</a>', base, val)
}

# Generate gene_info dataframe with additional columns
gene_info = data.frame(
  symbols = symbols,
  geneIds = geneIds,
  geneNames = geneNames,
  geneAlias = geneAlias,
  ensemblIds = ensemblIds,
  uniprotIds = uniprotIds,
  stringsAsFactors = FALSE
)


write.table(gene_info, file = "gene_info.csv", sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)