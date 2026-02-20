library(GEOquery)
library(parallel)
options('download.file.method.GEOquery' = 'wget')
setwd("gse_gpl")

output_dir <- getwd()
meta_dir <- file.path(output_dir, "gse_gpl")
if (!dir.exists(meta_dir)) dir.create(meta_dir)

all_file <- file.path(output_dir, "all.csv")
if (!file.exists(all_file)) {
  stop("The 'all.csv' file does not exist in the current directory.")
}

gse_ids <- read.csv(all_file, header = FALSE, stringsAsFactors = FALSE)[, 1]
if (length(gse_ids) == 0) {
  stop("No valid GEO IDs found in the 'all.csv' file.")
}

quick_download_expression_matrix <- function(gse_id, output_dir, meta_dir) {
  tryCatch({
    tmp = tempdir()
    gse_data <- getGEO(gse_id, GSEMatrix = TRUE, AnnotGPL = FALSE, destdir = tempdir())
    if (length(gse_data) == 0) {
      return(NULL)
    }
    for (i in seq_along(gse_data)) {
      current_dataset <- gse_data[[i]]
      if (is.null(exprs(current_dataset))) {
        next
      }
      expr_matrix <- exprs(current_dataset)
      platform_id <- annotation(current_dataset)
      meta_file <- file.path(meta_dir, paste0(gse_id, "_", platform_id, ".csv"))
      if (file.exists(meta_file)) {
        next
      }
      write.csv(expr_matrix, file = meta_file, row.names = TRUE)
      rm(current_dataset)
    }
    rm(gse_data)
    gc() 
    unlink(temp_dir, recursive = TRUE, force = TRUE)
    Sys.sleep(0.1)
  }, error = function(e) {
    return(NULL)
  })
}

num_cores <- max(1, detectCores() - 6)
cl <- makeCluster(num_cores)

clusterExport(cl, list("quick_download_expression_matrix", "output_dir", "meta_dir"))
clusterEvalQ(cl, library(GEOquery))

parLapply(cl, gse_ids, function(gse_id) {
  quick_download_expression_matrix(gse_id, output_dir, meta_dir)
})

stopCluster(cl)
message("All available files have been downloaded and saved to the specified folder.")
