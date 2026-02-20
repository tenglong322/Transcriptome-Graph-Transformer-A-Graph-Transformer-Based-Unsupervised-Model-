library(GEOquery)
library(parallel)

output <- read.csv("gse_gpl.csv", sep=',')
platform_ids <- unique(output$GPL)

dir.create("./gpls_compressed/", showWarnings = FALSE)
dir.create("./gpls/", showWarnings = FALSE)

# Define the function to download and save the platform data
download_platform_data <- function(platform_id) {
  cat("Processing platform", platform_id, "\n")
  
  platform_data <- tryCatch({
    getGEO(platform_id, destdir = "./gpls_compressed/")
  }, error = function(e) {
    cat("Error downloading platform ", platform_id, ": ", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(platform_data)) {
    platform_table <- Table(platform_data)
    csv_file <- paste0("./gpls/", platform_id, ".csv")
    write.csv(platform_table, csv_file, row.names = FALSE)
    cat("Platform", platform_id, "saved to", csv_file, "\n")
  }
}

# Use mclapply to download in parallel (for Linux/MacOS systems)
# For Windows, use parLapply with a cluster

num_cores <- detectCores() - 10  # Use one less than the total number of cores

platform_ids_split <- split(platform_ids, ceiling(seq_along(platform_ids) / num_cores))

# Using mclapply for parallel processing (Linux/MacOS)
# If you are on Windows, replace mclapply with parLapply
result <- mclapply(platform_ids_split, function(ids) {
  lapply(ids, download_platform_data)
}, mc.cores = num_cores)

cat("Download and CSV export completed.\n")