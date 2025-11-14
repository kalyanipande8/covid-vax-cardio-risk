# Placeholder R data ingestion script
# Use `readr::read_csv()` or `arrow::read_parquet()` depending on file type.

load_tabular <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext == "csv") {
    if (!requireNamespace("readr", quietly = TRUE)) stop("readr required")
    return(readr::read_csv(path))
  } else if (ext %in% c("parquet", "feather")) {
    if (!requireNamespace("arrow", quietly = TRUE)) stop("arrow required for parquet/feather")
    return(arrow::read_parquet(path))
  }
  stop("Unsupported file extension: ", ext)
}

basic_profile <- function(df) {
  n <- nrow(df)
  list(n_rows = n, n_cols = ncol(df), n_missing_total = sum(is.na(df)))
}
