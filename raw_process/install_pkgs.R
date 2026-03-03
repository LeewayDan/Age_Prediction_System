<<<<<<< HEAD
# 1. 设置镜像
options(
  repos = c(CRAN = "https://mirrors.pku.edu.cn/CRAN/"),
  BioC_mirror = "https://mirrors.westlake.edu.cn/bioconductor"
)
options(install.packages.compile.from.source = "never")

# 2. 检查 Rtools（Windows 必需）
if (.Platform$OS.type == "windows") {
  if (!requireNamespace("pkgbuild", quietly = TRUE)) install.packages("pkgbuild")
  if (!pkgbuild::has_rtools(debug = TRUE)) {
    stop("Rtools 未检测到，请先安装 Rtools 4.3 并配置 PATH")
  } else {
    message("检测到 Rtools，可编译 Bioconductor 源码包")
  }
}

# 3. 安装基础工具包
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!requireNamespace("pak", quietly = TRUE))
  install.packages("pak")
if (!requireNamespace("remotes", quietly = TRUE))
  install.packages("remotes")

# 4. 确认 Bioconductor 版本
BiocManager::install(version = "3.18", ask = FALSE)

# 5. 定义 MAPLE 依赖
cran_packages <- c(
  "xml2",
  "data.table",
  "foreach",
  "iterators",
  "doParallel",
  "locfit",
  "matrixStats",
  "remotes",
  "pkgbuild"
)

bioc_packages <- c(
  "GenomeInfoDbData",
  "GenomeInfoDb",
  "GenomicRanges",
  "IRanges",
  "S4Vectors",
  "SummarizedExperiment",
  "MatrixGenerics",
  "matrixStats",
  "Biobase",
  "Biostrings",
  "bumphunter",
  "scrime",
  "minfi",
  "ENmix"
)

github_packages <- c("MengweiLi-project/gmqn")
all_packages <- c(cran_packages, bioc_packages)

# 6. 清理旧包
message("检查并卸载旧的 Bioconductor 包...")
for (pkg in bioc_packages) {
  if (pkg %in% rownames(installed.packages())) {
    message("卸载旧包: ", pkg)
    remove.packages(pkg)
  }
}

# 7. 安装 CRAN 包
install.packages(cran_packages, dependencies = TRUE)

# 8. 安装 Bioconductor 包（注意：强制从源码安装 minfi 和 ENmix）
for (pkg in bioc_packages) {
  if (pkg %in% c("minfi","ENmix")) {
    BiocManager::install(pkg, type = "source", ask = FALSE, force = TRUE)
  } else {
    BiocManager::install(pkg, ask = FALSE, update = TRUE, force = TRUE)
  }
}

# 9. 安装 GitHub 包
pak::pkg_install(github_packages)

# 10. 检查安装结果
installed <- sapply(c(all_packages, "gmqn"), function(pkg) isTRUE(requireNamespace(pkg, quietly = TRUE))
)
versions <- sapply(c(all_packages, "gmqn"), function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    tryCatch(as.character(packageVersion(pkg)), error = function(e) NA)
  } else {
    NA
  }
})

df <- data.frame(
  Package = c(all_packages, "gmqn"),
  Installed = installed,
  Version = versions,
  row.names = NULL,
  stringsAsFactors = FALSE
)
=======
# 1. 设置镜像
options(
  repos = c(CRAN = "https://mirrors.pku.edu.cn/CRAN/"),
  BioC_mirror = "https://mirrors.westlake.edu.cn/bioconductor"
)
options(install.packages.compile.from.source = "never")

# 2. 检查 Rtools（Windows 必需）
if (.Platform$OS.type == "windows") {
  if (!requireNamespace("pkgbuild", quietly = TRUE)) install.packages("pkgbuild")
  if (!pkgbuild::has_rtools(debug = TRUE)) {
    stop("Rtools 未检测到，请先安装 Rtools 4.3 并配置 PATH")
  } else {
    message("检测到 Rtools，可编译 Bioconductor 源码包")
  }
}

# 3. 安装基础工具包
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!requireNamespace("pak", quietly = TRUE))
  install.packages("pak")
if (!requireNamespace("remotes", quietly = TRUE))
  install.packages("remotes")

# 4. 确认 Bioconductor 版本
BiocManager::install(version = "3.18", ask = FALSE)

# 5. 定义 MAPLE 依赖
cran_packages <- c(
  "xml2",
  "data.table",
  "foreach",
  "iterators",
  "doParallel",
  "locfit",
  "matrixStats",
  "remotes",
  "pkgbuild"
)

bioc_packages <- c(
  "GenomeInfoDbData",
  "GenomeInfoDb",
  "GenomicRanges",
  "IRanges",
  "S4Vectors",
  "SummarizedExperiment",
  "MatrixGenerics",
  "matrixStats",
  "Biobase",
  "Biostrings",
  "bumphunter",
  "scrime",
  "minfi",
  "ENmix"
)

github_packages <- c("MengweiLi-project/gmqn")
all_packages <- c(cran_packages, bioc_packages)

# 6. 清理旧包
message("检查并卸载旧的 Bioconductor 包...")
for (pkg in bioc_packages) {
  if (pkg %in% rownames(installed.packages())) {
    message("卸载旧包: ", pkg)
    remove.packages(pkg)
  }
}

# 7. 安装 CRAN 包
install.packages(cran_packages, dependencies = TRUE)

# 8. 安装 Bioconductor 包（注意：强制从源码安装 minfi 和 ENmix）
for (pkg in bioc_packages) {
  if (pkg %in% c("minfi","ENmix")) {
    BiocManager::install(pkg, type = "source", ask = FALSE, force = TRUE)
  } else {
    BiocManager::install(pkg, ask = FALSE, update = TRUE, force = TRUE)
  }
}

# 9. 安装 GitHub 包
pak::pkg_install(github_packages)

# 10. 检查安装结果
installed <- sapply(c(all_packages, "gmqn"), function(pkg) isTRUE(requireNamespace(pkg, quietly = TRUE))
)
versions <- sapply(c(all_packages, "gmqn"), function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    tryCatch(as.character(packageVersion(pkg)), error = function(e) NA)
  } else {
    NA
  }
})

df <- data.frame(
  Package = c(all_packages, "gmqn"),
  Installed = installed,
  Version = versions,
  row.names = NULL,
  stringsAsFactors = FALSE
)
>>>>>>> 040f1c6a868faffebaac0b04ba3e3c569e649088
print(df)