Lab 9 - HPC
================

# Learning goals

In this lab, you are expected to practice the following skills:

- Evaluate whether a problem can be parallelized or not.
- Practice with the parallel package.
- Use Rscript to submit jobs.

## Problem 1

Give yourself a few minutes to think about what you learned about
parallelization. List three examples of problems that you believe may be
solved using parallel computing, and check for packages on the HPC CRAN
task view that may be related to it.

1.  Image processing for large datasets - Processing large sets of
    images for tasks such as object detection, segmentation, or
    enhancement can be computationally intensive. Each image can be
    processed independently. Packages: parallel, RcppParallel.

2.  Big data analysis using MapReduce - Analyzing large-scale datasets,
    such as log files or user data, using MapReduce techniques. Tasks
    like filtering, aggregating, and summarizing data can be
    parallelized efficiently. Packages - Rhipe, RHadoop.

3.  Monte Carlo simulations - Can simplify Monte Carlo simulation
    studies by automatically setting up loops to run over parameter
    grids and parellelizing the repitions. Packagges - MonteCarlo.

## Problem 2: Pre-parallelization

The following functions can be written to be more efficient without
using `parallel`:

1.  This function generates a `n x k` dataset with all its entries
    having a Poisson distribution with mean `lambda`.

``` r
library(microbenchmark)

fun1 <- function(n = 100, k = 4, lambda = 4) {
  x <- NULL
  
  for (i in 1:n)
    x <- rbind(x, rpois(k, lambda))
  
  return(x)
}

fun1alt <- function(n = 100, k = 4, lambda = 4) {
  x <- matrix(rpois(n * k, lambda), nrow = n, ncol = k)
  return(x)
}

# Benchmarking
microbenchmark::microbenchmark(
  fun1(),
  fun1alt()
)
```

<div data-pagedtable="false">

<script data-pagedtable-source type="application/json">
{"columns":[{"label":["expr"],"name":[1],"type":["fct"],"align":["left"]},{"label":["time"],"name":[2],"type":["dbl"],"align":["right"]}],"data":[{"1":"fun1alt()","2":"193125"},{"1":"fun1()","2":"15179418"},{"1":"fun1alt()","2":"2137126"},{"1":"fun1alt()","2":"28876"},{"1":"fun1()","2":"297626"},{"1":"fun1alt()","2":"29542"},{"1":"fun1alt()","2":"27125"},{"1":"fun1alt()","2":"26709"},{"1":"fun1()","2":"294001"},{"1":"fun1()","2":"282751"},{"1":"fun1()","2":"283376"},{"1":"fun1()","2":"280834"},{"1":"fun1()","2":"276292"},{"1":"fun1alt()","2":"28500"},{"1":"fun1alt()","2":"27042"},{"1":"fun1alt()","2":"25626"},{"1":"fun1alt()","2":"26876"},{"1":"fun1alt()","2":"25792"},{"1":"fun1()","2":"277709"},{"1":"fun1alt()","2":"27918"},{"1":"fun1alt()","2":"25751"},{"1":"fun1()","2":"280250"},{"1":"fun1alt()","2":"26376"},{"1":"fun1alt()","2":"26126"},{"1":"fun1()","2":"277625"},{"1":"fun1()","2":"281459"},{"1":"fun1()","2":"277084"},{"1":"fun1alt()","2":"27168"},{"1":"fun1()","2":"276959"},{"1":"fun1()","2":"275292"},{"1":"fun1()","2":"276959"},{"1":"fun1()","2":"278792"},{"1":"fun1()","2":"306834"},{"1":"fun1()","2":"298001"},{"1":"fun1()","2":"306167"},{"1":"fun1()","2":"427625"},{"1":"fun1alt()","2":"28167"},{"1":"fun1()","2":"298167"},{"1":"fun1()","2":"307292"},{"1":"fun1()","2":"300834"},{"1":"fun1alt()","2":"29334"},{"1":"fun1alt()","2":"28376"},{"1":"fun1alt()","2":"28376"},{"1":"fun1()","2":"309543"},{"1":"fun1()","2":"310418"},{"1":"fun1alt()","2":"28168"},{"1":"fun1()","2":"316793"},{"1":"fun1alt()","2":"28792"},{"1":"fun1()","2":"304042"},{"1":"fun1alt()","2":"28417"},{"1":"fun1alt()","2":"28875"},{"1":"fun1alt()","2":"26792"},{"1":"fun1alt()","2":"27667"},{"1":"fun1()","2":"321667"},{"1":"fun1()","2":"309917"},{"1":"fun1()","2":"320959"},{"1":"fun1()","2":"303334"},{"1":"fun1()","2":"298667"},{"1":"fun1alt()","2":"29084"},{"1":"fun1()","2":"302584"},{"1":"fun1()","2":"307459"},{"1":"fun1()","2":"309501"},{"1":"fun1()","2":"297376"},{"1":"fun1()","2":"301543"},{"1":"fun1()","2":"304709"},{"1":"fun1()","2":"304542"},{"1":"fun1alt()","2":"29167"},{"1":"fun1alt()","2":"26834"},{"1":"fun1()","2":"310000"},{"1":"fun1alt()","2":"28501"},{"1":"fun1()","2":"316959"},{"1":"fun1()","2":"315126"},{"1":"fun1()","2":"351667"},{"1":"fun1()","2":"360375"},{"1":"fun1alt()","2":"27918"},{"1":"fun1()","2":"351168"},{"1":"fun1()","2":"348126"},{"1":"fun1alt()","2":"27584"},{"1":"fun1()","2":"369959"},{"1":"fun1()","2":"340876"},{"1":"fun1alt()","2":"29168"},{"1":"fun1alt()","2":"28542"},{"1":"fun1()","2":"398459"},{"1":"fun1()","2":"359042"},{"1":"fun1()","2":"369167"},{"1":"fun1()","2":"339709"},{"1":"fun1alt()","2":"31043"},{"1":"fun1()","2":"350959"},{"1":"fun1()","2":"362209"},{"1":"fun1()","2":"345959"},{"1":"fun1alt()","2":"27500"},{"1":"fun1alt()","2":"31167"},{"1":"fun1()","2":"349834"},{"1":"fun1alt()","2":"31501"},{"1":"fun1()","2":"354751"},{"1":"fun1()","2":"347334"},{"1":"fun1()","2":"371834"},{"1":"fun1alt()","2":"28292"},{"1":"fun1alt()","2":"32209"},{"1":"fun1()","2":"372626"},{"1":"fun1()","2":"363084"},{"1":"fun1alt()","2":"27084"},{"1":"fun1alt()","2":"31459"},{"1":"fun1alt()","2":"27000"},{"1":"fun1alt()","2":"27834"},{"1":"fun1alt()","2":"30917"},{"1":"fun1()","2":"364168"},{"1":"fun1alt()","2":"30793"},{"1":"fun1alt()","2":"27834"},{"1":"fun1alt()","2":"26209"},{"1":"fun1()","2":"377209"},{"1":"fun1()","2":"396709"},{"1":"fun1()","2":"357334"},{"1":"fun1alt()","2":"28334"},{"1":"fun1alt()","2":"28709"},{"1":"fun1alt()","2":"26959"},{"1":"fun1()","2":"362000"},{"1":"fun1()","2":"369917"},{"1":"fun1alt()","2":"27000"},{"1":"fun1alt()","2":"32917"},{"1":"fun1alt()","2":"27167"},{"1":"fun1alt()","2":"27417"},{"1":"fun1()","2":"368251"},{"1":"fun1alt()","2":"27167"},{"1":"fun1alt()","2":"27418"},{"1":"fun1alt()","2":"30959"},{"1":"fun1alt()","2":"26584"},{"1":"fun1()","2":"366126"},{"1":"fun1()","2":"356417"},{"1":"fun1alt()","2":"26875"},{"1":"fun1alt()","2":"29584"},{"1":"fun1alt()","2":"26875"},{"1":"fun1alt()","2":"30126"},{"1":"fun1()","2":"371001"},{"1":"fun1alt()","2":"27875"},{"1":"fun1()","2":"366876"},{"1":"fun1alt()","2":"28251"},{"1":"fun1()","2":"368500"},{"1":"fun1alt()","2":"28917"},{"1":"fun1alt()","2":"28125"},{"1":"fun1alt()","2":"31584"},{"1":"fun1()","2":"350709"},{"1":"fun1alt()","2":"31209"},{"1":"fun1()","2":"369626"},{"1":"fun1()","2":"355918"},{"1":"fun1alt()","2":"28543"},{"1":"fun1alt()","2":"29376"},{"1":"fun1()","2":"380709"},{"1":"fun1()","2":"360793"},{"1":"fun1alt()","2":"31334"},{"1":"fun1()","2":"400167"},{"1":"fun1()","2":"363834"},{"1":"fun1alt()","2":"30334"},{"1":"fun1()","2":"364334"},{"1":"fun1alt()","2":"32084"},{"1":"fun1alt()","2":"27750"},{"1":"fun1alt()","2":"29875"},{"1":"fun1alt()","2":"27209"},{"1":"fun1alt()","2":"30917"},{"1":"fun1alt()","2":"28292"},{"1":"fun1alt()","2":"27376"},{"1":"fun1alt()","2":"27709"},{"1":"fun1()","2":"375209"},{"1":"fun1alt()","2":"31042"},{"1":"fun1alt()","2":"27125"},{"1":"fun1alt()","2":"26417"},{"1":"fun1()","2":"367501"},{"1":"fun1()","2":"362543"},{"1":"fun1()","2":"357292"},{"1":"fun1()","2":"374084"},{"1":"fun1()","2":"360084"},{"1":"fun1()","2":"359084"},{"1":"fun1()","2":"361792"},{"1":"fun1()","2":"369959"},{"1":"fun1alt()","2":"30792"},{"1":"fun1()","2":"366001"},{"1":"fun1()","2":"368584"},{"1":"fun1()","2":"369084"},{"1":"fun1alt()","2":"28501"},{"1":"fun1alt()","2":"30500"},{"1":"fun1alt()","2":"27709"},{"1":"fun1alt()","2":"27875"},{"1":"fun1alt()","2":"30251"},{"1":"fun1alt()","2":"30668"},{"1":"fun1alt()","2":"27668"},{"1":"fun1alt()","2":"26709"},{"1":"fun1alt()","2":"30501"},{"1":"fun1()","2":"413876"},{"1":"fun1alt()","2":"31584"},{"1":"fun1()","2":"361709"},{"1":"fun1()","2":"373459"},{"1":"fun1()","2":"360375"},{"1":"fun1alt()","2":"28459"},{"1":"fun1()","2":"370500"},{"1":"fun1()","2":"365292"},{"1":"fun1alt()","2":"28751"},{"1":"fun1alt()","2":"26501"},{"1":"fun1alt()","2":"30250"},{"1":"fun1()","2":"369084"},{"1":"fun1alt()","2":"26709"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>

</div>

How much faster?

fun1() takes about 135 microseconds on average and the alternate one
takes about 14 microseconds.

2.  Find the column max (hint: Checkout the function `max.col()`).

``` r
# Data Generating Process (10 x 10,000 matrix)
set.seed(1234)
x <- matrix(rnorm(1e4), nrow=10)

# Find each column's max value
fun2 <- function(x) {
  apply(x, 2, max)
}

fun2alt <- function(x) {
  x[cbind(max.col(t(x)), 1:ncol(x))]
}

# Benchmarking
benchmark <- microbenchmark::microbenchmark(
  fun2(x),
  fun2alt(x)
)

benchmark
```

<div data-pagedtable="false">

<script data-pagedtable-source type="application/json">
{"columns":[{"label":["expr"],"name":[1],"type":["fct"],"align":["left"]},{"label":["time"],"name":[2],"type":["dbl"],"align":["right"]}],"data":[{"1":"fun2(x)","2":"1398501"},{"1":"fun2(x)","2":"3072793"},{"1":"fun2(x)","2":"1254918"},{"1":"fun2alt(x)","2":"1232750"},{"1":"fun2alt(x)","2":"3114001"},{"1":"fun2(x)","2":"1353501"},{"1":"fun2alt(x)","2":"198459"},{"1":"fun2alt(x)","2":"182084"},{"1":"fun2(x)","2":"1402667"},{"1":"fun2(x)","2":"1316167"},{"1":"fun2(x)","2":"1403293"},{"1":"fun2(x)","2":"1350001"},{"1":"fun2alt(x)","2":"172792"},{"1":"fun2(x)","2":"1242917"},{"1":"fun2(x)","2":"1274709"},{"1":"fun2alt(x)","2":"186001"},{"1":"fun2alt(x)","2":"180834"},{"1":"fun2alt(x)","2":"181876"},{"1":"fun2(x)","2":"1362042"},{"1":"fun2alt(x)","2":"193876"},{"1":"fun2alt(x)","2":"180084"},{"1":"fun2(x)","2":"1342626"},{"1":"fun2(x)","2":"1454375"},{"1":"fun2(x)","2":"1435542"},{"1":"fun2(x)","2":"1401917"},{"1":"fun2alt(x)","2":"192709"},{"1":"fun2alt(x)","2":"186000"},{"1":"fun2(x)","2":"1383043"},{"1":"fun2(x)","2":"1521543"},{"1":"fun2alt(x)","2":"188459"},{"1":"fun2(x)","2":"1746167"},{"1":"fun2(x)","2":"1652001"},{"1":"fun2(x)","2":"1655084"},{"1":"fun2alt(x)","2":"195959"},{"1":"fun2(x)","2":"1656293"},{"1":"fun2alt(x)","2":"213251"},{"1":"fun2(x)","2":"1679292"},{"1":"fun2alt(x)","2":"185125"},{"1":"fun2alt(x)","2":"188626"},{"1":"fun2alt(x)","2":"182043"},{"1":"fun2(x)","2":"1652084"},{"1":"fun2(x)","2":"1628417"},{"1":"fun2(x)","2":"1661042"},{"1":"fun2alt(x)","2":"202417"},{"1":"fun2alt(x)","2":"187042"},{"1":"fun2(x)","2":"3571834"},{"1":"fun2(x)","2":"1160750"},{"1":"fun2alt(x)","2":"149917"},{"1":"fun2alt(x)","2":"149417"},{"1":"fun2alt(x)","2":"150542"},{"1":"fun2alt(x)","2":"150001"},{"1":"fun2(x)","2":"1190918"},{"1":"fun2alt(x)","2":"147501"},{"1":"fun2alt(x)","2":"144625"},{"1":"fun2(x)","2":"1174542"},{"1":"fun2(x)","2":"1173042"},{"1":"fun2(x)","2":"1165292"},{"1":"fun2alt(x)","2":"146459"},{"1":"fun2alt(x)","2":"145293"},{"1":"fun2alt(x)","2":"135542"},{"1":"fun2alt(x)","2":"137251"},{"1":"fun2(x)","2":"1161584"},{"1":"fun2alt(x)","2":"138792"},{"1":"fun2(x)","2":"1196167"},{"1":"fun2alt(x)","2":"139959"},{"1":"fun2(x)","2":"1155959"},{"1":"fun2(x)","2":"1174334"},{"1":"fun2(x)","2":"1150042"},{"1":"fun2(x)","2":"1163084"},{"1":"fun2(x)","2":"1173709"},{"1":"fun2(x)","2":"1162376"},{"1":"fun2(x)","2":"1151376"},{"1":"fun2alt(x)","2":"142417"},{"1":"fun2alt(x)","2":"141959"},{"1":"fun2(x)","2":"1185792"},{"1":"fun2alt(x)","2":"140334"},{"1":"fun2alt(x)","2":"137418"},{"1":"fun2(x)","2":"1176292"},{"1":"fun2alt(x)","2":"149667"},{"1":"fun2alt(x)","2":"134876"},{"1":"fun2(x)","2":"1174626"},{"1":"fun2alt(x)","2":"137250"},{"1":"fun2(x)","2":"1265251"},{"1":"fun2alt(x)","2":"143501"},{"1":"fun2alt(x)","2":"135125"},{"1":"fun2alt(x)","2":"135292"},{"1":"fun2(x)","2":"1249251"},{"1":"fun2(x)","2":"1240084"},{"1":"fun2alt(x)","2":"140459"},{"1":"fun2alt(x)","2":"145917"},{"1":"fun2alt(x)","2":"134792"},{"1":"fun2(x)","2":"1332709"},{"1":"fun2(x)","2":"1341459"},{"1":"fun2(x)","2":"1270167"},{"1":"fun2(x)","2":"1261917"},{"1":"fun2alt(x)","2":"179417"},{"1":"fun2(x)","2":"1317459"},{"1":"fun2alt(x)","2":"175418"},{"1":"fun2(x)","2":"1325126"},{"1":"fun2alt(x)","2":"168084"},{"1":"fun2(x)","2":"3651292"},{"1":"fun2(x)","2":"1156875"},{"1":"fun2alt(x)","2":"138459"},{"1":"fun2alt(x)","2":"132750"},{"1":"fun2(x)","2":"1189167"},{"1":"fun2(x)","2":"1174543"},{"1":"fun2alt(x)","2":"135626"},{"1":"fun2alt(x)","2":"134043"},{"1":"fun2(x)","2":"1161501"},{"1":"fun2alt(x)","2":"133917"},{"1":"fun2(x)","2":"1159625"},{"1":"fun2alt(x)","2":"135251"},{"1":"fun2(x)","2":"1156668"},{"1":"fun2(x)","2":"1186376"},{"1":"fun2alt(x)","2":"134959"},{"1":"fun2(x)","2":"1164418"},{"1":"fun2(x)","2":"1211709"},{"1":"fun2alt(x)","2":"136542"},{"1":"fun2(x)","2":"1179584"},{"1":"fun2alt(x)","2":"135417"},{"1":"fun2alt(x)","2":"133125"},{"1":"fun2(x)","2":"1167042"},{"1":"fun2alt(x)","2":"134334"},{"1":"fun2(x)","2":"1178917"},{"1":"fun2alt(x)","2":"134875"},{"1":"fun2(x)","2":"1154251"},{"1":"fun2(x)","2":"1146459"},{"1":"fun2(x)","2":"1174376"},{"1":"fun2alt(x)","2":"137834"},{"1":"fun2alt(x)","2":"135001"},{"1":"fun2alt(x)","2":"134293"},{"1":"fun2alt(x)","2":"137918"},{"1":"fun2alt(x)","2":"133459"},{"1":"fun2(x)","2":"1190501"},{"1":"fun2(x)","2":"1202125"},{"1":"fun2alt(x)","2":"137584"},{"1":"fun2(x)","2":"1189750"},{"1":"fun2(x)","2":"1179793"},{"1":"fun2alt(x)","2":"136001"},{"1":"fun2(x)","2":"1185459"},{"1":"fun2(x)","2":"1183834"},{"1":"fun2alt(x)","2":"138417"},{"1":"fun2alt(x)","2":"136334"},{"1":"fun2alt(x)","2":"134959"},{"1":"fun2(x)","2":"1195709"},{"1":"fun2alt(x)","2":"134500"},{"1":"fun2(x)","2":"1178209"},{"1":"fun2alt(x)","2":"135667"},{"1":"fun2alt(x)","2":"132000"},{"1":"fun2(x)","2":"1178709"},{"1":"fun2alt(x)","2":"134625"},{"1":"fun2alt(x)","2":"134584"},{"1":"fun2alt(x)","2":"137418"},{"1":"fun2(x)","2":"1191168"},{"1":"fun2alt(x)","2":"134334"},{"1":"fun2alt(x)","2":"191584"},{"1":"fun2(x)","2":"1265542"},{"1":"fun2(x)","2":"2925292"},{"1":"fun2alt(x)","2":"134126"},{"1":"fun2alt(x)","2":"131751"},{"1":"fun2alt(x)","2":"130500"},{"1":"fun2alt(x)","2":"131626"},{"1":"fun2alt(x)","2":"130626"},{"1":"fun2(x)","2":"1133959"},{"1":"fun2alt(x)","2":"131418"},{"1":"fun2(x)","2":"1141251"},{"1":"fun2alt(x)","2":"132167"},{"1":"fun2(x)","2":"1135375"},{"1":"fun2alt(x)","2":"133042"},{"1":"fun2alt(x)","2":"132751"},{"1":"fun2(x)","2":"1156293"},{"1":"fun2(x)","2":"1156542"},{"1":"fun2alt(x)","2":"136168"},{"1":"fun2alt(x)","2":"133376"},{"1":"fun2alt(x)","2":"132875"},{"1":"fun2(x)","2":"1156626"},{"1":"fun2alt(x)","2":"134667"},{"1":"fun2(x)","2":"1185417"},{"1":"fun2alt(x)","2":"134001"},{"1":"fun2(x)","2":"1167709"},{"1":"fun2(x)","2":"1162793"},{"1":"fun2(x)","2":"1169501"},{"1":"fun2(x)","2":"1158709"},{"1":"fun2alt(x)","2":"135792"},{"1":"fun2alt(x)","2":"132501"},{"1":"fun2(x)","2":"1227459"},{"1":"fun2(x)","2":"1156626"},{"1":"fun2alt(x)","2":"133626"},{"1":"fun2(x)","2":"1171876"},{"1":"fun2(x)","2":"1162375"},{"1":"fun2alt(x)","2":"133001"},{"1":"fun2alt(x)","2":"138418"},{"1":"fun2alt(x)","2":"132043"},{"1":"fun2(x)","2":"1169626"},{"1":"fun2(x)","2":"1160334"},{"1":"fun2alt(x)","2":"135084"},{"1":"fun2(x)","2":"1149167"},{"1":"fun2(x)","2":"1158376"},{"1":"fun2alt(x)","2":"135293"},{"1":"fun2(x)","2":"1158334"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>

</div>

``` r
library(ggplot2)

benchmark_df <- as.data.frame(benchmark)

ggplot(benchmark_df, aes(x = expr, y = time / 1e6, fill = expr)) +
  geom_boxplot() +
  labs(title = "Execution Time Comparison: fun2 vs. fun2alt",
       x = "Function",
       y = "Execution Time (milliseconds)") +
  theme_minimal() +
  scale_fill_manual(values = c("#FF9999", "#9999FF"))
```

![](lab09-hpc_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

## Problem 3: Parallelize everything

We will now turn our attention to non-parametric
[bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
Among its many uses, non-parametric bootstrapping allow us to obtain
confidence intervals for parameter estimates without relying on
parametric assumptions.

The main assumption is that we can approximate many experiments by
resampling observations from our original dataset, which reflects the
population.

This function implements the non-parametric bootstrap:

``` r
library(parallel)

my_boot <- function(dat, stat, R, ncpus = 1L) {
  
  # Getting the random indices
  n <- nrow(dat)
  idx <- matrix(sample.int(n, n*R, TRUE), nrow=n, ncol=R)
  
  ans <- lapply(seq_len(R), function(i) {
    stat(dat[idx[,i], , drop=FALSE])
  })
  
  # Coercing the list into a matrix
  ans <- do.call(rbind, ans)
  
  ans
}
```

1.  Use the previous pseudocode, and make it work with `parallel`. Here
    is just an example for you to try:

``` r
my_boot_parallel <- function(dat, stat, R, ncpus = 1L) {
  
  # Getting the random indices
  n <- nrow(dat)
  idx <- matrix(sample.int(n, n*R, TRUE), nrow=n, ncol=R)
 
  # Making the cluster using `ncpus`
  # STEP 1:
  cl <- makeCluster(ncpus)
  # STEP 2: GOES HERE
  clusterExport(cl, varlist = c("dat", "stat", "idx"), envir = environment())
  
  # STEP 3: THIS FUNCTION NEEDS TO BE REPLACED WITH parLapply
  ans <- parLapply(cl, seq_len(R), function(i) {
    stat(dat[idx[,i], , drop=FALSE])
  })
  
  # Coercing the list into a matrix
  ans <- do.call(rbind, ans)
  
  # STEP 4: GOES HERE
  stopCluster(cl)
  
  ans
  
}

# Bootstrap of a linear regression model
my_stat <- function(dat) {
  fit <- lm(y ~ x, data = dat)
  coef(fit)
} 

# DATA SIM
set.seed(1)
n <- 500 
R <- 1e4
x <- cbind(rnorm(n)) 
y <- x*5 + rnorm(n)
dat <- data.frame(x = x, y = y)

# Check if we get something similar as lm
ans0 <- confint(lm(y ~ x, data=dat))

# Using non-parallel function
ans1 <- my_boot(dat, my_stat, R)
boot_ci <- apply(ans1, 2, quantile, probs = c(0.025, 0.975))

# Using parallel function
ncpus <- detectCores() - 1
ans2 <- my_boot_parallel(dat, my_stat, R, ncpus)
boot_ci_parallel <- apply(ans2, 2, quantile, probs = c(0.025, 0.975))

print("Confidence intervals from lm:")
```

    ## [1] "Confidence intervals from lm:"

``` r
print(ans0)
```

    ##                  2.5 %     97.5 %
    ## (Intercept) -0.1379033 0.04797344
    ## x            4.8650100 5.04883353

``` r
print("Bootstrap confidence intervals (single-core):")
```

    ## [1] "Bootstrap confidence intervals (single-core):"

``` r
print(boot_ci)
```

    ##       (Intercept)        x
    ## 2.5%  -0.13869034 4.868516
    ## 97.5%  0.04856752 5.043512

``` r
print("Bootstrap confidence intervals (parallel):")
```

    ## [1] "Bootstrap confidence intervals (parallel):"

``` r
print(boot_ci_parallel)
```

    ##       (Intercept)        x
    ## 2.5%   -0.1378909 4.869065
    ## 97.5%   0.0482891 5.044215

2.  Check whether your version actually goes faster than the
    non-parallel version:

``` r
benchmark <- microbenchmark::microbenchmark(
  my_boot(dat, my_stat, R),
  my_boot_parallel(dat, my_stat, R, ncpus),
  times = 5
)

print(benchmark)
```

    ## Unit: seconds
    ##                                      expr      min       lq     mean   median
    ##                  my_boot(dat, my_stat, R) 6.741333 6.795753 6.858891 6.908971
    ##  my_boot_parallel(dat, my_stat, R, ncpus) 2.525347 2.546626 2.614576 2.566572
    ##        uq      max neval
    ##  6.920894 6.927506     5
    ##  2.670820 2.763514     5

``` r
boxplot(benchmark, main = "Execution Time Comparison", ylab = "Time (milliseconds)")
```

![](lab09-hpc_files/figure-gfm/benchmark-problem3-1.png)<!-- -->

We can see from the results that the parallel function runs much faster
than the non-parallel one.

## Problem 4: Compile this markdown document using Rscript

Once you have saved this Rmd file, try running the following command in
your terminal:

``` bash
Rscript --vanilla -e 'rmarkdown::render("[full-path-to-your-Rmd-file.Rmd]")' &
```

Where `[full-path-to-your-Rmd-file.Rmd]` should be replace with the full
path to your Rmd file… :).
