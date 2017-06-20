require(sfsmisc)
require(mgcv)
svd_file <- "/home/giodiro/Desktop/CIL/exercises/ex2copy/saved_data/rsearch/svd_rsearch_configs.csv"
svdpp_file <- "/home/giodiro/Desktop/CIL/exercises/ex2copy/saved_data/rsearch/SVD++_rsearch_configs.csv"
integrated_file <- "/home/giodiro/Desktop/CIL/exercises/ex2copy/saved_data/rsearch/integrated_rsearch_configs.csv"
simple_file <- "/home/giodiro/Desktop/CIL/exercises/ex2copy/saved_data/rsearch/simple_rsearch_configs.csv"

analyze <- function (file, name) {
  data <- read.csv(file, header=TRUE)
  data.used <- subset(data, select=-c(num_threads, nusers, nitems))
  data.valid <- data.used[which(data.used$score < 2),]
  best_score <- data.used[which.min(data.used$score),]$score
  print(paste("Best score for", name, "is:", best_score))
  print(data.used[which.min(data.used$score),])
  data.valid
}

data.svd <- analyze(svd_file, "SVD")
data.svdpp <- analyze(svdpp_file, "SVD++")
data.integrated <- analyze(integrated_file, "Integrated")
data.simple <- analyze(simple_file, "Simple model")

## Analysis:

## Simple model
data.simple <- subset(data.simple, select=-c(max_iter))
pairs(data.simple, gap=0, pch=18)
# the lrate* and lrate_reduction parameters seem to have little effect.

# the regl7 parameter seems the most important in this model, and should be set around 0.01 for best results
# regl7 is a regularizer penalizing large magnitude of the user and item vectors (see sgd.cpp line 250)
pairs(data.simple[which(data.simple$score < 1.0),][c("score", "regl7")], gap=0, pch=18)
# the num_factors parameter has a slightly positive correlation with accuracy, however after num_factors=60 there 
# seem to be little benefits.

## Integrated model
data.integrated <- subset(data.integrated, select=-c(correlation_shrinkage, K2, K1))
pairs(data.integrated, gap=0, pch=18)
# For the integrated model less data (650 evaluations) is available due to the amount of time it takes to run it.
# Furthermore there seems to be less variation in the scores with different parameters than in the simple model
# note that this could just be a spurious correlation due to the exclusion of parameters generating errors 
# (which could happen more often in the integrated model).
# The integrated model achieves best results: minimum error is 0.976816

## SVD++ model
pairs(data.svdpp, gap=0, pch=18)
# As with the simple model regl7 is very important, however here it should be set around 0.06.
# Other parameters have little effect
# SVD++ achieves the second best result: 0.979596

## SVD model
data.svd <- subset(data.svd, select=-c(max_iter))
pairs(data.svd, gap=0, pch=18)
# Here it is very easy to see correlations thanks to large (~20000) amounts of data.
# num_factors is the most influent parameter, with a optimal setting around 19.
# Follows K2, which is optimally set at 16.
# K1 has a linear relationship with the score so we set it at the lowest possible value of 
# 1 where it doesn't affect regularization.


### Other random cruft
#data.used <- subset(data, select=-c(correlation_shrinkage, num_threads, nusers, nitems, K1, K2))
data.used <- subset(data, select=-c(max_iter, num_threads, nusers, nitems))
data.scoring <- subset(data.used, select=-c(score))
data.scoring$score <- factor(data$score > 10, labels = c("low", "high"))
pairs(data.used, gap=0, pch=18, col = c("green", "red")[data.scoring$score])

data.valid <- data.used[which(data.used$score < 2),]

for (cn in names(data.valid)) {
  if (cn != "score") {
    data.valid[[cn]] <- log(data.valid[[cn]])
  }
}

pairs(data.valid, gap=0, pch=18)

form <- formula(score ~ .)
gamForm <- wrapFormula(form, data=data.valid, wrapString = "s(*)")

fl <- lm(form, data = data.valid)

g1 <- gam(gamForm, data = data.valid)
par(mfrow = c(3,3), mgp = c(1.5, 0.6, 0), mar = 0.1 + c(3, 3, 1, 1),
    oma = c(0,0, 2.5, 0))
plot(g1, pages = 1, shade = TRUE)

# Best score
data.used[which.min(data.used$score),]

