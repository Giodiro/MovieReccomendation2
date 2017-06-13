require(sfsmisc)
require(mgcv)
file <- "/home/gmeanti/Desktop/CIL/exercises/ex2/saved_data/rsearch_configs.csv"

data <- read.csv(file, header=TRUE)
data.used <- subset(data, select=-c(correlation_shrinkage, num_threads, nusers, nitems, K1, K2))
data.scoring <- subset(data.used, select=-c(score))
data.scoring$score <- factor(data$score > 10, labels = c("low", "high"))
pairs(data.used, gap=0, pch=18, col = c("green", "red")[data.scoring$score])

data.valid <- data.used[which(data.used$score < 1),]

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
