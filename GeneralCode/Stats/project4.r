#Worked with Julie Whitmore
#Question 1
#import csv
gpa <- read.csv(file = "gpa.csv")

#extract columns as appropriate data types
gpaList <- as.double(gpa$gpa)
actList <- strtoi(gpa$act)

#scatterplot
par(mfrow = c(1, 1))
plot(gpaList, actList, xlab = "gpa", ylab = "act", main = "GPA vs ACT")
abline(lm(actList~gpaList), col = "red")

#bootstrap
library(boot)
#calculate sample correlation
cov.npar <- function(x, indices)
{
    gpaList <- as.double(x$gpa)
    gpaBoot <- gpaList[indices]
    actList <- strtoi(x$act)
    actBoot <- actList[indices]
    result <- cor(gpaBoot, actBoot)
    return(result)
}

print(cov.npar.boot <- boot(gpa, cov.npar, R = 999, sim = "ordinary", stype = "i"))

print("Point estimate")
print(mean(cov.npar.boot$t))

print("Bias of point estimate")
print(mean(cov.npar.boot$t) - cov.npar.boot$t0)

print("Std error of point estimate")
print(sd(cov.npar.boot$t))

print(boot.ci(cov.npar.boot, conf = 0.95, type = c("perc")))

#########################################################################
#Question 2
#import csv
x <- read.csv("VOLTAGE.csv")

#remote location data
y <- subset(x, location == 0, select = c(voltage))
#local location data
z <- subset(x, location == 1, select = c(voltage))

#5 point summary of data
print("Summary of remote")
print(summary(y))
print("Summary of local")
print(summary(z))

par(mfrow = c(3, 2))
hist(y[, 1], main = "Remote", freq = FALSE)
hist(z[, 1], main = "Local", freq = FALSE)
boxplot(y[, 1], main = "Remote")
boxplot(z[, 1], main = "Local")
qqnorm(y[, 1], main = "Remote")
qqline(y[, 1])
qqnorm(z[, 1], main = "Local")
qqline(z[, 1])

print("CI lower bound")
print((mean(y[, 1]) - mean(z[, 1])) - 1.96 * sqrt(var(y[, 1]) / length(y[, 1]) + var(z[, 1]) / length(z[, 1])))
print("CI upper bound")
print((mean(y[, 1]) - mean(z[, 1])) + 1.96 * sqrt(var(y[, 1]) / length(y[, 1]) + var(z[, 1]) / length(z[, 1])))

#########################################################################
#Question 3
#import csv
vapor <- read.csv("VAPOR.csv")

#separate data
theoretical <- as.numeric(vapor$theoretical)
experimental <- as.numeric(vapor$experimental)

#5 point summary of data
print("Summary of theoretical")
print(summary(theoretical))
print("Summary of experimental")
print(summary(experimental))

par(mfrow = c(3, 2))
hist(theoretical, main = "Theoretical", freq = FALSE)
hist(experimental, main = "Experimental", freq = FALSE)
boxplot(theoretical, main = "Theoretical")
boxplot(experimental, main = "Experimental")
qqnorm(theoretical, main = "Theoretical")
qqline(theoretical)
qqnorm(experimental, main = "Experimental")
qqline(experimental)

d_mean <- mean(theoretical) - mean(experimental)
d <- theoretical - experimental

print("CI lower bound")
print(d_mean + qt(p = .05 / 2, df = (length(theoretical) - 1)) * sd(d) / sqrt(length(theoretical)))

print("CI upper bound")
print(d_mean - qt(p = .05 / 2, df = (length(theoretical) - 1)) * sd(d) / sqrt(length(theoretical)))