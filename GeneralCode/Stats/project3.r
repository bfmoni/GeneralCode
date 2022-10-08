#Worked with Julie Whitmore
#QUESTION 1

nlist <- c(1, 2, 3, 5, 10, 30)
thetalist <- c(1, 5, 50, 100)
results <- matrix(nrow = 2 * length(nlist), ncol = 4)

#results = n(mle), n(mme) x theta

j <- 1
#iterate through all possible combinations of n, theta
for (n in nlist)
{
    i <- 1

    for (theta in thetalist)
    {
        #reset arrays
        mme <- NULL
        mle <- NULL
        MSEmme <- NULL
        MSEmle <- NULL
        x <- NULL
        print("n:")
        print(n)
        print("theta:")
        print(theta)
        #do 1000 draws of current (n, theta)
        x <- replicate(1000, runif(n, 0, theta))

        #if n=1, array is 1D, special case
        if (n == 1)
        {
            MSEmme <- (x - theta) ^ 2
            MSEmme <- sum(MSEmme) / 1000
            results[j * 2 - 1, i] <- MSEmme

            MSEmle <- (x - theta) ^ 2
            MSEmle <- sum(MSEmle) / 1000
            results[j * 2, i] <- MSEmle

            i <- i + 1
        } else
        {
            #mean of each column * 2
            mme <- apply(x, 2, mean)
            mme <- 2 * mme
            #max of each column
            mle <- apply(x, 2, max)

            #calculate mse of mme and mle
            MSEmme <- (mme - theta) ^ 2
            MSEmme <- sum(MSEmme) / 1000
            results[j * 2 - 1, i] <- MSEmme

            MSEmle <- (mle - theta) ^ 2
            MSEmle <- sum(MSEmle) / 1000
            results[j * 2, i] <- MSEmle

            i <- i + 1
        }
    }

    j <- j + 1
}

print(results)

##### N VS MSE #####
par(mfrow = c(2, 2))
#all n, theta = 1
#MME
plot(nlist, results[seq(nrow(results)) %% 2 == 1, 1], type="l", xlab = "n", ylab = "MSE", main = "Theta = 1")
#MLE
lines(nlist, results[seq(nrow(results)) %% 2 == 0, 1], col = "red")
legend("topright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

#all n, theta = 5
#MME
plot(nlist, results[seq(nrow(results)) %% 2 == 1, 2], type="l", xlab = "n", ylab = "MSE", main = "Theta = 5")
#MLE
lines(nlist, results[seq(nrow(results)) %% 2 == 0, 2], col = "red")
legend("topright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

#all n, theta = 50
#MME
plot(nlist, results[seq(nrow(results)) %% 2 == 1, 3], type="l", xlab = "n", ylab = "MSE", main = "Theta = 50")
#MLE
lines(nlist, results[seq(nrow(results)) %% 2 == 0, 3], col = "red")
legend("topright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

#all n, theta = 100
#MME
plot(nlist, results[seq(nrow(results)) %% 2 == 1, 4], type="l", xlab = "n", ylab = "MSE", main = "Theta = 100")
#MLE
lines(nlist, results[seq(nrow(results)) %% 2 == 0, 4], col = "red")
legend("topright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))


##### THETA VS MSE #####
par(mfrow = c(3, 2))
#all theta, n = 1
#MME
plot(thetalist, results[1, ], type = "l", xlab = "theta", ylab = "MSE", main = "n = 1")
#MLE
lines(thetalist, results[2, ], col = "red")
legend("bottomright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

#all theta, n = 2
#MME
plot(thetalist, results[3, ], type = "l", xlab = "theta", ylab = "MSE", main = "n = 2")
#MLE
lines(thetalist, results[4, ], col = "red")
legend("bottomright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

#all theta, n = 3
#MME
plot(thetalist, results[5, ], type = "l", xlab = "theta", ylab = "MSE", main = "n = 3")
#MLE
lines(thetalist, results[6, ], col = "red")
legend("bottomright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

#all theta, n = 5
#MME
plot(thetalist, results[7, ], type = "l", xlab = "theta", ylab = "MSE", main = "n = 5")
#MLE
lines(thetalist, results[8, ], col = "red")
legend("bottomright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

#all theta, n = 10
#MME
plot(thetalist, results[9, ], type = "l", xlab = "theta", ylab = "MSE", main = "n = 10")
#MLE
lines(thetalist, results[10, ], col = "red")
legend("bottomright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

#all theta, n = 30
#MME
plot(thetalist, results[11, ], type = "l", xlab = "theta", ylab = "MSE", main = "n = 30")
#MLE
lines(thetalist, results[12, ], col = "red")
legend("bottomright", legend = c("MME", "MLE"), lwd = 3, col = c("black", "red"))

######################################################

#QUESTION 2

result <- NULL
z <- c(21.72, 14.65, 50.42, 28.78, 11.23)

#negative log likelihood function to minimize
neg.loglik.fun <- function(par, dat)
{
    result <- length(dat) * log(par) - par * sum(log(dat)) - sum(log(dat))
    return(-result)
}

#minimization function
ml.est <- optim(par=1, fn=neg.loglik.fun, method = "L-BFGS-B", lower = 0.1, hessian=TRUE, dat=z)
print(ml.est)

#standard error
se <- sqrt(diag(solve(ml.est$hessian)))
print("Standard error")
print(se)

#confidence interval
alpha <- 0.05
conf <- ml.est$par + c(-1, 1) * qnorm(1 - (alpha / 2)) * se / sqrt(length(z))
print("Confidence Interval")
print(conf)