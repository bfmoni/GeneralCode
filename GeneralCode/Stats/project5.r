#Worked with Julie Whitmore
#Question 1
#import csv
data <- read.csv(file = "bodytemp-heartrate.csv")

#PART A
#extract columns as appropriate data types
male_temp <- subset(data, gender == 1, select = c(body_temperature))
female_temp <- subset(data, gender == 2, select = c(body_temperature))

#5 point summary of data
print("Summary of males")
print(summary(male_temp))
print("Summary of females")
print(summary(female_temp))

#barplot, boxplot, QQ plot of males and females
par(mfrow = c(3, 2))
hist(male_temp[, 1], main = "Male Temp", freq = FALSE)
hist(female_temp[, 1], main = "Female Temp", freq = FALSE)
boxplot(male_temp[, 1], main = "Male Temp")
boxplot(female_temp[, 1], main = "Female Temp")
qqnorm(male_temp[, 1], main = "Male Temp")
qqline(male_temp[, 1])
qqnorm(female_temp[, 1], main = "Female Temp")
qqline(female_temp[, 1])

#calculating CI for mean(males) - mean(females)
print("CI lower bound")
print((mean(male_temp[, 1]) - mean(female_temp[, 1])) - 1.96 * sqrt(var(male_temp[, 1]) / length(male_temp[, 1]) + var(female_temp[, 1]) / length(female_temp[, 1])))
print("CI upper bound")
print((mean(male_temp[, 1]) - mean(female_temp[, 1])) + 1.96 * sqrt(var(male_temp[, 1]) / length(male_temp[, 1]) + var(female_temp[, 1]) / length(female_temp[, 1])))

#calculating Hypothesis testing for mean(males) - mean(females)
print("Hypothesis Test")
h <- (mean(male_temp[, 1]) - mean(female_temp[, 1]) - 0) / sqrt(var(male_temp[, 1]) / length(male_temp[, 1]) + var(female_temp[, 1]) / length(female_temp[, 1]))
print(h)

#PART B
#extract columns as appropriate data types
male_heart <- subset(data, gender == 1, select = c(heart_rate))
female_heart <- subset(data, gender == 2, select = c(heart_rate))

#5 point summary of data
print("Summary of males")
print(summary(male_heart))
print("Summary of females")
print(summary(female_heart))

#barplot, boxplot, QQ plot of males and females
par(mfrow = c(3, 2))
hist(male_heart[, 1], main = "Male Heart", freq = FALSE)
hist(female_heart[, 1], main = "Female Heart", freq = FALSE)
boxplot(male_heart[, 1], main = "Male Heart")
boxplot(female_heart[, 1], main = "Female Heart")
qqnorm(male_heart[, 1], main = "Male Heart")
qqline(male_heart[, 1])
qqnorm(female_heart[, 1], main = "Female Heart")
qqline(female_heart[, 1])

#calculating CI for mean(males) - mean(females)
print("CI lower bound")
print((mean(male_heart[, 1]) - mean(female_heart[, 1])) - 1.96 * sqrt(var(male_heart[, 1]) / length(male_heart[, 1]) + var(female_heart[, 1]) / length(female_heart[, 1])))
print("CI upper bound")
print((mean(male_heart[, 1]) - mean(female_heart[, 1])) + 1.96 * sqrt(var(male_heart[, 1]) / length(male_heart[, 1]) + var(female_heart[, 1]) / length(female_heart[, 1])))

#calculating Hypothesis testing for mean(males) - mean(females)
print("Hypothesis Test")
h <- (mean(male_heart[, 1]) - mean(female_heart[, 1]) - 0) / sqrt(var(male_heart[, 1]) / length(male_heart[, 1]) + var(female_heart[, 1]) / length(female_heart[, 1]))
print(h)


#Part C
#male scatterplot
par(mfrow = c(1, 2))
plot(male_temp[, 1], male_heart[, 1], xlab = "Temperature", ylab = "Heart", main = "Male Temp vs Heart")
abline(lm(male_heart[, 1]~male_temp[, 1]), col = "red")
#correlation
print("Male Correlation")
print(cor(male_temp[, 1], male_heart[, 1]))

#female scatterplot
plot(female_temp[, 1], female_heart[, 1], xlab = "Temperature", ylab = "Heart", main = "Female Temp vs Heart")
abline(lm(female_heart[, 1]~female_temp[, 1]), col = "red")
#correlation
print("Female Correlation")
print(cor(female_temp[, 1], female_heart[, 1]))

#QUESTION 2#######################################################################
#PART A/B
# z-interval
# n is number of samples, l is lambda
conf.int <- function(n, l)
{
    #draw n samples
    x <- rexp(n, rate = l)
    #calculate z confidence interval
    ci <- mean(x) + c(-1, 1) * 1.96 * sqrt(var(x)) / sqrt(n)
    #check if CI contains actual mean
    if(ci[1] <= 1 / l & 1 / l <= ci[2])
    {
        return(1)
    }
    return(0)
}

n_list <- c(5, 10, 30, 100)
l_list <- c(0.01, 0.1, 1, 10)
nsim <- 5000
temp <- NULL
results <- NULL

#iterate through all combinations of n and lambda
for(n in n_list)
{
    for(l in l_list)
    {
        z_conf <- replicate(nsim, conf.int(n, l))
        temp <- cbind(temp, sum(z_conf) / nsim)
    }
    results <- rbind(results, temp)
    temp <- NULL
}
print("Z Interval")
rownames(results) <- n_list
colnames(results) <- l_list

print(results)



conf.int.boot <- function(n, l)
{
    #estimate a mean
    q <- mean(rexp(n, rate = l))
    sample_mean = NULL
    #create a population of means based on estimated means
    for(i in 1:80)
    {
        x <- rexp(n, rate = 1/q)
        sample_mean <- append(sample_mean, mean(x))
    }
    sample_mean <- sort(sample_mean)
    return(sample_mean)
}

conf.int.boot.calc <- function(n, l)
{
    #create a population of means
    x <- conf.int.boot(n,l)
    #check to see if 1/l is in the percintile bootstrap
    if(x[length(x)*.025] <= 1/l & x[length(x) * .975] >= 1/l)
    {
        return(1)
    }
    return(0)
}


temp <- NULL
results <- NULL

#iterate through all combinations of n and lambda
for(n in n_list)
{
    for(l in l_list)
    {
        means <- replicate(nsim, conf.int.boot.calc(n, l))


        temp <- cbind(temp, (sum(means) / nsim))
    }
    results <- rbind(results, temp)
    temp <- NULL
}
print("Parametric Percentile")
rownames(results) <- n_list
colnames(results) <- l_list
print(results)