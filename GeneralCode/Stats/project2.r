#Worked with Julie Whitmore
#import csv
roadrace <- read.csv(file = "roadrace.csv")
#count the number of maine and away runners
maine <- c(sum(roadrace$Maine == "Maine"), sum(roadrace$Maine == "Away"))
#add labels to values
bar_maine <- c(Maine = maine[1], Away = maine[2])
#plot values (adjusting ylim so the bars do not extend past the table)
barplot(bar_maine, ylim = c(0, maine[1] + maine[2]))

print("Maine runners")
print(maine[1])
print("Away runners")
print(maine[2])

######################################################

#get the minute values of maine and away groups
maine_mins <- subset(roadrace[, 12], roadrace$Maine == "Maine")
away_mins <- subset(roadrace[, 12], roadrace$Maine == "Away")

#histogram with same ylim so scale remains the same
par(mfrow = c(1, 2))
hist(maine_mins, xlab = "Minutes", ylim = c(0, 2000))
hist(away_mins, xlab = "Minutes", ylim = c(0, 2000))

print("MAINE")
#min, Q1, median, Q3, max
print("min, Q1, median, Q3, max")
print(fivenum(maine_mins))
#standard deviation
print("standard deviation")
print(sd(maine_mins))
#variance
print("variance")
print(var(maine_mins))
#mean
print("mean")
print(mean(maine_mins))
#interquartile range
print("interquartile range")
print(IQR(maine_mins))


print("AWAY")
#min, Q1, median, Q3, max
print("min, Q1, median, Q3, max")
print(fivenum(away_mins))
#standard deviation
print("standard deviation")
print(sd(away_mins))
#variance
print("variance")
print(var(away_mins))
#mean
print("mean")
print(mean(away_mins))
#interquartile range
print("interquartile range")
print(IQR(away_mins))

##########################################################

#boxplot with same ylim so scale remains the same
par(mfrow = c(1, 2))
boxplot(maine_mins, xlab = "Maine", ylim = c(0, 160))
boxplot(away_mins, xlab = "Away", ylim = c(0, 160))

##########################################################

male_age <- subset(strtoi(roadrace$Age), roadrace$Sex == "M")
female_age <- subset(strtoi(roadrace$Age), roadrace$Sex == "F")

#boxplot with same ylim so scale remains the same
par(mfrow = c(1, 2))
boxplot(male_age, xlab = "Male", ylim = c(0, 100))
boxplot(female_age, xlab = "Female", ylim = c(0, 100))

print("MALE")
#min, Q1, median, Q3, max
print("min, Q1, median, Q3, max")
print(fivenum(male_age))
#standard deviation
print("standard deviation")
print(sd(male_age))
#variance
print("variance")
print(var(male_age))
#mean
print("mean")
print(mean(male_age))
#interquartile range
print("interquartile range")
print(IQR(male_age))


print("FEMALE")
#min, Q1, median, Q3, max
print("min, Q1, median, Q3, max")
print(fivenum(female_age))
#standard deviation
print("standard deviation")
print(sd(female_age))
#variance
print("variance")
print(var(female_age))
#mean
print("mean")
print(mean(female_age))
#interquartile range
print("interquartile range")
print(IQR(female_age))

##########################################################
#import csv
motorcycle <- read.csv(file = "motorcycle.csv")
par(mfrow = c(1, 1))
boxplot(motorcycle[, 2], xlab = "Fatal Accidents")

#min, Q1, median, Q3, max
print("MOTORCYCLE")
print("min, Q1, median, Q3, max")
print(fivenum(motorcycle[, 2]))

#standard deviation
print("standard deviation")
print(sd(motorcycle[, 2]))

#variance
print("variance")
print(var(motorcycle[, 2]))

#mean
print("mean")
print(mean(motorcycle[, 2]))

#interquartile range
print("interquartile range")
print(IQR(motorcycle[, 2]))

##########################################################

summary <- fivenum(motorcycle[, 2])
motor_IQR <- 1.5 * IQR(motorcycle[, 2])
trouble_county <- subset(motorcycle[, 1], motorcycle[, 2] < (summary[2] - motor_IQR) | motorcycle[, 2] > (summary[4] + motor_IQR))
print(trouble_county)