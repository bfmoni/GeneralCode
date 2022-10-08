#Worked with Julie Whitmore
# psa is label
# quantitative
# - cancervol
# - weight
# - age
# - benpros
# - cpspen
# - gleason

# qualitative
# - vesinv

prostate_set <- read.csv("prostate_cancer.csv")
attach(prostate_set)

#############################
# try all features together #
#############################

fit_all <- lm(psa ~ cancervol + weight + age + benpros + vesinv + capspen + gleason)
# print(summary(fit_all))

################################################################
# try the seemingly 2 most meaningful features from approach 1 #
################################################################

fit_min <- lm(psa ~ cancervol + vesinv)
# print(summary(fit_min))

##############################################################
# look at what each feature represents and guess what is     #
# meaningful with inference from lm of all features together #
##############################################################

fit_guess <- lm(psa ~ cancervol + vesinv + capspen + gleason)
# print(summary(fit_guess))

####################################
# compare the models to each other #
####################################

# print(anova(fit_min, fit_guess, fit_all))

########################################
# check assumptions by checking errors #
# and normal distribution of error     #
########################################

# par(mfrow =  c(1, 2))
# plot(fitted(fit_guess), resid(fit_guess))
# abline(h = 0)

# qqnorm(resid(fit_guess))
# qqline(resid(fit_guess))

########################################
# try taking log of psa due to q-qplot #
# of best model not following correct  #
# assumptions.                         #
# repeat process with log transform    #
########################################

################
# all features #
################

fit_all_update <- update(fit_all, log(psa) ~ .)
# print(summary(fit_all_update))

#########################################################
# log transformed psa "best" features from all_features #
#########################################################

fit_min_updated <- lm(log(psa) ~ cancervol + benpros + vesinv + gleason)
# print(summary(fit_min_updated))

##############################################################
# look at what each feature represents and guess what is     #
# meaningful with inference from lm of log(psa) all features #
##############################################################

fit_guess_updated <- lm(log(psa) ~ cancervol + benpros + vesinv + gleason + capspen)
# print(summary(fit_guess_updated))

##################
# compare models #
##################

# print(anova(fit_all_update, fit_guess_updated, fit_min_updated))

###############################################
# while similar take fit_min_updated as model #
# check model assumptions                     #
###############################################

# print(summary(fit_min_updated))
# par(mfrow =  c(1, 2))
# plot(fitted(fit_min_updated), resid(fit_min_updated))
# abline(h = 0)

# qqnorm(resid(fit_min_updated))
# qqline(resid(fit_min_updated))

################################################################
# Assumptions seem to hold use the final model to predict the  #
# PSA level for a patient whose quantitative predictors are    #
# the sample means of the variables and qualitative predictors #
# are the most frequent category                               #
# cancervol, benpros, vesinv, gleason                          #
################################################################

temp <- if(sum(vesinv) > length(vesinv) / 2) 1 else 0
xValues <- data.frame(cancervol = c(mean(cancervol)),
                      benpros = c(mean(benpros)),
                      vesinv = c(temp),
                      gleason = c(mean(gleason)))
print(xValues)
print(predict(fit_min_updated, newdata = xValues))