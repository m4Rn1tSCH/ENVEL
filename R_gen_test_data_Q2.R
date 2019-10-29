Age - sample(x=16, size = 500000, replace = TRUE)
Student - sample(x=01, size = 500000, replace = TRUE)

#RNG setup
RNGkind(kind = NULL, normal.kind = NULL, sample.kind = NULL)
RNGversion(vstr)
set.seed(seed, kind = NULL, normal.kind = NULL, sample.kind = NULL)
Random_seed <- c(rng.kind, n1, n2)
#Set up the artificial data set
#uses normal distribution with number of observations
#AMOUNTS IN USD!
#Account Balance
#sample: simulate different balances
#u100: less than USD 100
#o100u10,000: more than USD 100 - less than USD 10,000
#o10,000: more than USD 10,000
transactionCount 
transactionId 
masterId
customerId
typeCode
tag
friendlyDescription
description
status
createdDate 
amount_total 
CS 
Gas_EXP1 
DIN1 
TRANS1 
ENT1 
GROC1 
HOTEL1 
AIR1 
isCredit 
settledDate 
availableDate 
voidedDate 
returnCode 
feeCode 
feeDescription 
cardId 
subTypeCode 
subType 
institutionName 
check
AccountBalance <- sample(x=c(u100, o100u1000, o1000u10000, o10000), size = 500000, replace = TRUE)
CS <- sample(x=c(Average, Excellent, Poor), size = 500000, replace = TRUE)
#Gas first
#httpscars.usnews.comcars-trucksdaily-news110505-americans-spend-386-09-monthly-on-gasoline
Gas_EXP1 <- rnorm(500000, mean = 4633.08, sd=400)

#Dining first
#httpswww.thesimpledollar.comdont-eat-out-as-often-188365
DIN1 <- rnorm(500000, mean = 2784, sd =300)

#Transportation First
#httpswww.mic.comarticles180000how-much-money-should-you-spend-on-commuting-here-are-average-american-transportation-costs
TRANS1 <- rnorm(500000, mean=3655, sd = 400)

#Entertainment First
#httpswww.marketwatch.comstorypostrecession-this-is-what-americans-are-really-spending-their-money-on-2017-12-13
ENT1 <- rnorm(500000, mean=2539.23, sd = 400)

#Groceries First
#httpswww.creditdonkey.comaverage-cost-food-per-month.html
GROC1 <- rnorm(500000, mean=3000, sd = 600)

#Hotels First
#httpswww.valuepenguin.comaverage-cost-vacation
HOTEL1 <- rnorm(500000, mean = 581, sd = 100)

#Airplane First
AIR1 <- rnorm(500000, mean = 1270.78, sd=300)
#######################################

#produce the data frame and insert the values
CreditCard - NA
#produce the data frame with following columns
#directly taken from Q2 API
#OLD>>> CreditCard, Age, Student, AccountBalance, CS, Gas_EXP1, DIN1, TRANS1, ENT1, GROC1, HOTEL1, AIR1 <<<
create <- data.frame(transactionCount, transactionId, masterId, customerId, typeCode, tag, friendlyDescription, 
                    description, status, createdDate, amount_total, CS, Gas_EXP1, DIN1, TRANS1, ENT1, GROC1, 
                    HOTEL1, AIR1, isCredit, settledDate, availableDate, voidedDate, returnCode, feeCode, 
                    feeDescription, cardId, subTypeCode, subType, institutionName, check)
#generate the columns
#uniform distribution
#split up the expenses in categories
#second set
Randnum1 <- runif(500000, -0.3, 0.5)
create$Gas_EXP2 - round((create$Gas_EXP1 + (create$Gas_EXP1Randnum1) + (create$Student1500)), digits = 2)
Randnum2 <- runif(500000, -0.3, 0.5)
create$DIN2 <- round((create$DIN1 + (create$DIN1Randnum2)), digits=2)
Randnum3 <- runif(500000, -0.3, 0.5)
create$TRANS2 <- round((create$TRANS1 + (create$TRANS1Randnum3) + (create$Student500000)), digits =2)
Randnum4 <- runif(500000, -0.3, 0.5)
create$ENT2 <- round((create$ENT1 + (create$ENT1Randnum4)), digits =2)
Randnum5 <- runif(500000, -0.3, 0.5)
create$GROC2 <- round((create$GROC1 + (create$GROC1Randnum5) + (create$Student800)), digits = 2)
Randnum6 <- runif(500000, -0.3, 0.5)
create$HOTEL2 <- round((create$HOTEL1 + (create$HOTEL1Randnum6)), digits = 2)
Randnum7 <- runif(500000, -0.3, 0.5)
create$AIR2 <- round((create$AIR1 + (create$AIR1Randnum7) + (create$Student500)), digits =2)
#third set
Randnum1 <- runif(500000, -0.3, 0.5)
create$Gas_EXP3 <- round((create$Gas_EXP2 + (create$Gas_EXP2Randnum1) + (create$Student1500)), digits = 2)
Randnum2 <- runif(500000, -0.3, 0.5)
create$DIN3 <- round((create$DIN2 + (create$DIN2Randnum2)), digits = 2)
Randnum3 <- runif(500000, -0.3, 0.5)
create$TRANS3 <- round((create$TRANS2 + (create$TRANS2Randnum3) + (create$Student500000)), digits = 2)
Randnum4 <- runif(500000, -0.3, 0.5)
create$ENT3 <- round((create$ENT2 + (create$ENT2Randnum4)), digits =2)
Randnum5 <- runif(500000, -0.3, 0.5)
create$GROC3 <- round ((create$GROC2 + (create$GROC2Randnum5) + (create$Student800)), digits =2)
Randnum6 <- runif(500000, -0.3, 0.5)
create$HOTEL3 <- round((create$HOTEL2 + (create$HOTEL2Randnum6)), digits =2)
Randnum7 <- runif(500000, -0.3, 0.5)
create$AIR3 <- round((create$AIR2 + (create$AIR2Randnum7) + (create$Student500)), digits =2)

#simulate various user experiences
create$studentexp1 - NA
create$studentexp2 - NA
create$studentexp3 - NA
#################################

for(val in nrow(create)){
  if(create$Student[val] == 1){
    create$studentexp1[val] - runif(1, 8500, 21050)
    create$studentexp2[val] - create$studentexp1[val] + runif(1, -0.3, 0.3)create$studentexp1[val]
    create$studentexp3[val] - create$studentexp2[val] + runif(1, -0.3, 0.3)create$studentexp2[val]
    
  }
  else{
    create$studentexp1[val] - 0
    create$studentexp2[val] - 0
    create$studentexp3[val] - 0
  }
}

avg - create[create$CS == Average, ]
exc - create[create$CS == Excellent, ]

avg$totalqs1 - NA
avg$totaljourneyover - NA

for(val in nrow(avg)){
  if(avg$Student[val] == 1){
    avg$totaljourneyover[val] - avg$studentexp1[val]  0.0125 + avg$studentexp2[val]  0.0125 + avg$studentexp3[val]0.0125
  }
  else{
    avg$totaljourneyover[val] - 0
  }
}

for (val2 in nrow(avg)){
  if(avg$Student[val2] == 1){
    avg$totalqs1[val2] - avg$studentexp1[val2]  0.015 + avg$studentexp2[val2]  0.015 + avg$studentexp3[val2]0.015 - 39 - 39 - 39
  }
  else{
    avg$totalqs1[val2] - avg$Gas_EXP1[val2]  0.015 + avg$DIN1[val2]0.015 + avg$TRANS1[val2]0.015 + avg$ENT1[val2]0.015 + avg$GROC1[val2]0.015 + avg$HOTEL1[val2]0.015 + avg$AIR1[val2]0.015 - 39
    + avg$Gas_EXP2[val2]  0.015 + avg$DIN2[val2]0.015 + avg$TRANS2[val2]0.015 + avg$ENT2[val2]0.015 + avg$GROC2[val2]0.015 + avg$HOTEL2[val2]0.015 + avg$AIR2[val2]0.015 - 39
    + avg$Gas_EXP3[val2]  0.015 + avg$DIN3[val2]0.015 + avg$TRANS3[val2]0.015 + avg$ENT3[val2]0.015 + avg$GROC3[val2]0.015 + avg$HOTEL3[val2]0.015 + avg$AIR3[val2]0.015 - 39
  }
}

avg$qstotal - 0
avg$bptotal - 0
avg$savorone - 0
avg$ventureone - 0

#############excellent cards########################
exc$totalqs1 - 0
exc$totaljourneyover - 0
exc$qstotal - NA

for(val in nrow(exc)){
  exc$qstotal[val] - exc$Gas_EXP1[val]  0.015 + exc$DIN1[val] 0.015 + exc$TRANS1[val]  0.015 + exc$ENT1[val]  0.015 + exc$GROC1[val]  0.015 + exc$HOTEL1[val]  0.015 + exc$AIR1[val]  0.015
  + exc$Gas_EXP2[val]  0.015 + exc$DIN2[val] 0.015 + exc$TRANS2[val]  0.015 + exc$ENT2[val]  0.015 + exc$GROC2[val]  0.015 + exc$HOTEL2[val]  0.015 + exc$AIR2[val]  0.015 + 
    exc$Gas_EXP3[val]  0.015 + exc$DIN3[val] 0.015 + exc$TRANS3[val]  0.015 + exc$ENT3[val]  0.015 + exc$GROC3[val]  0.015 + exc$HOTEL3[val]  0.015 + exc$AIR3[val]  0.015
}

exc$bptotal - NA
for(val in nrow(exc)){
  spend1 - exc$Gas_EXP1[val] + exc$DIN1[val] + exc$TRANS1[val] + exc$ENT1[val] + exc$GROC1[val] + exc$HOTEL1[val] + exc$AIR1[val]
  spend2 - exc$Gas_EXP2[val] + exc$DIN2[val] + exc$TRANS2[val] + exc$ENT2[val] + exc$GROC2[val] + exc$HOTEL2[val] + exc$AIR2[val]
  spend3 - exc$Gas_EXP3[val] + exc$DIN3[val] + exc$TRANS3[val] + exc$ENT3[val] + exc$GROC3[val] + exc$HOTEL3[val] + exc$AIR3[val]
  
  if(spend1 = 5000){
    save1 - spend1  0.05
  }
  else{
    save1 - (5000  0.05) + ((spend1 - 5000)  0.02)
  }
  
  if(spend2 = 5000){
    save2 - spend2  0.05
  }
  else{
    save2 - (5000  0.05) + ((spend2 - 5000)  0.02)
  }
  
  if(spend3 = 5000){
    save3 - spend3  0.05
  }
  else{
    save3 - (5000  0.05) + ((spend3 - 5000)  0.02)
  }
  
  exc$bptotal[val] - save1 + save2 + save3
}

exc$savorone - NA

for(val in nrow(exc)){
  exc$savorone - (exc$DIN1 + exc$DIN2 + exc$DIN3)  0.03 + (exc$GROC1 + exc$GROC2 + exc$GROC3)  0.02 + 
    (exc$Gas_EXP1 + exc$TRANS1 + exc$ENT1 + exc$HOTEL1 + exc$AIR1 + exc$Gas_EXP2 + exc$TRANS2 + exc$ENT2 + exc$HOTEL2 + exc$AIR2
     + exc$TRANS3 + exc$ENT3 + exc$HOTEL3 + exc$AIR3)  0.01 - 95 - 95 - 95
}

exc$ventureone - NA
for(val in nrow(exc)){
  spend1 - exc$Gas_EXP1[val] + exc$DIN1[val] + exc$TRANS1[val] + exc$ENT1[val] + exc$GROC1[val] + exc$HOTEL1[val] + exc$AIR1[val]
  spend2 - exc$Gas_EXP2[val] + exc$DIN2[val] + exc$TRANS2[val] + exc$ENT2[val] + exc$GROC2[val] + exc$HOTEL2[val] + exc$AIR2[val]
  spend3 - exc$Gas_EXP3[val] + exc$DIN3[val] + exc$TRANS3[val] + exc$ENT3[val] + exc$GROC3[val] + exc$HOTEL3[val] + exc$AIR3[val]
  
  airspend1 - spend1  1.25
  airspend2 - spend2  1.25
  airspend3 - spend3  1.25
  
  airmon1 - airspend1  0.013
  airmon2 - airspend2 0.013
  airmon3 - airspend3  0.013
  
  if(airmon1 = exc$AIR1[val]){
    saving1 - exc$AIR1[val]
  }
  else{
    saving1 - airmon1
  }
  
  if(airmon2 = exc$AIR2[val]){
    saving2 - exc$AIR2[val]
  }
  else{
    saving2 - airmon2
  }
  
  if(airmon3 = exc$AIR3[val]){
    saving3 - exc$AIR3[val]
  }
  else{
    saving3 - airmon3
  }
  
  exc$ventureone[val] - saving1 + saving2 + saving3 - 95 - 95 - 95
}

for(val in nrow(exc)){
  if((exc$AccountBalance[val] == o10000) && (exc$bptotal[val]  exc$qstotal[val]) && (exc$bptotal[val]  exc$ventureone[val]) && (exc$bptotal[val]  exc$savorone[val])){
    exc$CreditCard[val] = BuyPower
  }
  else if((exc$ventureone[val]  exc$qstotal[val]) && (exc$ventureone[val]  exc$savorone[val])){
    exc$CreditCard[val] = VentureOne
  }
  else if((exc$savorone[val]  exc$qstotal[val])){
    exc$CreditCard[val] = SavorOne
  }
  else{
    exc$CreditCard[val] = QuickSilver
  }
}

for(val in nrow(avg)){
  if(avg$totalqs1[val]  avg$totaljourneyover[val]){
    avg$CreditCard[val] = QuickSilverOne
  }
  else{
    avg$CreditCard[val] = JourneyStudent
  }
}

total - rbind(avg, exc)

write.csv(total, file = total.csv)