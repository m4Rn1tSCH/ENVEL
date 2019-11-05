require(magrittr)
######TEST DATA SET#########

#finished: 11/5/2019
#R-version: 3.6.1 (x64)

#set up dates in a way that start_date <= end_date
#difference between the two dates determines the length of the data frame
#alternative: enter observation_size manually

####SET DATES#####
start_date <- as.Date("2019-11-01")
end_date <- Sys.Date()
#end_date <- as.Date("2019-11-01")

void_date <- "5days"
settlement_date <- "2days"

#set size of the data set
observation_size = end_date - start_date
#manual input for the data set length (int only)
#observation_size = 50000



#############DEFINITION VARIABLES################
#uses normal distribution with number of observations
#AMOUNTS IN USD!
#Account Balance
#sample: simulate different balances
#u100: less than USD 100
#o100u10,000: more than USD 100 - less than USD 10,000
#o10,000: more than USD 10,000
#balance encoding
u100 <- 4
o100u1000 <- 3 
o1000u10000 <- 2
o10000 <- 1

#traditional credit score encoding (FICO)
Excellent <- 01
Average <- 02
Poor <- 03

#credit score encoding
Exceptional_850_800 <- 1
Very_Good_799_740 <- 2
Average_701 <- 3
Good_739_670 <- 4
Fair_669_580 <- 5
Very_Poor_579_300 <- 6

#user experience encoding
highly_satisfied <- 2
satisfied <- 1
neutral <- 0
dissatified <- -1
highly_dissatisfied <- -2

#type (human-readable) encoding
CorePro_Deposit <- "CorePro Deposit"
CorePro_Withdrawal <- "CorePro Withdrawal"
Internal_CorePro_Transfer <- "Internal CorePro Transfer"
Interest_Paid <- "Interest Paid"
CorePro_Recurring_Withdrawal <- "CorePro Recurring Withdrawal"
Manual_Adjustment <- "Manual Adjustment"
Interest_Adjustment <- "Interest Adjustment"
 
transaction_draw <- c(CorePro_Deposit, 
                      CorePro_Withdrawal, 
                      Internal_CorePro_Transfer, 
                      Interest_Paid, 
                      CorePro_Recurring_Withdrawal,
                      Manual_Adjustment,
                      Interest_Adjustment)


#status encoding#
Initiated <- 0  #transaction created but not yet in NACHA file
Pending <- 100  #transaction created but amended to NACHA file
Settled <- 200  #transaction has been posted to the account
Voided <- 300   #transaction has been voided 

#fee code
#RGD-Regulation_D_fee
RGD <- 099
#RTN-Return_Item_fee
RTN <- 095
#NSF-Insufficient_Funds_fee
NSF <- 101

############RNG SETUP##############
##kind - character or NULL. 
#If kind is a character string, set R's RNG to the kind desired. Use "default" to return to the
#R default.
#normal.kind - 	character string or NULL. If it is a character string,
#set the method of Normal generation. Use "default" to return to the R default. NULL makes no change.
#sample.kind - character string or NULL.
#If it is a character string, set the method of discrete uniform generation 
#(used in sample, for instance). Use "default" to return to the R default. NULL makes no change.
#seed - a single value, interpreted as an integer, or NULL (see 'Details').
#vstr - a character string containing a version number, e.g., "1.6.2". The default RNG configuration of the current R version is used if vstr is greater than the current version.
#rng.kind - integer code in 0:k for the above kind.

.Random.seed <- c(rng.kind = "default", n1 = 0, n2 = 12)
RNGkind(kind = "Mersenne-Twister", normal.kind = "Box-Muller", sample.kind = NULL)
RNGversion(getRversion())
#for reproducibility set seed and identical normal.kind values
set.seed(seed = 12, kind = "Mersenne-Twister", normal.kind = 'Box-Muller', sample.kind = NULL)
#certain columns will ony take a single sample value to imitiate unique IDs

#######YES/NO SAMPLE################
yn_draw <- gl(n = 2, k = 100, labels = c("Y","N"))

##########BANK NAME LIST#################
#labels needs a vector
list_banks = c("Bank_of_America",
                "Toronto_Dominion_Bank",
                "Citizens_Bank",
                "Webster Bank",
                "CHASE Bank",
                "Citigroup",
                "Capital One",
                "HSBC Bank USA",
                "State Street Corporation",
                "MUFG Union Bank",
                "Citizens Bank",
                "Barclays",
                "New York Community Bank",
                "CIT Group",
                "Santander Bank",
                "Royal Bank of Scotland",
                "First Rand Bank",
                "Budapest Bank"
               )

bank_draw <- gl(n = length(list_banks), k = 3, labels = list_banks)


#######################COLUMN NAMES#######################
####INDIVIDUAL COLUMNS THAT COULD PROVIDE MORE INFO BUT ARE NOT PART OF A ROW THAT Q2 TRANSACTION WOULD GENERATE
Age <- sample(x = 18:35, size = observation_size, replace = TRUE)
Student <- as.numeric(sample(x = 0:1, size = observation_size, replace = TRUE))
account_balance <- sample(x=c(u100, o100u1000, o1000u10000, o10000), size = observation_size, replace = TRUE)
CS_FICO_str <- round(sample(x=c(Exceptional_850_800, Very_Good_799_740, Average_701, Good_739_670, Fair_669_580, Very_Poor_579_300), size = observation_size, replace = TRUE), digits = 0) #STR
CS_FICO_num <- round(rnorm(n = observation_size, mean = 701, sd = 100), digits = 0)
CS <- sample(x=c(Average, Excellent, Poor), size = observation_size, replace = TRUE)

##Q2 format columns# LEAVE SPACING AND SPELLING TO BE IDENTICAL WITH THE Q2 OBJECT!!!

##number of transactions that meet query criteria; returns a single page per call
transactionCount <- as.numeric(seq_len(observation_size)) #NUM int32
##unique identifier created by CorePro for each transaction
transactionId <- abs(as.numeric(sample(x = .Random.seed * 0.5, size = observation_size, replace = TRUE))) #NUM int64
##unique transaction identifier created by CorePro to group transactions together (Debit card auth + completions; ACH withdrawal + subsequent return)
masterId <- as.numeric(sample(x = .Random.seed ^ 2, size = observation_size, replace = TRUE)) #NUM int64
##customer who is in possession of the bank account
customerId <- abs(as.numeric(sample(x = .Random.seed, size = observation_size, replace = TRUE))) #NUM
##human-readable description of the type of a transaction (CorePro Withdrawal, CorePro Deposit) SEE TRANSACTION TYPES
type <- sample(x = transaction_draw, size = observation_size, replace = TRUE)
##programmatic code to indicate the type of the transaction
typeCode <- abs(as.numeric(sample(x = .Random.seed, size = observation_size, replace = TRUE)))#NUM/STR
##program-wide unique identifier provided by the caller at transfer/create time; not from CorePro
tag <- abs(as.numeric(sample(x = .Random.seed, size = observation_size, replace = TRUE)))#NUM
##human-readable description about the transactions; automatically generated; driven by "typeCode" of the transaction
friendlyDescription <- abs(as.numeric(sample(x = .Random.seed, size = observation_size, replace = TRUE)))#STR
##client-specified description; contains NACHA info; ATM name/location (ISO-8583 interface); BY LAW, MUST BE SHOWN TO THE CLIENT
description <- as.numeric(sample(x = .Random.seed ^ 2, size = observation_size, replace = TRUE))#STR
##indicates status of the transaction; 
status <- sample(x = c(Initiated, Pending, Settled, Voided), size = observation_size, replace = TRUE)#STR
##exact date and time the transaction was created; returned in time zone of the bank
createdDate <- seq(from = start_date, to = end_date, length.out = observation_size) #DATE
##amount in USD
amount <- rnorm(n = observation_size, mean = 4500, sd = 480) #USD
##TRUE if the amount is credited to the accountId, FALSE if the amount is debited
isCredit <- sample(x = c("Y", "N"), size = observation_size, replace = TRUE)#BOOL; crediting or debiting
##date and time at which the transaction was settled; returned in time zone of the bank
settledDate <- seq(from = start_date, to = end_date, length.out = observation_size) #DATE
##date and time at which funds associated with the account become available; returned in time zone of the bank
availableDate <- seq(from = start_date, to = end_date, length.out = observation_size) #DATE
##date and time at which the transaction was voided; returned in time zone of the bank
voidedDate <- seq(from = start_date, to = end_date, length.out = observation_size) #DATE
##gives a reason why the transaction was returned; 
returnCode <- sample(x = c(000, 111, 222,333), size = observation_size, replace = TRUE) #NUM
##fee code a transaction
feeCode <- sample(x = c(RGD, RTN, NSF), size = observation_size, replace = TRUE) #NUM
##description of the fee
feeDescription <- sample(x = gl(n = 2, k = observation_size, labels = c("Gallia est omnis divisa in partes tres", "Abeus Papam")), size = observation_size, replace = TRUE) #STR
##CorePro-generated identifier for the card that created the transaction; empty or 0 if not tied to a card
cardId <- abs(sample(x = .Random.seed, size = observation_size, replace = TRUE))
##value are TBD; only referring to card-based transactions
subTypeCode <- sample(x = yn_draw, size = observation_size, replace = TRUE)#NUM only card-based transactions
##description of the subtype of the transaction; only card-based transactions
subType <- sample(x = yn_draw, size = observation_size, replace = TRUE)#NUM; only card-based transactions
##name of the institution that originated the transaction; not always available
institutionName <- sample(x = bank_draw, size = observation_size, replace = TRUE)#STR
##related to check deposit transaction types
check <- sample(x = gl(n = 2, k = 1, labels = c("Y","N")), size = observation_size, replace = TRUE) #NUM; YES = 1 and NO = 0



##########DATA FRAME#################
df <- data.frame(transactionCount, 
                  transactionId, 
                  masterId, 
                  customerId,
                  type,
                  typeCode,
                  tag,
                  friendlyDescription,
                  description,
                  status,
                  createdDate,
                  amount,
                  isCredit,
                  settledDate,
                  availableDate,
                  voidedDate,
                  returnCode,
                  feeCode, 
                  feeDescription,
                  cardId,
                  subTypeCode,
                  subType,
                  institutionName,
                  check,
                  Student,
                  account_balance,
                  Age,
                  CS,
                  CS_FICO_num,
                  CS_FICO_str)
                  
print(df)
if (is.data.frame(df) == TRUE) {
  write.csv(x = df, file = "Q_test_data.csv", append = FALSE)
}