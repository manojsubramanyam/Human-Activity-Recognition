rm(list = ls())

# required packages
require(data.table)
require(stringi)
require(dplyr)

# datasets 
features = fread('UCI HAR Dataset/features.txt')
y_classes = fread('UCI HAR Dataset/activity_labels.txt')
trainX = fread('UCI HAR Dataset/train/X_train.txt', col.names = features$V2)
trainY = fread('UCI HAR Dataset/train/y_train.txt', col.names = 'Activity')
testX = fread('UCI HAR Dataset/test/X_test.txt', col.names = features$V2)
testY = fread('UCI HAR Dataset/test/y_test.txt', col.names = 'Activity')
# train = cbind(trainY, trainX)
# test = cbind(testY, testX)
# rm(list = setdiff(ls(), c('train','test')))

# setDF(train)
# setDF(test)
# Feature engneering 

names(trainX)
# we can see all the variables given here are various statistical parameter of the calibration reading
# we have mean, std, mad, max, min, sma, energy, iqr, entropy and arCoeff and correlation w.r.t other planes

# Let's see different signals that were calibrated

u_signals = data.frame()
for(x in names(trainX)){
  flag = na.omit(unlist(stri_match_all(str = x, regex = c('X,Y','Y,Z','X,Z','X','Y','Z'))))[1]
  u_signals = rbind(u_signals, data.frame(strsplit(x, '-')[[1]][1], if_else(is.na(flag),'',
                                                                            flag)))
}
u_signals = cbind(names(trainX), u_signals)
names(u_signals) = c('variables', 'signal','Plane')

for(i in names(u_signals)){
  u_signals[,i] = as.character(u_signals[,i])
}

# from the above activity we can see that variables are grouped by various calibrations of X,Y,Z and their pair-wise combinations too

# we're gonna analyze each subset
signals = unique(u_signals$signal) # 24 signals 
# So there are 17 calibrated signals which are analyzed on X,Y and Z planes and different variables are created
# There are 7 more variables derived on angles between those certain calibrated signals

# Let's consider tBodyAcc and X plane variables 

tBodyAcc_X_var = u_signals$variables[u_signals$signal == 'tBodyAcc' & u_signals$Plane =='X'] # 12 variables

tBodyAcc_X = as.data.frame(trainX)[tBodyAcc_X_var]

# Let's check correlations 
ggcorrplot::ggcorrplot(cor(tBodyAcc_X), lab = T, type = 'upper')
# we can observe serious multicollinearity problem within this small subset
# By analyzing this, we can also infer a similar pattern in other planes also...

## APPROACH ##
# We're gonna do principal component analysis on each subset and bind them and build the model now

# Let's build a loop to do this...helpful for test data set creation

# Different combinations of signals and planes 
u_combination = u_signals %>% group_by(signal, Plane) %>% summarise(column_count = n())
# we will consider the combinations that are having more than 5 variables in it
valid_combinations = u_combination[u_combination$column_count>5,]

data_train = as.data.frame(trainX)
data_test = as.data.frame(testX)

pca_train = data.frame(character(nrow(trainX)))[-1]
pca_test =  data.frame(character(nrow(testX)))[-1]


for(i in seq(nrow(valid_combinations))){
  
  train_subset = data_train[u_signals$variables[u_signals$signal == valid_combinations$signal[i] & u_signals$Plane == valid_combinations$Plane[i]]]
  test_subset = data_test[u_signals$variables[u_signals$signal == valid_combinations$signal[i] & u_signals$Plane == valid_combinations$Plane[i]]]
  
  pca = princomp(train_subset)
  var = (pca$sdev)^2
  cum_var = cumsum(var/sum(var))
 
  cum_var_name = cum_var[cum_var <.9]
  
  tmp_train = as.data.frame(pca$scores[,names(cum_var_name)])
  
  names(tmp_train) = paste0(valid_combinations$signal[i],'_',valid_combinations$Plane[i] , seq(cum_var_name))
  
  pca_train = cbind(pca_train, tmp_train)
  
  # By using PCA model built on train, we are predicting PCs (principal components) for test also
  tmp_test = as.data.frame(predict(pca, test_subset))[,seq(cum_var_name)]
  pca_test = cbind(pca_test, tmp_test)
  
}

names(pca_test) = names(pca_train)
# Let's check the correlations of this new dataset
ggcorrplot::ggcorrplot(cor(pca_train), type = 'upper')
# There is still so many principal components  being correlated other signals'

## We now have two ways doing things

# 1. Building a PCA on this data set and take top PCs with variance - Dimensionality reduction (one more time)
# 2. Removing highly correlated variable from the dataset and model it - feature selection

# 1.
pca_new = princomp(pca_train)

cum_var = cumsum((pca_new$sdev)^2/sum((pca_new$sdev)^2))

cum_var_name = cum_var[cum_var <.9]

train_final = as.data.frame(pca_new$scores[,names(cum_var_name)])
# for test
test_final = as.data.frame(predict(pca_new, pca_test))[,names(cum_var_name)]

## MODELING ##
# We use h2o to build models 
require(h2o)

# initializing 
h2o.init(nthreads = -1)

# converting into H2oFrames
train.h = as.h2o(cbind(train_final, Activity = as.factor(trainY$Activity)))
test.h = as.h2o(cbind(test_final, Activity = as.factor(testY$Activity)))

y = grep('Activity', names(train.h))
x = setdiff(seq(names(train.h)), y)

# random forest model
model_rf = h2o.randomForest(x,y,training_frame = train.h)

# predicting on test data set
pred = as.data.frame(h2o.predict(model_rf, test.h))

caret::confusionMatrix(pred$predict, as.factor(testY$Activity))  # 0.7855 accuracy


# 2. 
## Let's do feature selection on pca_test and pca_train
cr = cor(pca_train)
cr[!lower.tri(cr)] = 0

var_cor_no <- names(pca_train)[!apply(cr,2,function(x) any(x > 0.7))]


# converting into H2oFrames
train.h = as.h2o(cbind(pca_train[var_cor_no], Activity = as.factor(trainY$Activity)))
test.h = as.h2o(cbind(pca_test[var_cor_no], Activity = as.factor(testY$Activity)))

y = grep('Activity', names(train.h))
x = setdiff(seq(names(train.h)), y)

# random forest model
model_rf = h2o.randomForest(x,y,training_frame = train.h)

# predicting on test data set
pred = as.data.frame(h2o.predict(model_rf, test.h))

caret::confusionMatrix(pred$predict, as.factor(testY$Activity))  # 0.7855 accuracy
