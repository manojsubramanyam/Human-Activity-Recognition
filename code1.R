rm(list = ls())

# required packages
require(data.table)

# loading data 
features = fread('UCI HAR Dataset/features.txt')
y_classes = fread('UCI HAR Dataset/activity_labels.txt')
trainX = fread('UCI HAR Dataset/train/X_train.txt', col.names = features$V2)
trainY = fread('UCI HAR Dataset/train/y_train.txt', col.names = 'Activity')
testX = fread('UCI HAR Dataset/test/X_test.txt', col.names = features$V2)
testY = fread('UCI HAR Dataset/test/y_test.txt', col.names = 'Activity')

# checking proportions
table(trainY$Activity)
table(testY$Activity)

# Data munging- changing label names
levels(as.factor(trainY$Activity))
levels(as.factor(testY$Activity))

# type conversions
trainY$Activity = as.factor(trainY$Activity)
testY$Activity = as.factor(testY$Activity)




levels(trainY$Activity) = y_classes$V2
levels(testY$Activity) = y_classes$V2

# train = cbind(trainX, trainY)
# test  = cbind(testX,  testY)
# rm(list = setdiff(ls(), c('train','test')))

# sampling - stratified
set.seed(123)
spl = caTools::sample.split(trainY$Activity, .7)

train = subset(trainX, spl == T)
test = subset(trainX, spl == F)

trainy = trainY[spl == T]
testy = trainY[spl == F]

# Let's check correlations in the data
# checking correlation matrix
cr = cor(train)
cr[!lower.tri(cr)] = 0

# As there are many variables to check correlations with, it's difficult to visualize also.
# So we check the sum of instances for each variable in the correlation matrix that are crossing a certain threshold.

View(sapply(data.frame(cr), function(x) sum(x>0.7)))
sum(sapply(data.frame(cr), function(x) sum(x>0.7))>1) # 363

# This is a multi-(High)dimensional and multi-collinearity problem as there are 363 variables that being more than 70% correlated with other variables 

# We use one of the dimensionality reduction and extract few highly concentrated variables from it.

# Principal component analysis (PCA)
pca = princomp(train)

# Let's check the variance contributed by each component
var = (pca$sdev)^2
cum_var = cumsum(var/sum(var))
plot(cum_var)
# We're considering a threshold of 85 percent variance explaining variables from the model.
# By this, we are saying that these set of high concentrated variables can explain 85% of entire data's information.
cum_var_name = cum_var[cum_var <.9]

train_pca = as.data.frame(pca$scores[,names(cum_var_name)])

# By using PCA model built on train, we are predicting PCs (principal components) for test also
test_pca = as.data.frame(predict(pca, test))[,seq(cum_var_name)]


# We use h2o to build models 
require(h2o)

# initializing 
h2o.init(nthreads = -1)

# converting into H2oFrames
train.h = as.h2o(cbind(train_pca, Activity = trainy))
test.h = as.h2o(cbind(test_pca, Activity = testy))

y = grep('Activity', names(train.h))
x = setdiff(seq(names(train.h)), y)

# random forest model
model_rf = h2o.randomForest(x,y,training_frame = train.h)

# predicting on test data set
pred = as.data.frame(h2o.predict(model_rf, test.h))

caret::confusionMatrix(pred$predict, testy$Activity)

## The above is the basic model with .924 accuracy


# Let's validate this on real test data set- validation set
# We have to find the PCAs for the validation set from original pca model

val_pca = as.data.frame(predict(pca, testX))[,names(cum_var_name)]
val.h = as.h2o(cbind(val_pca, testY))

pred_val = as.data.frame(h2o.predict(model_rf, val.h))

caret::confusionMatrix(pred_val$predict, testY$Activity)
 # The model is giving .88 accuracy on validation set

## NEED to perform tuning

