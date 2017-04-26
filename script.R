

library(h2o)
h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O


knitr::opts_chunk$set(echo = TRUE)



h2o.no_progress()  # Disable progress bars for Rmd


## Load and Check the Data



mnist <- read.csv("./train.csv")
mnist_ts <- read.csv("./test.csv")
test <- as.h2o(mnist_ts)
mnist$label <- factor(mnist$label)
mnisth2o <- as.h2o(mnist) # convert to h2o dataframe

# split data into training and validation
splits <- h2o.splitFrame(data = mnisth2o, 
                         ratios = c(0.8),  #partition data into 80% and 20%
                         seed =1)  #setting a seed will guarantee reproducibility
train <- splits[[1]]
valid <- splits[[2]]

dim(train)
dim(valid)


y <- "label"
x <- setdiff(names(train),y)



dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            validation_frame = valid,
                            model_id = "dl_fit1",
                            hidden = c(20,20),
                            seed = 1)

dl_perf1 <- h2o.performance(model = dl_fit1)
h2o.mse(dl_perf1)
h2o.confusionMatrix(dl_fit1)
plot(dl_fit1, 
     timestep = "epochs", 
     metric = "classification_error")



dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            validation_frame = valid,
                            model_id = "dl_fit2",
                            hidden = c(20,20),
                            epochs=20,
                            activation = "Rectifier",
                            seed = 1)

dl_perf2 <- h2o.performance(model = dl_fit2)
h2o.mse(dl_perf2,valid=T)
h2o.confusionMatrix(dl_fit2, valid=T)
plot(dl_fit2, 
     timestep = "epochs", 
     metric = "classification_error")


h2o.predict(dl_fit1,newdata = test)
h2o.predict(dl_fit2,newdata = test)



activation_opt <- c("Rectifier", "Maxout", "Tanh")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs=60, max_models=20)

dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = train,
                    validation_frame = valid,
                    seed = 1,
                    hidden = c(20,20),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)

dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "accuracy",
                           decreasing = TRUE)
print(dl_gridperf)


best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)

