# Load package(s)
library(tidymodels)
library(tidyverse)
library(earth)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

## load data
wildfires_dat <- read_csv("data/wildfires.csv") %>%
  janitor::clean_names() %>%
  mutate(
    winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
    traffic = factor(traffic, levels = c("lo", "med", "hi")),
    wlf = factor(wlf, levels = c(1, 0), labels = c("yes", "no"))) %>%
    select(-burned)

# split data
fire_split <- initial_split(wildfires_dat, prop = .8, strata = wlf)
fire_train <- training(fire_split)
fire_test  <- testing(fire_split)

# fold data
fire_fold <- vfold_cv(fire_train, v = 5, repeats = 3)

# recipe
fire_recipe <- recipe(wlf ~ ., data = fire_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  prep() 
bake(fire_recipe, new_data = NULL)

#set options for tuning process
keep_pred <- control_grid(save_pred = TRUE, save_workflow = TRUE)

# Metric set
wf_metrics <- metric_set(accuracy, roc_auc, precision, recall, sensitivity, specificity, f_meas)

#Save initial setup
save(fire_split, file = "initial_setup/fire_split.rda")

save(keep_pred, fire_fold, fire_recipe, "initial_setup/tuning_setup.rda")

#set up models
elastic_model <- 
  linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

knn_model <- nearest_neighbor(mode = "classification", neighbors = tune()) %>% 
  set_engine("kknn")

rf_model <- rand_forest(mode = "classification",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")

xgboost_model <- 
  boost_tree(trees = tune(), min_n = tune(), learn_rate = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("xgboost")

svm_model <- svm_poly(mode = "classification",
                      cost = tune(), degree = tune(), scale_factor = tune())

svmrbf_model <- svm_rbf(mode = "classification",
                        cost = tune(), rbf_sigma = tune())

mlp_model <- mlp(mode = "classification", 
                 engine = "nnet",
                 hidden_units = tune(),
                 penalty = tune())
  
mars_model <- mars(mode = "classification", 
                   engine = "earth",
                   num_terms = tune(),
                   prod_degree = tune())

elastic_params <- extract_parameter_set_dials(elastic_model)
knn_params <- extract_parameter_set_dials(knn_model)
rf_params <- extract_parameter_set_dials(rf_model) %>%
  update(mtry = mtry(c(1, 15)))
xgboost_params <- extract_parameter_set_dials(xgboost_model)
svm_params <- extract_parameter_set_dials(svm_model)
svmrbf_params <- extract_parameter_set_dials(svmrbf_model)
mlp_params <- extract_parameter_set_dials(mlp_model)
mars_params <- extract_parameter_set_dials(mars_model)

elastic_grid <- grid_regular(elastic_params, levels = 5)
knn_grid <- grid_regular(knn_params, levels = 5)
rf_grid <- grid_regular(rf_params, levels = 5) 
xgboost_grid <- grid_regular(xgboost_params, levels = 5)
svm_grid <- grid_regular(svm_params, levels = 5)
svmrbf_grid <- grid_regular(svmrbf_params, levels = 5)
mlp_grid <- grid_regular(mlp_params, levels = 5)
mars_grid <- grid_regular(mars_params, levels = 5)

elastic_workflow <- workflow() %>% 
  add_model(elastic_model) %>% 
  add_recipe(fire_recipe)
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(fire_recipe)
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(fire_recipe)
xgboost_workflow <- workflow() %>% 
  add_model(xgboost_model) %>% 
  add_recipe(fire_recipe)
svm_workflow <- workflow() %>% 
  add_model(svm_model) %>% 
  add_recipe(fire_recipe)
svmrbf_workflow <- workflow() %>% 
  add_model(svmrbf_model) %>% 
  add_recipe(fire_recipe)
mlp_workflow <- workflow() %>% 
  add_model(mlp_model) %>% 
  add_recipe(fire_recipe)
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(fire_recipe)

elastic_tuned <- elastic_workflow %>% 
  tune_grid(fire_fold, grid = elastic_grid)
knn_tuned <- knn_workflow %>% 
  tune_grid(fire_fold, grid = knn_grid)
rf_tuned <- rf_workflow %>% 
  tune_grid(fire_fold, grid = rf_grid)
xgboost_tuned <- xgboost_workflow %>% 
  tune_grid(fire_fold, grid = xgboost_grid)
svm_tuned <- svm_workflow %>% 
  tune_grid(fire_fold, grid = svm_grid)
svmrbf_tuned <- svmrbf_workflow %>% 
  tune_grid(fire_fold, grid = svmrbf_grid)
mlp_tuned <- mlp_workflow %>% 
  tune_grid(fire_fold, grid = mlp_grid)
mars_tuned <- mars_workflow %>% 
  tune_grid(fire_fold, grid = mars_grid)