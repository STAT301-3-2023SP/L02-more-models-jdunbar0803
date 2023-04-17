# Multivariate adaptive regression splines tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)
library(kernlab)

# handle common conflicts
tidymodels_prefer()

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

# load required objects ----
load("initial_setup/tuning_setup.rda")
load("initial_setup/wf_metrics.rda")
load("results/en_tune.rda")
load("results/rf_tune.rda")
load("results/knn_tune.rda")
load("results/xgboost_tune.rda")
load("results/svm_tune.rda")
load("results/svmrbf_tune.rda")
load("results/mlp_tune.rda")
load("results/mars_tune.rda")

show_best(en_tune, metric = "accuracy")
show_best(knn_tune, metric = "accuracy")
show_best(rf_tune, metric = "accuracy")
show_best(xgboost_tune, metric = "accuracy")
show_best(svm_tune, metric = "accuracy")
show_best(svmrbf_tune, metric = "accuracy")
show_best(mlp_tune, metric = "accuracy")
show_best(mars_tune, metric = "accuracy")

mlp_wflow_tuned <- mlp_wflow %>% 
  finalize_workflow(select_best(mlp_tune, metric = "accuracy"))

mlp_results <- fit(mlp_wflow_tuned, fire_train)


