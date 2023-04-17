# Boosted tree tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# set up parallel processing
parallel::detectCores()
registerDoMC(cores = 8)

# load required objects ----
load("initial_setup/tuning_setup.rda")
load("initial_setup/wf_metrics.rda")

# Define model ----
xgboost_model <-
boost_tree(trees = tune(), min_n = tune(), learn_rate = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("xgboost")
# set-up tuning grid ----
xgboost_params <- hardhat::extract_parameter_set_dials(xgboost_model) %>%
  update(learn_rate = learn_rate()
  )

# define tuning grid
xgboost_grid <- grid_regular(xgboost_params, levels = 5)

# workflow ----
xgboost_wflow <-
  workflow() %>%
  add_model(xgboost_model) %>%
  add_recipe(fire_recipe)

# Tuning/fitting ----
tic.clearlog()
tic("Boosted Tree")
xgboost_tune <- 
  xgboost_wflow %>%
  tune_grid(
    resamples = fire_fold,
    grid = xgboost_grid,
    control = keep_pred,
    metrics = wf_metrics
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

xgboost_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  run_time = end_time - start_time
)

# Write out results & workflow
save(xgboost_tune, xgboost_tictoc, file = "results/xgboost_tune.rda")