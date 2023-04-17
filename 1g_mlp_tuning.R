# Single Layer Neural Network tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)
library(kernlab)

# handle common conflicts
tidymodels_prefer()

# set up parallel processing
parallel::detectCores()
registerDoMC(cores = 8)

# load required objects ----
load("initial_setup/tuning_setup.rda")
load("initial_setup/wf_metrics.rda")

# Define model ----
mlp_model <- mlp(mode = "classification", 
                 engine = "nnet",
                 hidden_units = tune(),
                 penalty = tune())

# set-up tuning grid ----
mlp_params <- hardhat::extract_parameter_set_dials(mlp_model) %>%
  update(penalty = penalty(),
         hidden_units = hidden_units()
  )

# define tuning grid
mlp_grid <- grid_regular(mlp_params, levels = 5)

# workflow ----
mlp_wflow <-
  workflow() %>%
  add_model(mlp_model) %>%
  add_recipe(fire_recipe)

# Tuning/fitting ----
tic.clearlog()
tic("Single Layer Neural Network")
mlp_tune <- 
  mlp_wflow %>%
  tune_grid(
    resamples = fire_fold,
    grid = mlp_grid,
    control = keep_pred,
    metrics = wf_metrics
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

mlp_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  run_time = end_time - start_time
)

# Write out results & workflow
save(mlp_tune, mlp_tictoc, file = "results/mlp_tune.rda")