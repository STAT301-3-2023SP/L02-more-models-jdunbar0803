# Radial Basis Function tuning ----

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
svmrbf_model <- svm_rbf(mode = "classification",
                        cost = tune(), rbf_sigma = tune())

# set-up tuning grid ----
svmrbf_params <- hardhat::extract_parameter_set_dials(svmrbf_model) %>%
  update(cost = cost(),
         rbf_sigma = rbf_sigma()
  )

# define tuning grid
svmrbf_grid <- grid_regular(svmrbf_params, levels = 5)

# workflow ----
svmrbf_wflow <-
  workflow() %>%
  add_model(svmrbf_model) %>%
  add_recipe(fire_recipe)

# Tuning/fitting ----
tic.clearlog()
tic("Support Vector Machine (Radial Basis Function)")
svmrbf_tune <- 
  svmrbf_wflow %>%
  tune_grid(
    resamples = fire_fold,
    grid = svmrbf_grid,
    control = keep_pred,
    metrics = wf_metrics
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

svmrbf_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  run_time = end_time - start_time
)

# Write out results & workflow
save(svmrbf_tune, svmrbf_tictoc, file = "results/svmrbf_tune.rda")