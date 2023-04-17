# Elastic Net tuning ----

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

# Update base recipe ----
en_recipe <- fire_recipe %>%
  step_interact(wlf ~ all_numeric_predictors()^2)

# Define model ----
en_model <- 
  logistic_reg(
    mode = "classification",
    penalty = tune(),
    mixture = tune()) %>%
  set_engine("glmnet")
# set-up tuning grid ----
en_params <- hardhat::extract_parameter_set_dials(en_model) %>%
  update(penalty = penalty(), 
         mixture = mixture()
         )

# define tuning grid
en_grid <- grid_regular(en_params, levels = 5)

# workflow ----
en_wflow <-
  workflow() %>%
  add_model(en_model) %>%
  add_recipe(en_recipe)

# Tuning/fitting ----
tic.clearlog()
tic("Elastic Net")
en_tune <- 
  en_wflow %>%
  tune_grid(
  resamples = fire_fold,
  grid = en_grid,
  control = keep_pred,
  metrics = wf_metrics
)
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

en_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  run_time = end_time - start_time
)

# Write out results & workflow
save(en_tune, en_tictoc, file = "results/en_tune.rda")
