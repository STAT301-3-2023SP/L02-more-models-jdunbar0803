# Random Forest tuning ----

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
rf_model <- rand_forest(mode = "classification",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")
# set-up tuning grid ----
rf_params <- hardhat::extract_parameter_set_dials(rf_model) %>%
  update(mtry = mtry(c(1,15)))

# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_wflow <-
  workflow() %>%
  add_model(rf_model) %>%
  add_recipe(fire_recipe)

# Tuning/fitting ----
tic.clearlog()
tic("Random Forest")
rf_tune <- 
  rf_wflow %>%
  tune_grid(
    resamples = fire_fold,
    grid = rf_grid,
    control = keep_pred,
    metrics = wf_metrics
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

rf_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  run_time = end_time - start_time
)

# Write out results & workflow
save(rf_tune, rf_tictoc, file = "results/rf_tune.rda")
