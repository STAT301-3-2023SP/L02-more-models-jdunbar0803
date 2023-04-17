# Knn tuning ----

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
knn_model <- 
  nearest_neighbor(mode = "classification", neighbors = tune()) %>% 
  set_engine("kknn")
# set-up tuning grid ----
knn_params <- hardhat::extract_parameter_set_dials(knn_model) %>%
  update(neighbors = neighbors(), 
  )

# define tuning grid
knn_grid <- grid_regular(knn_params, levels = 5)

# workflow ----
knn_wflow <-
  workflow() %>%
  add_model(knn_model) %>%
  add_recipe(fire_recipe)

# Tuning/fitting ----
tic.clearlog()
tic("Nearest Neighbors")
knn_tune <- 
  knn_wflow %>%
  tune_grid(
    resamples = fire_fold,
    grid = knn_grid,
    control = keep_pred,
    metrics = wf_metrics
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

knn_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  run_time = end_time - start_time
)

# Write out results & workflow
save(knn_tune, knn_tictoc, file = "results/knn_tune.rda")
