# Support Vector Machine tuning ----

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
svm_model <- svm_poly(mode = "classification",
                      cost = tune(), degree = tune(), scale_factor = tune())

# set-up tuning grid ----
svm_params <- hardhat::extract_parameter_set_dials(svm_model) %>%
  update(cost = cost(),
         degree = degree(),
         scale_factor = scale_factor()
  )

# define tuning grid
svm_grid <- grid_regular(svm_params, levels = 5)

# workflow ----
svm_wflow <-
  workflow() %>%
  add_model(svm_model) %>%
  add_recipe(fire_recipe)

# Tuning/fitting ----
tic.clearlog()
tic("Support Vector Machine")
svm_tune <- 
  svm_wflow %>%
  tune_grid(
    resamples = fire_fold,
    grid = svm_grid,
    control = keep_pred,
    metrics = wf_metrics
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

svm_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  run_time = end_time - start_time
)

# Write out results & workflow
save(svm_tune, svm_tictoc, file = "results/svm_tune.rda")