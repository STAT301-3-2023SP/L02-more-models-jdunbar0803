# Multivariate adaptive regression splines tuning ----

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
mars_model <- mars(mode = "classification", 
                   engine = "earth",
                   num_terms = tune(),
                   prod_degree = tune())

# set-up tuning grid ----
mars_params <- hardhat::extract_parameter_set_dials(mars_model) %>%
  update(num_terms = num_terms(c(1,5)),
         prod_degree = prod_degree()
  )

# define tuning grid
mars_grid <- grid_regular(mars_params, levels = 5)

# workflow ----
mars_wflow <-
  workflow() %>%
  add_model(mars_model) %>%
  add_recipe(fire_recipe)

# Tuning/fitting ----
tic.clearlog()
tic("MARS")
mars_tune <- 
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

mars_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  run_time = end_time - start_time
)

# Write out results & workflow
save(mars_tune, mars_tictoc, file = "results/mars_tune.rda")