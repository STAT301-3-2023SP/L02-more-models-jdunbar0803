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

save(keep_pred, fire_fold, fire_recipe, file = "initial_setup/tuning_setup.rda")

save(wf_metrics, file = "initial_setup/wf_metrics.rda")