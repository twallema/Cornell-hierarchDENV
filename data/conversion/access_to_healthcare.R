library(datazoom.amazonia)
library(tidyr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(TTR)


# Step 1: Download treated data with the number of available beds
data <- load_datasus(
  dataset = "datasus_cnes_lt",
  time_period = 2005:2023,
  raw_data = FALSE
)

# Step 2: Summarise the number of beds per Brazilian UF and state
beds_summary <- data %>%
  group_by(year, month, abbrev_state) %>%
  summarise(
    total_beds = sum(n_existing_beds, na.rm = TRUE),
    total_beds_sus = sum(n_beds_sus, na.rm = TRUE),
    .groups = 'drop'
  )

# Step 3: Replace YEAR + MONTH by end of month
beds_summary <- beds_summary %>%
  mutate(
    date = ceiling_date(make_date(year, month, 1), "month") - days(1)
  ) %>%
  select(UF = abbrev_state, date, total_beds, total_beds_sus)

# Step 4: Expand dataset to 1996-2025 by padding
# Step 4a: Generate monthly date sequence
full_dates <- seq(ymd("1996-01-01"), ymd("2025-12-01"), by = "1 month") %>%
  ceiling_date(unit = "month") - days(1)
# Step 4b: Get unique UF list
all_states <- na.omit(unique(beds_summary$UF))
# Step 4c: Create all combinations of UF and date
full_grid <- expand.grid(UF = all_states, date = full_dates)
# Step 4d: Convert both UF columns to character to avoid join errors
full_grid$UF <- as.character(full_grid$UF)
beds_summary$UF <- as.character(beds_summary$UF)
# Step 4e: Merge with original data and fill missing values for both total and SUS beds
padded_data <- full_grid %>%
  left_join(beds_summary, by = c("UF", "date")) %>%
  arrange(UF, date) %>%
  group_by(UF) %>%
  fill(total_beds, total_beds_sus, .direction = "downup")


# Step 5: Get demographics per UF per year
# Step 5a: Load demographic data
demographics <- read.csv('../raw/sprint_2025/datasus_population_2001_2024.csv')
# Step 5b: Load region mapping
region_map <- read.csv('../raw/sprint_2025/map_regional_health.csv')
# Step 5c: Extract the unique geocode and UF pairs
geocode_uf_map <- unique(region_map[, c("geocode", "uf")])
# Step 5d: Merge demographic data with UF info
demographics <- merge(demographics, geocode_uf_map, by = "geocode")
# Step 5e: Convert demographics to per UF per year
demographics <- demographics %>%
  group_by(uf, year) %>%
  summarise(population = sum(population, na.rm = TRUE), .groups = "drop")

# Step 6: Normalise the number of beds with the population size
# Step 6a: Extract year from the date colum
padded_data$date <- as.Date(padded_data$date)
padded_data$year <- as.numeric(format(padded_data$date, "%Y"))
# Step 6b: Append demograhpics to beds dataset
beds_with_pop <- padded_data %>%
  left_join(demographics, by = c("UF" = "uf", "year"))
# Step 6d: Pad pre 2005
beds_with_pop <- beds_with_pop %>%
  group_by(UF) %>%
  mutate(
    population = ifelse(year < 2005,
                        population[year == 2005][1],  # use 2005 population for that UF
                        population)
  ) %>%
  ungroup()
# Step 6e: Pad post-2024
beds_with_pop <- beds_with_pop %>%
  group_by(UF) %>%
  mutate(
    population = ifelse(year > 2024,
                        population[year == 2024][1],  # use 2024 population for that UF
                        population)
  ) %>%
  ungroup()
# Step 6f: Summarise results
final_data <- beds_with_pop %>%
  select(-year) %>%
  mutate(
    beds_per_1000 = total_beds / population * 1000,
    beds_sus_per_1000 = total_beds_sus / population * 1000,
    fraction_sus_beds = total_beds_sus / total_beds
  )


# Step 7: Smooth the data
n_ema <- 36
final_data <- final_data %>%
  group_by(UF) %>%
  arrange(date) %>%
  mutate(
    total_beds = EMA(total_beds, n = n_ema),
    total_beds_sus = EMA(total_beds_sus, n = n_ema),
    beds_per_1000 = EMA(beds_per_1000, n = n_ema),
    beds_sus_per_1000 = EMA(beds_sus_per_1000, n = n_ema)
  )


# Step 8: Save results
write.csv(x = final_data, file = "../interim/BR_hospital-beds-per-capita_2005-2023.csv")

# Step 9: Visualisation
selected_state <- "CE"
# Filter and plot
final_data %>%
  filter(UF == selected_state) %>%
  ggplot(aes(x = date, y = beds_per_1000)) +
  geom_line(color = "steelblue", size = 1) +
  labs(
    title = paste("Hospital Beds Over Time in", selected_state),
    x = "Date",
    y = "Number of Existing Hospital Beds"
  ) +
  theme_minimal()


