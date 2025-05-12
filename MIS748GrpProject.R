# COCA-COLA Project: Time Series Analysis

# Load required libraries
install.packages(c("ggplot2", "forecast", "tseries", "TSA", "zoo", "readr"))
library(ggplot2)
library(forecast)
library(tseries)
library(TSA)
library(zoo)
library(readr)
library(dplyr)
library(fBasics)
library(fUnitRoots)
library(lmtest)
library(astsa)

getwd()


df = read_csv("C:/Users/asonawane8445/OneDrive - San Diego State University (SDSU.EDU)/Documents/coca_cola_final.csv", 
              col_types = cols(Date = col_date(format = "%m/%d/%Y")))

print(head(df))
print(summary(df))
sd(df$Close)
var(df$Close)


################################################################### Step 1: Create Time Series Object
ts_data = ts(df$Close, frequency = 12, start = c(1990, 1))
ggplot(df, aes(x = Date, y = Close)) +
  geom_line(color = "black") +
  labs(title = "Time Series Plot", x = "Date", y = "Close")

################################################################### Step 2: Log Transformation
log_ts_data = log(ts_data)
plot(log_ts_data, main = "Logarithmic Time Series", ylab = "Log Values")

par(mfrow = c(1, 2))
acf(log_ts_data, main = "ACF of Log Data")
pacf(log_ts_data, main = "PACF of Log Data")

adf.test(log_ts_data)

################################################################## Step 3: Differencing and Plots
par(mfrow = c(1, 1))
diff_log_ts_data = diff(log_ts_data)
plot(diff_log_ts_data, main = "Differenced Logarithmic Time Series", ylab = "Log Values")

par(mfrow = c(1, 2))
acf(diff_log_ts_data, main = "ACF of Differenced Log Data", lag.max = 30)
pacf(diff_log_ts_data, main = "PACF of Differenced Log Data", lag.max = 30)

adf.test(diff_log_ts_data)

eacf(diff_log_ts_data)

################################################################# Step 4: Model Selection
first_model = arima(diff_log_ts_data, order = c(0, 1, 8))

second_model = arima(diff_log_ts_data, order = c(1, 1, 8))

third_model = arima(diff_log_ts_data, order = c(5, 1, 5))

#summary(first_model)
#summary(second_model)
#summary(third_model)

coeftest(first_model)
coeftest(second_model)
coeftest(third_model)

################################################################# Step 5: Selecting best model
updated_first_model = arima(diff_log_ts_data, order = c(0, 1, 8), fixed = c(NA, 0, 0, 0, 0, 0, 0, 0))

updated_second_model = arima(diff_log_ts_data, order = c(1, 1, 8), fixed = c(NA, NA, NA, NA, 0, 0, 0, 0, NA))

updated_third_model = arima(diff_log_ts_data, order = c(5, 1, 5), fixed = c(NA, NA, NA, NA, 0, NA, 0, 0, 0, NA))

#summary(updated_first_model)
#summary(updated_second_model)
#summary(updated_third_model)

coeftest(updated_first_model)
coeftest(updated_second_model)
coeftest(updated_third_model)

first_model$aic
updated_first_model$aic

BIC(first_model)
BIC(updated_first_model)
############################################
second_model$aic
updated_second_model$aic

BIC(second_model)
BIC(updated_second_model)
#############################################
third_model$aic
updated_third_model$aic

BIC(third_model)
BIC(updated_third_model)

##Based on the AIC results, we will proceed with updated_first_model ARIMA(0,1,1), 
##second_model ARIMA(1,1,8) and third_model ARIMA(5,1,5)

################################################################### Step 6: Tests for model diagnostics
fm_residuals = updated_first_model$residuals
plot(fm_residuals, main = "Residuals of ARIMA(0,1,1)", xlab = "Time", ylab = "Residuals")

sm_residuals = second_model$residuals
plot(sm_residuals, main = "Residuals of ARIMA(1,1,8)", xlab = "Time", ylab = "Residuals")

tm_residuals = third_model$residuals
plot(tm_residuals, main = "Residuals of ARIMA(5,1,5)", xlab = "Time", ylab = "Residuals")


acf(fm_residuals, main = "ACF of residuals for ARIMA(0,1,1)")
pacf(fm_residuals, main = "PACF of residuals for ARIMA(0,1,1)")
eacf(fm_residuals)
polyroot(c(1, fm_residuals[1]))
abs(polyroot(c(1, fm_residuals[1])))


acf(sm_residuals, main = "ACF of residuals for ARIMA(1,1,8)")
pacf(sm_residuals, main = "PACF of residuals for ARIMA(1,1,8)")
eacf(sm_residuals)
polyroot(c(1, -sm_residuals[1]))
abs(polyroot(c(1, -sm_residuals[1])))

polyroot(c(1, sm_residuals[2:9]))
abs(polyroot(c(1, sm_residuals[2:9])))


acf(tm_residuals, main = "ACF of residuals for ARIMA(5,1,5)")
pacf(tm_residuals, main = "PACF of residuals for ARIMA(5,1,5)")
eacf(tm_residuals)
polyroot(c(1, -tm_residuals[1:5]))
abs(polyroot(c(1, -tm_residuals[1:5])))

polyroot(c(1, tm_residuals[6:10]))
abs(polyroot(c(1, tm_residuals[6:10])))

Box.test(fm_residuals, lag = 12, type = "Ljung-Box")

Box.test(sm_residuals, lag = 12, type = "Ljung-Box")

Box.test(tm_residuals, lag = 12, type = "Ljung-Box")



##################################################################### Step 7: Rolling forecast
source("rolling.forecast.R")
print(rolling.forecast(diff_log_ts_data, 24, length(diff_log_ts_data)-40, order = c(5, 1, 5)))


##################################################################### Step 8: Future Prediction

# Fit the ARIMA model to the differenced log-transformed data
pp <- predict(third_model, n.ahead = 12)  # Predict for the next 12 months

# Extract the last observed value from the log-transformed time series
last_log_value <- as.numeric(tail(log_ts_data, 1))

# Re-integrate predictions to the log scale by cumulative summation
log_predictions <- cumsum(pp$pred) + last_log_value

# Back-transform predictions to the original scale
pred <- ts(exp(log_predictions), start = c(2022, 1), frequency = 12)
pred.upp <- ts(exp(log_predictions + 2 * pp$se), start = c(2022, 1), frequency = 12)
pred.low <- ts(exp(log_predictions - 2 * pp$se), start = c(2022, 1), frequency = 12)

# Extract historical data for plotting
# Adjust the historical data segment
nb <- 36# Number of months to display (past 12 months)
nn = length(log_ts_data)
tt <- (nn - nb + 1):nn  # Index for the last 12 months
start_year <- 2019  # Adjust based on your data
xxx <- ts(ts_data[tt], start = c(start_year, 1), frequency = 12)

# Print to verify
print(xxx)

# Plot predictions with 95% prediction intervals
rr <- range(c(xxx, pred, pred.upp, pred.low))
par(mfrow = c(1, 1))
plot(xxx, type = 'o', pch = 3, xlim = c(2019, 2023), ylim = rr, xlab = 'Time', ylab = 'Stock Price',
     main = '12-Month Stock Price Prediction')
points(pred, pch = 2, col = 'red')  # Predicted values
lines(pred.upp, lty = 2, col = 'red')  # Upper prediction interval
lines(pred.low, lty = 2, col = 'red')  # Lower prediction interval
legend("topleft", legend = c("Historical Data", "Predicted Values", "Prediction Interval"),
       col = c("black", "red", "red"), pch = c(3, 2, NA), lty = c(NA, NA, 2))


###########################################################################

