import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import periodogram

plt.rcParams["figure.figsize"] = (10, 4)

# LOAD DATA
df = pd.read_csv("churn_clean.csv")
rev = pd.to_numeric(df["Revenue"], errors="coerce").dropna().reset_index(drop=True)

# D1. Line Graph of time series
plt.figure()
plt.plot(rev, marker="o", linestyle="-")
plt.title("Daily Revenue")
plt.xlabel("Day")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

# D2. Check missing values
expected = np.arange(rev.index.min(), rev.index.max() + 1)
missing = np.setdiff1d(expected, rev.index.values)
if len(missing) == 0:
    print("No missing days: regular daily time series.")
else:
    print("Missing Day values found:", missing)

# D3. The stationarity of the time series
print("ADF Test on Original Series")
adf_full = adfuller(rev.dropna())
print("ADF statistic:", adf_full[0])
print("p-value:", adf_full[1])

if adf_full[1] < 0.05:
    print("Series appears stationary.")
else:
    print("Series appears NON-stationary.")

# First difference to remove trend
rev_diff = rev.diff().dropna()

plt.figure()
plt.plot(rev_diff, marker="o")
plt.title("Differenced Revenue")
plt.xlabel("Day")
plt.tight_layout()
plt.show()

print("ADF Test on First Difference")
adf_diff = adfuller(rev_diff.dropna())
print("ADF statistic:", adf_diff[0])
print("p-value:", adf_diff[1])

# D4. TRAIN/TEST SPLIT LAST 30 DAYS
forecast_horizon = 30
train = rev.iloc[:-forecast_horizon]
test = rev.iloc[-forecast_horizon:]

print("\nTrain size:", len(train), "| Test size:", len(test))
print("Train ends at Day", train.index.max())
print("Test starts at Day", test.index.min())

# D5. Save Cleaned Data
cleaned_df = pd.DataFrame({
    "Time": rev.index,
    "Revenue": rev.values
})

cleaned_df.to_csv("revenue_clean_timeseries.csv", index=False)

print("\nCleaned dataset saved as: revenue_clean_timeseries.csv")

# E1. AUTOCORRELATION
plt.figure()
plot_acf(rev_diff, lags=40)
plt.title("ACF of Differenced Revenue")
plt.tight_layout()
plt.show()

plt.figure()
plot_pacf(rev_diff, lags=40, method="ywm")
plt.title("PACF of Differenced Revenue")
plt.tight_layout()
plt.show()

# E1. SPECTRAL DENSITY
freqs, power = periodogram(rev, detrend="linear")

plt.figure()
plt.plot(freqs[1:], power[1:])
plt.title("Periodogram (Spectral Density) of Revenue")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.tight_layout()
plt.show()

# E1. SEASONAL DECOMPOSITION
decomp = seasonal_decompose(rev, model="additive", period=7)

fig = decomp.plot()
fig.set_size_inches(10, 8)
plt.suptitle("Seasonal Decomposition (Trend, Seasonal, Residual)")
plt.tight_layout()
plt.show()

# E1. Check residuals for lack of trend
resid = decomp.resid.dropna()

plt.figure()
plt.plot(resid)
plt.title("Decomposition Residuals")
plt.xlabel("Day")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

plt.figure()
plot_acf(resid, lags=40)
plt.title("ACF of Residuals")
plt.tight_layout()
plt.show()

# E2. ARIMA MODEL
order = (1, 1, 1)
model = ARIMA(train, order=order)
result = model.fit()

print("ARIMA(1,1,1) Model Summary")
print(result.summary())

# E3. FORECAST NEXT 30 DAYS
forecast_obj = result.get_forecast(steps=forecast_horizon)
forecast_mean = forecast_obj.predicted_mean
forecast_ci = forecast_obj.conf_int(alpha=0.05)
forecast_index = test.index

# E3. FORECAST PLOT VS TEST SET
plt.figure()
plt.plot(train.index, train.values, label="Train")
plt.plot(test.index, test.values, label="Test", color="black")
plt.plot(forecast_index, forecast_mean, label="Forecast", linestyle="--")

plt.fill_between(
    forecast_index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.2,
    label="95% Prediction Interval"
)

plt.title("30 Day Forecast- ARIMA(1,1,1)")
plt.xlabel("Day")
plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.show()

# E4. FORECAST ACCURACY METRICS
errors = forecast_mean.values-test.values
rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))
mape = np.mean(np.abs(errors/test.values))*100

print("Forecast Accuracy")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
