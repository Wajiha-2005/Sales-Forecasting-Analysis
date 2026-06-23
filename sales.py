import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv("MonthlySales.csv")
df['Month_Number']=np.arange(len(df))
X=df[['Month_Number']]
y=df['Sales']
model=LinearRegression()
model.fit(X,y)
historical_pred=model.predict(X)
future_months=np.arange(len(df),len(df)+12).reshape(-1,1)
future_sales=model.predict(future_months)
future_dates=pd.date_range(
    start='2026-01-01',
    periods=12,
    freq='MS'
)
future_labels=[date.strftime('%Y-%m')for date in future_dates]
print("\n HISTORICAL SALES DATA\n")
print(df[['Month','Sales']])
forecast_df=pd.DataFrame({
    'Month':future_labels,
    'Predicted Sales':future_sales.round().astype(int)
})
print("\nFORECASTED SALES FOR 2026\n")
print(forecast_df)
plt.figure(figsize=(14,6))
# For Historical sales
plt.plot(
    df['Month'],
    df['Sales'],
    marker='o',
    linewidth=2,
    label='Historical Sales'
)
plt.plot(
    future_labels,
    future_sales,
    marker='o',
    linestyle='--',
    linewidth=2,
    label='Forecasted Sales'
)
for x, y_val in zip(future_labels, future_sales):
    plt.text(
        x,
        y_val,
        str(round(y_val)),
        fontsize=8
    )
plt.title("Sales Forecasting Analysis (2024-2025 Historical, 2026 Forecast)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()