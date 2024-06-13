import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import numpy as np

# Define the Entry widgets as global variables
tickerSymbolEntry = None
forecastOutEntry = None
displayDaysEntry = None

def predict():
    global tickerSymbolEntry, forecastOutEntry, displayDaysEntry

    # Get the ticker symbol and the prediction period from the Entry widgets
    tickerSymbol = tickerSymbolEntry.get()
    forecast_out = int(forecastOutEntry.get())
    display_days = int(displayDaysEntry.get())

    # Download historical data for desired ticker symbol
    tickerData = yf.Ticker(tickerSymbol)
    df = tickerData.history(period='1d', start='2010-1-1', end=datetime.date.today())

    # Use only Close price for prediction
    df = df[['Close']]

    # Create another column (the target) shifted 'n' units up
    df['Prediction'] = df[['Close']].shift(-forecast_out)

    # Create the independent data set (X)
    X = df.drop(columns=['Prediction'])

    # Create the dependent data set (y)
    y = df['Prediction']

    # Remove the last 'n' rows where 'n' is the forecast_out
    X = X[:-forecast_out]
    y = y[:-forecast_out]

    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and train the Linear Regression Model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # Test the model using score
    lr_confidence = lr.score(x_test, y_test)
    print("lr confidence: ", lr_confidence)

    # Create an array of 'n' future dates starting from the next day
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(days=1), periods=forecast_out)

    # Create a DataFrame for the future dates
    future_df = pd.DataFrame(index=future_dates, columns=df.columns)

    # Fill the future_df DataFrame with the last known price
    future_df['Close'] = df['Close'].iloc[-1]

    # Append the future_df to the original df
    df = pd.concat([df, future_df])

    # Set x_forecast equal to the last 'n' rows of the original data set from Close column
    x_forecast = df.drop(columns=['Prediction']).tail(forecast_out)

    # Print linear regression model predictions for the next 'n' days
    lr_prediction = lr.predict(x_forecast)

    # Create a new DataFrame for the predicted prices
    predictions = pd.DataFrame(lr_prediction, index=x_forecast.index, columns=['Prediction'])

    # Plot the actual closing prices and the predicted prices for the last 'm' days
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'].tail(display_days))
    plt.plot(predictions)
    plt.legend(['Actual', 'Prediction'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.show()

root = tk.Tk()
root.geometry("300x200")
root.title("Numera")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

label = ttk.Label(mainframe, text="Welcome to Numera!", font=("Helvetica", 16, "bold"))
label.grid(column=1, row=1, sticky=(tk.W, tk.E))

tickerSymbolLabel = ttk.Label(mainframe, text="Enter the ticker symbol:")
tickerSymbolLabel.grid(column=1, row=2, sticky=tk.W)
tickerSymbolEntry = ttk.Entry(mainframe)
tickerSymbolEntry.grid(column=1, row=3, sticky=(tk.W, tk.E))

forecastOutLabel = ttk.Label(mainframe, text="Enter the prediction period (in days):")
forecastOutLabel.grid(column=1, row=4, sticky=tk.W)
forecastOutEntry = ttk.Entry(mainframe)
forecastOutEntry.grid(column=1, row=5, sticky=(tk.W, tk.E))

displayDaysLabel = ttk.Label(mainframe, text="Enter the number of days to display:")
displayDaysLabel.grid(column=1, row=6, sticky=tk.W)
displayDaysEntry = ttk.Entry(mainframe)
displayDaysEntry.grid(column=1, row=7, sticky=(tk.W, tk.E))

button = ttk.Button(mainframe, text="Predict", command=predict)
button.grid(column=1, row=8, sticky=tk.W)

for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

root.mainloop()