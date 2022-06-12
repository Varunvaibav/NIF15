import io
from application import app
from flask import Response, render_template
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests as r
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas_datareader as pd_data
from cProfile import label
import datetime
from datetime import date
from datetime import datetime
from datetime import timedelta
from statistics import mean
import yfinance as yf
from bs4 import BeautifulSoup
from urllib.request import urlopen

import chart_studio.plotly as py
import math as m
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error

ticker = {"Tech Mahindra" : "TECHM",
    "Reliance Industries" : "RELIANCE",
    "TCS" : "TCS",
    "HDFC Bank" : "HDFCBANK",
    "Hindustan Unilever" : "HINDUNILVR",
    "ICICI Bank" : "ICICIBANK",
    "ITC" : "ITC",
    "Kotak Mahindra Bank" : "KOTAKBANK",
    "Infosys" : "INFY",
    "SBI" : "SBIN",
    "Bajaj Finance" : "BAJFINANCE",
    "Bharati Airtel" : "BHARTIARTL",
    "Axis Bank" : "AXISBANK",
    "Asian Paints" : "ASIANPAINT",
    "HCL Technologies" : "HCLTECH" }

@app.route('/', defaults={'path': ''})
@app.route('/home')
def home():
    ticker = {"Tech Mahindra" : "TECHM",
    "Reliance Industries" : "RELIANCE",
    "TCS" : "TCS",
    "HDFC Bank" : "HDFCBANK",
    "Hindustan Unilever" : "HINDUNILVR",
    "ICICI Bank" : "ICICIBANK",
    "ITC" : "ITC",
    "Kotak Mahindra Bank" : "KOTAKBANK",
    "Infosys" : "INFY",
    "SBI" : "SBIN",
    "Bajaj Finance" : "BAJFINANCE",
    "Bharati Airtel" : "BHARTIARTL",
    "Axis Bank" : "AXISBANK",
    "Asian Paints" : "ASIANPAINT",
    "HCL Technologies" : "HCLTECH" }
    

    avgGraphJSON, high, low = avgGraph(ticker)
    highLowGraphJSON = plot_high_low()
    return render_template('home.html', avgGraphJSON = avgGraphJSON, ticker=ticker, high=high, low=low, highLowGraphJSON=highLowGraphJSON)


def avgGraph(tickerDict):

    ticker = tickerDict

    df = []

    for i in ticker:
        df.append(getDF(ticker[i]))

    data = pd.DataFrame(df)
    average = data.mean(axis=0)
    fig = px.line(average)
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Stock Price')
    fig.update_xaxes(
    ticktext=["9.15", "10.15", "11.15", "12.15", "1.15", "2.15", "3.15"],
    tickvals=["0", "60", "120", "180", "240", "300", "360"],
    )
    fig.update_layout(showlegend=False)
    avgGraphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    high = round(max(average),2)
    low = round(min(average),2)
    return avgGraphJSON, high, low


def getDF(tickerID):

    action_getURL = 'https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/'+str(tickerID)+'/daily?intervalInMinutes=1'
    res = r.get(action_getURL)
    search_cookies = res.cookies
    get_data = {'method' : 'GET'}
    res_get = r.get(action_getURL, data=get_data, cookies=search_cookies)
    stock_values = res_get.json()["candles"]
    df = pd.DataFrame(stock_values)
    return df[4]

def plot_high_low():

    today = date.today()
    start_date = today - timedelta(days=9)

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    tickerYF = ['TECHM.NS', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'KOTAKBANK.NS',
    'INFY', 'SBIN.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'HCLTECH.NS']

    for i in tickerYF:
        data = yf.download(i, start=start_date, end=today)
        data = pd.DataFrame(data)
        df1[i] = data.High
        df2[i] = data.Low
    highAverage = df1.mean(axis=1)
    lowAverage = df2.mean(axis=1)

    fig = Figure()
    axis = fig.add_subplot(1,1,1)
    ys1 = highAverage.values
    ys2 = lowAverage.values
    xs = highAverage.index.tolist()
    
    trace1 = go.Scatter(
                    x = xs,
                    y = ys1,
                    mode = "lines",
                    name = "High",
                    marker = dict(color = "#293462"),
                    )
    trace2 = go.Scatter(
                    x = xs ,
                    y = ys2,
                    mode = "lines",
                    name = "Low",
                    marker = dict(color = "#F24C4C")
                    )

    data =[trace1,trace2]

    fig = go.Figure(data)
    fig.update_layout(showlegend=False)
    highLowGraphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return highLowGraphJSON


@app.route('/plot/<stockName>.png')
def plot_png(stockName):
    fig = create_figure(stockName)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(stockName):

    ticker = {"Tech Mahindra" : "TECHM",
    "Reliance Industries" : "RELIANCE",
    "TCS" : "TCS",
    "HDFC Bank" : "HDFCBANK",
    "Hindustan Unilever" : "HINDUNILVR",
    "ICICI Bank" : "ICICIBANK",
    "ITC" : "ITC",
    "Kotak Mahindra Bank" : "KOTAKBANK",
    "Infosys" : "INFY",
    "SBI" : "SBIN",
    "Bajaj Finance" : "BAJFINANCE",
    "Bharati Airtel" : "BHARTIARTL",
    "Axis Bank" : "AXISBANK",
    "Asian Paints" : "ASIANPAINT",
    "HCL Technologies" : "HCLTECH" }

    
    action_getURL = 'https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/'+str(ticker[stockName])+'/daily?intervalInMinutes=1'
    res = r.get(action_getURL)
    search_cookies = res.cookies
    get_data = {'method' : 'GET'}
    res_get = r.get(action_getURL, data=get_data, cookies=search_cookies)
    stock_values = res_get.json()["candles"]
    df = pd.DataFrame(stock_values)
    fig = Figure()
    axis = fig.add_subplot(1,1,1)
    xs = df[4]
    axis.plot(xs)
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    axis.axis("off")

    return fig

    

@app.route('/stocks/<stockName>')
def liveChart(stockName):

    ticker = {"Tech Mahindra" : "TECHM",
    "Reliance Industries" : "RELIANCE",
    "TCS" : "TCS",
    "HDFC Bank" : "HDFCBANK",
    "Hindustan Unilever" : "HINDUNILVR",
    "ICICI Bank" : "ICICIBANK",
    "ITC" : "ITC",
    "Kotak Mahindra Bank" : "KOTAKBANK",
    "Infosys" : "INFY",
    "SBI" : "SBIN",
    "Bajaj Finance" : "BAJFINANCE",
    "Bharti Airtel" : "BHARTIARTL",
    "Axis Bank" : "AXISBANK",
    "Asian Paints" : "ASIANPAINT",
    "HCL Technologies" : "HCLTECH" }

    df = getDF(ticker[stockName])
    fig = px.line(df)
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Stock Price')
    fig.update_xaxes(
    ticktext=["9.15", "10.15", "11.15", "12.15", "1.15", "2.15", "3.15"],
    tickvals=["0", "60", "120", "180", "240", "300", "360"],
    )
    fig.update_layout(showlegend=False)
    stockGraphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    links = {"Tech Mahindra" : "tech-mahindra-ltd",
    "Reliance Industries" : "reliance-industries-ltd",
    "TCS" : "tata-consultancy-services-ltd",
    "HDFC Bank" : "hdfc-bank-ltd",
    "Hindustan Unilever" : "hindustan-unilever-ltd",
    "ICICI Bank" : "icici-bank-ltd",
    "ITC" : "itc-ltd",
    "Kotak Mahindra Bank" : "kotak-mahindra-bank-ltd",
    "Infosys" : "infosys-ltd",
    "SBI" : "state-bank-of-india",
    "Bajaj Finance" : "bajaj-finance-ltd",
    "Bharti Airtel" : "bharti-airtel-ltd",
    "Axis Bank" : "axis-bank-ltd",
    "Asian Paints" : "asian-paints-ltd",
    "HCL Technologies" : "hcl-technologies-ltd" }


    url = "https://groww.in/stocks/" + str(links[stockName])
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    details = soup.find(class_="acs67Para").text
    companyDetails = details[280:]
    about = ".".join(companyDetails.split(".")[:3])

    tickerYF = {"Tech Mahindra" : "TECHM.NS",
    "Reliance Industries" : "RELIANCE.NS",
    "TCS" : "TCS.NS",
    "HDFC Bank" : "HDFCBANK.NS",
    "Hindustan Unilever" : "HINDUNILVR.NS",
    "ICICI Bank" : "ICICIBANK.NS",
    "ITC" : "ITC.NS",
    "Kotak Mahindra Bank" : "KOTAKBANK.NS",
    "Infosys" : "INFY",
    "SBI" : "SBIN",
    "Bajaj Finance" : "BAJFINANCE.NS",
    "Bharti Airtel" : "BHARTIARTL.NS",
    "Axis Bank" : "AXISBANK.NS",
    "Asian Paints" : "ASIANPAINT.NS",
    "HCL Technologies" : "HCLTECH.NS" }

    today = date.today()
    start_date = today - timedelta(days=120)
    data = yf.download(tickerYF[stockName], start=start_date, end=today)
    data = pd.DataFrame(data)

    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data.Open, high=data.High,
                low=data.Low, close=data.Close)
                     ])

    fig.update_layout(xaxis_rangeslider_visible=False)
    candleGraphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    predictionGraphJSON = prediction(tickerYF[stockName])
    pieChartJSON = piechart(tickerYF[stockName])
    highLowJSON = highLowGraph(tickerYF[stockName])

    df = getDF(ticker[stockName])
    high = max(df)
    low = min(df)


    return render_template('stocks.html', companyName = stockName, 
    stockGraphJSON=stockGraphJSON, about=about, candleGraphJSON=candleGraphJSON, 
    predictionGraphJSON=predictionGraphJSON, pieChartJSON=pieChartJSON, highLowJSON=highLowJSON,
    high=high, low=low)


def get_data(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        return data

def prediction(ticker):
    data = get_data(ticker)
    df = pd.DataFrame(data=data)
    df = df[["Close"]].copy()
    df_log = np.log(df["Close"])
    start=date.today()
    end = datetime(start.year,start.month,start.day+6)
    test = pd.date_range(start=start, end=end, freq='D')
    test2 = pd.DataFrame(test)
    model_autoARIMA = auto_arima(df_log, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    pred = pd.DataFrame(model_autoARIMA.predict(n_periods = 7),index = test)
    pred['exp_data'] = np.exp(pred)
    total_pred = pd.DataFrame(model_autoARIMA.predict(n_periods = len(df)),index = df.index)
    total_pred['exp_data'] = np.exp(total_pred)

    trace1 = go.Scatter(
                    x = df.index,
                    y = df.Close,
                    mode = "lines",
                    name = "Actual price",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    )
    trace2 = go.Scatter(
                    x =pred.index ,
                    y = pred.exp_data,
                    mode = "lines",
                    name = "Predicted price",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)')
                    )

    data =[trace1,trace2]

    fig = go.Figure(data)
    fig.update_layout(xaxis_rangeslider_visible=False)

    predictionGraphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return predictionGraphJSON

def piechart(ticker):
    df = pd.read_csv("marketcap.csv")

    a = 0
    for i in range(len(df)):
        if df['Symbol'][i] != ticker:
            a += df['Market Capitalisation (In lakhs)'][i]
        else:
            b = df['Market Capitalisation (In lakhs)'][i]

    fig = px.pie(df, values=[a,b], names=['Other companies', ticker])
    fig.update_layout(xaxis_rangeslider_visible=False)

    piechartJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return piechartJSON

def highLowGraph(ticker):
    today = date.today()
    start_date = today - timedelta(days=9)

    dataYF = yf.download(ticker, start=start_date, end=today)
    dataYF = pd.DataFrame(dataYF)

    fig = Figure()
    xs = dataYF.index.tolist()
    ys1 = dataYF.High
    ys2 = dataYF.Low
    
    trace1 = go.Scatter(
                    x = xs,
                    y = ys1,
                    mode = "lines",
                    name = "High",
                    marker = dict(color = "#293462"),
                    )
    trace2 = go.Scatter(
                    x = xs ,
                    y = ys2,
                    mode = "lines",
                    name = "Low",
                    marker = dict(color = "#F24C4C")
                    )

    data =[trace1,trace2]

    fig = go.Figure(data)
    fig.update_layout(showlegend=False)
    highLowGraphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return highLowGraphJSON


@app.route('/analyze')
def analyze():
    return render_template('analyze.html')