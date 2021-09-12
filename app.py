import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import streamlit as st 
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

st.sidebar.header('User Input Parameters')
Total_beds = st.sidebar.number_input("Enter the number of Total beds")
Days = st.sidebar.number_input("Enter the number of days of prediction",min_value = 1, max_value = 365)
df = pd.read_csv("Beds_Occupied.csv")
for x in ['Total Inpatient Beds']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    df.loc[df[x] < min,x] = np.nan
    df.loc[df[x] > max,x] = np.nan

df['Available_beds']= Total_beds-df['Total Inpatient Beds']
df["date"] = pd.to_datetime(df.collection_date, format="%d-%m-%Y")
df2=df.set_index('date')
df3 = df2.interpolate(method='time', axis=0)
r=pd.date_range(start='2020-06-15',end='2021-06-15')


def main():

    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Time Series Forecasting on the Availabilty of beds </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
main()

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df3.index, y=df3['Available_beds']))
	fig.layout.update(title_text='Time Series data of Available Beds with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()
input=Days
r=pd.date_range(start='2020-06-15',end='2021-06-15')
df4=df3.reindex(r).rename_axis('date').reset_index()

#prophet prediction
df_train = df4[['date','Available_beds']]
df_train = df_train.rename(columns={"date": "ds", "Available_beds": "y"})

m = Prophet(interval_width=0.95,changepoint_range=0.9,daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=input)
forecast = m.predict(future)
forecast1=forecast[['ds','yhat']]
forecast1.head()
forecast1=forecast.rename(columns={'ds':'Date', 'yhat':'Forecasted Beds'})
st.subheader('Forecast data')
st.write(forecast1[['Date','Forecasted Beds']].tail(input),)

st.write(f'Forecast data for {input} days')
st.title('Visualization of the forecasted data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
