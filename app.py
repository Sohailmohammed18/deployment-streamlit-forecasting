
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st 
import matplotlib.pyplot as plt
st.title('Time Series Forecasting')
st.sidebar.header('User Input Parameters')
Total_beds = st.sidebar.number_input("Insert the number of Total beds")
Days = st.sidebar.number_input("Insert the number of days of prediction",min_value = 1, max_value = 365)
df=pd.read_csv('Beds_Occupied.csv')
for x in ['Total Inpatient Beds']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    df.loc[df[x] < min,x] = np.nan
    df.loc[df[x] > max,x] = np.nan

df['Available Beds']= Total_beds-df['Total Inpatient Beds']
df["date"] = pd.to_datetime(df.collection_date, format = "%d-%m-%Y")
df2=df.set_index('date')
df2.sort_index(inplace = True)
df2.rename(columns=
{
"Available Beds": "Available_beds",

}, inplace=True)
r=pd.date_range(start='2020-06-15',end='2021-06-15')
df3=df2.reindex(r).rename_axis('date').reset_index()
df3=df3.set_index('date')
df4 = df3.interpolate(method='time', axis=0)
train = df4.iloc[0:280,:]
test = df4.iloc[280:,:]
hwe_model_add_sea = ExponentialSmoothing(train["Available_beds"],seasonal="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_sea_test = hwe_model_add_sea.predict(start = test.index[0],end = test.index[-1])
def main():
    st.title('Time Series Forecasting')
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Time Series Forecasting for the Availabilty of beds' </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

input=Days
prediction=hwe_model_add_sea.forecast(input)
st.subheader('Predicted Result')
st.write(prediction)
st.subheader("VISUALIZE FORECASTED DATA")
st.line_chart(df4['Available_beds'])
st.line_chart(prediction)
plt.plot(df4['Available_beds'], label='original')
plt.plot(prediction, label='forecast')
plt.title('Forecast')
plt.legend(loc='upper left', fontsize=8)
plt.show()