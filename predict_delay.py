import pandas as pd
import joblib

lr_model = joblib.load('lr_model.pkl')
rf_model = joblib.load('rf_model.pkl')

sample = pd.DataFrame([[
2015,2,21,2,3,45,12,830,845,15,120,900,1030,1045,0,0,5,0,7,2,0,33.94,-118.40
]], columns=[
'YEAR','MONTH','DAY','DAY_OF_WEEK','AIRLINE','ORIGIN_AIRPORT',
'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','DEPARTURE_TIME',
'DEPARTURE_DELAY','SCHEDULED_TIME','DISTANCE','SCHEDULED_ARRIVAL',
'ARRIVAL_TIME','DIVERTED','CANCELLED','AIR_SYSTEM_DELAY',
'SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY',
'WEATHER_DELAY','LATITUDE','LONGITUDE'
])

prediction_lr = lr_model.predict(sample)
print("Linear Regression based predicted delay: ", round(prediction_lr[0]), "minutes")

prediction_rf = rf_model.predict(sample)
print("Random Forest based predicted delay: ", round(prediction_rf[0]), "minutes")
