import joblib
from flask import Flask, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)
Model = joblib.load('Hotelbookingcancelation')
# 127.0.0.1:5000/

# facebook.com/

@app.route('/')
def Hotelbookingcancelation():
    df=pd.read_csv("hotel_bookings.csv")
    list_arrival_date_month=['arrival_date_month_January','arrival_date_month_February','arrival_date_month_March','arrival_date_month_April','arrival_date_month_June','arrival_date_month_November']
    list_meal=['meal_FB','meal_HB','meal_Undefined']
    list_market_segment=['market_segment_Corporate','market_segment_Direct','market_segment_Groups','market_segment_Offline TA/TO']
    list_distribution_channel=['distribution_channel_Corporate','distribution_channel_Direct','distribution_channel_TA/TO']
    list_reserved_room_type=['reserved_room_type_A','reserved_room_type_D','reserved_room_type_E','reserved_room_type_F']
    list_assigned_room_type=['assigned_room_type_A','assigned_room_type_B','assigned_room_type_C','assigned_room_type_D','assigned_room_type_E','assigned_room_type_F','assigned_room_type_I','assigned_room_type_K']
    list_deposit_type=['deposit_type_No Deposit','deposit_type_Non Refund']
    list_agent=['agent_0.0','agent_1.0','agent_3.0','agent_6.0','agent_7.0','agent_8.0','agent_9.0',
    'agent_12.0','agent_14.0','agent_16.0','agent_19.0',
    'agent_20.0','agent_21.0','agent_22.0','agent_27.0','agent_28.0','agent_29.0','agent_31.0',
    'agent_34.0','agent_36.0','agent_37.0',
    'agent_40.0','agent_41.0','agent_42.0','agent_44.0','agent_56.0','agent_58.0','agent_68.0',
    'agent_69.0','agent_85.0','agent_86.0',
    'agent_89.0','agent_115.0','agent_119.0',
    'agent_134.0','agent_138.0','agent_143.0',
    'agent_146.0','agent_152.0','agent_154.0',
    'agent_156.0','agent_162.0','agent_168.0',
    'agent_170.0','agent_196.0','agent_220.0',
    'agent_229.0','agent_235.0','agent_236.0',
    'agent_241.0','agent_243.0','agent_248.0',
    'agent_250.0','agent_251.0','agent_257.0','agent_286.0',
    'agent_314.0','agent_323.0','agent_326.0','agent_330.0',
    'agent_464.0','agent_492.0','agent_495.0']
    list_company=['company_0.0','company_40.0','company_67.0',
    'company_154.0','company_202.0','company_219.0','company_223.0','company_348.0','company_385.0']
    list_customer_type=['customer_type_Contract','customer_type_Group','customer_type_Transient','customer_type_Transient-Party']
    

    # return "Selamat Datang"
    return render_template('Hotelbookingcancelation.html',list_arrival_date_month=list_arrival_date_month, list_meal=list_meal,list_market_segment=list_market_segment,
     list_distribution_channel=list_distribution_channel,list_reserved_room_type=list_reserved_room_type,list_assigned_room_type=list_assigned_room_type,
     list_deposit_type=list_deposit_type,list_agent=list_agent,list_company=list_company,list_customer_type=list_customer_type)

# Request ==> Response 
# GET, POST, PATCH, DELETE, dll

@app.route('/klasifikasi', methods=['POST'])
def hasil():
    X = pd.read_csv('hotel_booking_cleaned_2.csv')
    if request.method == 'POST':
        DataUser = request.form

        lead_time = float(DataUser['lead_time'])
        arrival_date_week_number = float(DataUser['arrival_date_week_number'])
        stays_in_week_nights = float(DataUser['stays_in_week_nights'])
        adults = float(DataUser['adults'])
        babies = float(DataUser['babies'])
        is_repeated_guest = float(DataUser['is_repeated_guest'])
        previous_cancellations = float(DataUser['previous_cancellations'])
        previous_bookings_not_canceled = float(DataUser['previous_bookings_not_canceled'])
        booking_changes = float(DataUser['booking_changes'])
        days_in_waiting_list = float(DataUser['days_in_waiting_list'])
        adr = float(DataUser['adr'])
        required_car_parking_spaces = float(DataUser['required_car_parking_spaces'])
        total_of_special_requests = float(DataUser['total_of_special_requests'])
        price = float(DataUser['price'])

        arrival_date_month = DataUser['arrival_date_month']
        meal = DataUser['meal']
        market_segment = DataUser['market_segment']
        distribution_channel = DataUser['distribution_channel']
        reserved_room_type = DataUser['reserved_room_type']
        assigned_room_type = DataUser['assigned_room_type']
        deposit_type = DataUser['deposit_type']
        agent = DataUser['agent']
        company = DataUser['company']
        customer_type = DataUser['customer_type']
      
        def predict_cancel__(lead_time,
       arrival_date_week_number,
       stays_in_week_nights, adults, babies,
       is_repeated_guest, previous_cancellations,
       previous_bookings_not_canceled, booking_changes, days_in_waiting_list,adr,required_car_parking_spaces,
       total_of_special_requests, price,arrival_date_month,meal,market_segment,distribution_channel,reserved_room_type,
       assigned_room_type,deposit_type, agent,company, customer_type):    
    
            month_index = np.where(X.columns==(DataUser['arrival_date_month']))[0][0]
            meal_index = np.where(X.columns==(DataUser['meal']))[0][0]    
            market_segment_index = np.where(X.columns==(DataUser['market_segment']))[0][0]    
            distribution_channel_index = np.where(X.columns==(DataUser['distribution_channel']))[0][0] 
            reserved_room_type_index = np.where(X.columns==(DataUser['reserved_room_type']))[0][0] 
            assigned_room_type_index = np.where(X.columns==(DataUser['assigned_room_type']))[0][0]  
            deposit_type_index = np.where(X.columns==(DataUser['deposit_type']))[0][0] 
            customer_type_index = np.where(X.columns==(DataUser['customer_type']))[0][0] 
            x = np.zeros(len(X.columns))

            x[0] = lead_time
            x[1] = arrival_date_week_number
            x[2] = stays_in_week_nights
            x[3] = adults
            x[4] = babies
            x[5] = is_repeated_guest
            x[6] = previous_cancellations
            x[7] = previous_bookings_not_canceled
            x[8] = booking_changes
            x[9] = days_in_waiting_list
            x[10] = adr
            x[11] = required_car_parking_spaces
            x[12] = total_of_special_requests
            x[13] = price

            x[month_index] = 1
                        
            x[meal_index] = 1
                        
            x[market_segment_index] = 1
                    
            x[distribution_channel_index] = 1

            x[reserved_room_type_index] = 1
                        
            x[assigned_room_type_index] = 1
                        
            x[deposit_type_index] = 1

            x[customer_type_index] = 1
        
            return Model.predict([x])[0]
            
    pred = predict_cancel__(lead_time, arrival_date_week_number, stays_in_week_nights, adults, babies,
                is_repeated_guest, previous_cancellations,previous_bookings_not_canceled, booking_changes, days_in_waiting_list,adr,required_car_parking_spaces,
                total_of_special_requests, price,arrival_date_month,meal,market_segment,distribution_channel,reserved_room_type,
                assigned_room_type,deposit_type, agent,company, customer_type)

    return render_template('hasil.html', input=DataUser, predict = pred)

if __name__ == "__main__":

    app.run(debug=True)