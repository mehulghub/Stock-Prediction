def pred(val):
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.graph_objs as go
    import plotly.express as px 
    from dash import dcc, html, Dash
    from datetime import datetime as dt
    from dash.dependencies import Input, Output, State
    from dash.exceptions import PreventUpdate
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.model import Sequential
    from tensorflow.keras.layers import Dense, LSTM


    #Using MINMAXSCALER to scale the data
    df= yf.download(val, period="max")
    df= df.reset_index['close'] 
      
    from sklearn.preprocessing import MinMaxScaler
    scaler= MinMaxScaler(figure_range= (0,1))
    df1= scaler.transform_fit(np.array(df), reshape=(-1,1))

    #Splitting the dataset
    def split(df1):
        x,y=[],[]
        return x.append(df1[0:(len(df1)*0.9),]), y.append(df1[(len(df1)*0.9):,])
    
    traindata, testdata= split(df1)

    def cosplit(sample, timestep):
        x_tr, y_tr= [],[]
        for i in range(len(sample)- timestep-1):
            x_tr= x_tr.append(sample[0:i+ timestep,0])
            y_tr= y_tr.append(sample[i+timestep,0])
        return  x_tr, y_tr
    x_train, y_train= cosplit(traindata,100) 
    x_test, y_test= cosplit(testdata,100) 


    #Changing into 3-D
    def re(a):
        return a.reshape(a.shape[0], a.shape[1], 1)
    allowed= [x_train, y_train, x_test, y_test]
    for i in allowed:
        i= re(i)
        

    #LSTM model

    physical_devices= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model= Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)),
        LSTM(75,return_sequences= True),
        LSTM(100),
        Dense(1)
    ])
    model.compile(optimizer= 'adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x=x_train, y= y_train, validation_data=(x_test, y_test), batch_size=64, epochs=100, shufle=True, verbose=2)

    #prediction
    train_predict= model.predict(x_train)
    test_predict= model.predict(x_test)
    print(train_predict)

    train_predict= scaler.inverse_transform(train_predict)
    test_predict= scaler.inverse_transform(test_predict)
    y_train= scaler.inverse_transform(y_train)
    y_test= scaler.inverse_transform(y_test)

    import math
    from sklearn.metrics import mean_squared_error
    print(math.sqrt(mean_squared_error(y_train, train_predict)))
    print(math.sqrt(mean_squared_error(y_test, test_predict)))
    
    def pred():
        df_pred=df1[df1.shape[0]-100:] #last 100 inputs from the scaled data df1
        for i in range(100):
            final_pred= model.predict(df_pred) 
            df_pred.append[final_pred]
            df_pred=df_pred[1,]
                
        fi= px.line(
            x=np.array(),
            y=df_pred
        )        
        
        
    
    
    







    
