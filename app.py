import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px 
from dash import dcc, html, Dash
from datetime import datetime as dt
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

app= Dash(__name__)
server= app.server

#layout
app.layout= html.Div([
    #1st div input
    html.Div([
        html.P("WELCOME TO MY STOCK DASH APP", className="start"),
        html.Div(
            [
            dcc.Input(
                id= 'input_stockcode',
                type= 'text',
                placeholder= 'Input StockCode',
            ),

            html.Button('Submit', id='submit-stockcode'),
            html.Br(),

            ]), 
        
        html.Div([
            dcc.DatePickerRange(
                id="input_daterange",
                min_date_allowed= dt(2000,1,1),
                max_date_allowed= dt(2022,4,25),
                start_date= dt(2000,1,1).date(),
                end_date= dt(2022,4,25).date(),
                start_date_placeholder_text= "Start Date",
                end_date_placeholder_text= "End Date",
                display_format= 'YY, MM, Do',
                month_format= 'MMMM, YYYY',
                minimum_nights= 7,

                persistence= True,
                persistence_type= 'session',
            ),

            html.Button("OK", id="submit-date-range"),
            html.Br(),

            ]),
        
        html.Div([
            dcc.Input(
                id= "no_of_days",
                type= 'number',
                placeholder= "Input no. of Days",
                min= 1,
                max= 100,
                step= 1,
            ),

            html.Button("Submit", id="submit-no-of-dates"),
            html.Br(),
            html.Button("Forecast", id="forecast"),
            html.Br(),

            ]),
        ],
    className= 'nav'), 
    
    #2nd Div Output
    html.Div(
        [
         dcc.Graph(id='figure-output', figure={})  
        ],
    className= 'content'),
])


@app.callback(
    Output(component_id='figure-output', component_property= 'figure'),
    [Input(component_id='submit-stockcode', component_property= 'n_clicks')],
    [State(component_id='input_stockcode', component_property= 'value')],
    [Input(component_id='submit-date-range', component_property= 'n_clicks')],
    [State(component_id='input_daterange', component_property='start_date')],
    [State(component_id='input_daterange', component_property='end_date')],
    [Input(component_id='submit-no-of-dates', component_property= 'n_clicks')],
    [State(component_id='no_of_days', component_property='value')],
    [Input(component_id='forecast', component_property='n_clicks')]
)
def update_output(n_clicks1, value1, n_clicks2, start_date1, end_date1, n_clicks3, value2, n_clicks4):
    if n_clicks1 is None or n_clicks2 is None or n_clicks3 is None or n_clicks4 is None:
        raise PreventUpdate
    else:
        df= yf.download(value1, start_date1, end_date1)
        df.reset_index(inplace= True)
        days= value2
        def finalfunc(df):
                fi=px.line(df,
                    x="Date",
                    y=['High', 'Low'],
                    title="HIGHS AND LOWS OF {} Comapny".format(value1)
                    )
                return fi
    
        return finalfunc(df)
        

if __name__=="__main__":
    app.run_server(debug= True) 
        
    




    


        

    