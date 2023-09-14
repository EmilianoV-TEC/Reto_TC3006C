from flask import Flask, render_template, request, Response
import xgboost as xgb
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app = Flask(__name__)

df_x_train = pd.read_csv('data/df_x_train.csv')
df_x_train['date'] = pd.to_datetime(df_x_train.date)
df_x_train = df_x_train.set_index('date')

df_x_test = pd.read_csv('data/df_x_test.csv')
df_x_test['date'] = pd.to_datetime(df_x_test.date)
df_x_test = df_x_test.set_index('date')

df_y_train = pd.read_csv('data/df_y_train.csv')
df_y_train['date'] = pd.to_datetime(df_y_train.date)
df_y_train = df_y_train.set_index(['store_nbr', 'family', 'date']).sort_index()
df_y_train = df_y_train.unstack(['store_nbr','family'])

model = xgb.XGBRegressor()
model.load_model('models/xgb_model.json')

df_y_train_pred = pd.DataFrame(model.predict(df_x_train), index=df_x_train.index, columns=df_y_train.columns)
df_y_test_pred = pd.DataFrame(model.predict(df_x_test), index=df_x_test.index, columns=df_y_train.columns)

@app.route('/<store_number>/<family>/<start_year>/plot.png')
def get_prediction_graph(store_number, family, start_year):
    plt.cla()
    plt.clf()

    if family == "BREAD AND BAKERY":
        family = "BREAD/BAKERY"

    store_number = int(store_number)
    year_string = f'{start_year}-01-01'

    df_y_train_filtered = df_y_train.loc[year_string:]
    df_y_train_pred_filtered = df_y_train_pred.loc[year_string:]

    ax = df_y_train_filtered.loc(axis=1)['sales', store_number, family].plot(color = 'gray', style='.-', markeredgecolor = 'black', markerfacecolor = 'black', figsize = (20, 9))
    ax = df_y_train_pred_filtered.loc(axis=1)['onpromotion', store_number, family].plot(ax = ax)
    ax = df_y_test_pred.loc(axis=1)['onpromotion', store_number, family].plot(ax = ax)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas')
    ax.set_title(f'Ventas de la familia {family} en la tienda {store_number}')

    fig = ax.figure

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    #get_prediction_graph(23, 'CLEANING', 2016)
    return render_template("index.html")

if __name__ =='__main__':
    #app.debug = True
    app.run(debug = True)
