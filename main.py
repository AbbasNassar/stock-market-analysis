import os
from urllib import request

import numpy as np

from Algorithms.stockTraining import Analysis
import pandas as pd
from flask import Flask, request, render_template, url_for, redirect

app = Flask(__name__)

stock_files = [
    ("DataSets/A.csv", "A"),
    ("DataSets/AAMC.csv", "AAMC"),
    ("DataSets/AAOI.csv", "AAOI"),
    ("DataSets/AAT.csv", "AAT"),
    ("DataSets/ABBV.csv", "ABBV"),
    ("DataSets/ABTX.csv", "ABTX"),
    ("DataSets/ACA.csv", "ACA")
]

company_names = {
    "A": "Alpha Innovations Inc.",
    "AAMC": "American Asset Management Corporation",
    "AAOI": "Advanced Optical Instruments LLC",
    "AAT": "Atlantic Assets Trust",
    "ABBV": "AbbVie Pharmaceuticals",
    "ABTX": "Abtex Industries",
    "ACA": "Academy Enterprises"
}

# Define a simple route
@app.route('/')
def home():

    combined_last_rows = pd.DataFrame()

    for file, stock_name in stock_files:
        df = pd.read_csv(file)
        newdf = df.drop("Date", axis='columns')
        newdf.insert(0, 'Stock', stock_name)
        combined_last_rows = pd.concat([combined_last_rows, newdf.tail(1)])
    table_html = combined_last_rows.to_html(classes='table table-light table-striped', index=False)
    script_url = "static/JS/clickableRows.js"
    script_tag = f'<script src="{script_url}"></script>'
    return render_template('index.html', table=table_html + script_tag)

@app.route('/table', methods=['POST'])
def table():
    if request.method == 'POST':
        data = request.get_json()
        stock_name = data.get('stock_name')
        if stock_name != "Stock":
            print(stock_name)
            return redirect(url_for('render_stock_data', stock_name=stock_name))

@app.route('/stock/<stock_name>', methods=['GET'])
def render_stock_data(stock_name):
    company_name = company_names.get(stock_name)
    analysis = Analysis(stock_name)
    path_to_close_plot = analysis.get_close_plot()
    path_to_close_plot = "../" + path_to_close_plot
    path_to_rsi_plot = analysis.get_rsi_plot()
    path_to_rsi_plot = "../" + path_to_rsi_plot
    path_to_bollinger_band_plot = analysis.get_bollinger_bands_plot()
    path_to_bollinger_band_plot = "../" + path_to_bollinger_band_plot
    classification_report, confusion_matrix, accuracy_score, path_to_dt = analysis.train_model_with_df(test_size=0.2)
    path_to_dt = "../" + path_to_dt
    mse_knn, r2_knn, mape_knn, accuracy_knn, knn_plot_path = analysis.train_model_with_KNN(test_size=0.2, n=3, lookback=5)
    knn_plot_path= "../" + knn_plot_path
    return render_template('stock.html', stockName=stock_name,
                           stock=company_name,
                           close_plot_url=path_to_close_plot,
                           rsi_plot_url=path_to_rsi_plot,
                           bollinger_band_plot_url=path_to_bollinger_band_plot,
                           classification_report=classification_report,
                           confusion_matrix=confusion_matrix,
                           accuracy_score=accuracy_score,
                           dt_plot_url=path_to_dt,
                           mse_knn=mse_knn,
                           r2_knn=r2_knn,
                           mape_knn=mape_knn,
                           accuracy_knn=accuracy_knn,
                           knn_plot_url=knn_plot_path)


@app.route('/process', methods=['POST'])
def process():

    # Get the user input from the request
    data = request.get_json()
    user_input = data.get('value', '')
    user_input_float = float(user_input)
    stock_name = data.get('stockName', '')
    print(user_input, stock_name)
    analysis = Analysis(stock_name)
    path_to_close_plot = analysis.get_close_plot()
    path_to_close_plot = "../" + path_to_close_plot
    path_to_rsi_plot = analysis.get_rsi_plot()
    path_to_rsi_plot = "../" + path_to_rsi_plot
    path_to_bollinger_band_plot = analysis.get_bollinger_bands_plot()
    path_to_bollinger_band_plot = "../" + path_to_bollinger_band_plot
    classification_report, confusion_matrix, accuracy_score, path_to_dt = analysis.train_model_with_df(test_size=user_input_float)
    path_to_dt = "../" + path_to_dt
    mse_knn, r2_knn, mape_knn, accuracy_knn, knn_plot_path = analysis.train_model_with_KNN(test_size=user_input_float, n=3,
                                                                                           lookback=5)
    knn_plot_path = "../" + knn_plot_path
    # Path to the HTML file
    result_file_path = os.path.join("Templates", "machineLearningResults.html")

    # Read the HTML file as a string
    with open(result_file_path, 'r') as file:
        html_content = file.read()

    confusion_matrix_string = np.array2string(confusion_matrix)
    accuracy_score_string = str(accuracy_score)
    mse_knn_string = str(mse_knn)
    r2_knn_string = str(r2_knn)
    mape_knn_string = str(mape_knn)
    accuracy_knn_string = str(accuracy_knn)

    # Replace placeholder with user input
    html_content = html_content.replace("{{ classification_report }}", classification_report)
    html_content = html_content.replace("{{ confusion_matrix }}", confusion_matrix_string)
    html_content = html_content.replace("{{ accuracy_score }}", accuracy_score_string)
    html_content = html_content.replace("{{ dt_plot_url }}", path_to_dt)
    html_content = html_content.replace("{{ mse_knn }}", mse_knn_string)
    html_content = html_content.replace("{{ r2_knn }}", r2_knn_string)
    html_content = html_content.replace("{{ mape_knn }}", mape_knn_string)
    html_content = html_content.replace("{{accuracy_knn}}", accuracy_knn_string)
    html_content = html_content.replace("{{ knn_plot_url }}", knn_plot_path)
    return html_content


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
