from urllib import request

import pandas as pd
from flask import Flask, jsonify, request, render_template, url_for

# Create a Flask app instance
app = Flask(__name__)

# Define a simple route
@app.route('/')
def home():
    stock_files = [
    ("Data Sets/A.csv", "A"),
    ("Data Sets/AACG.csv", "AACG"),
    ("Data Sets/AAMC.csv", "AAMC"),
    ("Data Sets/AAOI.csv", "AAOI"),
    ("Data Sets/AAT.csv", "AAT"),
    ("Data Sets/ABBV.csv", "ABBV"),
    ("Data Sets/ABTX.csv", "ABTX"),
    ("Data Sets/ACA.csv", "ACA"),
    ("Data Sets/ACAM.csv", "ACAM"),
    ("Data Sets/ACAMU.csv", "ACAMU")
    ]

    # Initialize an empty DataFrame to store last rows
    combined_last_rows = pd.DataFrame()

    for file, stock_name in stock_files:
        df = pd.read_csv(file)
        # Infer the date format automatically
        df['Date'] = pd.to_datetime(df['Date'], format=None, errors='coerce')

        # Drop rows where 'Date' couldn't be converted
        df.dropna(subset=['Date'], inplace=True)

        df.set_index('Date', inplace=True)
        df.insert(0, 'Stock', stock_name)
        combined_last_rows = pd.concat([combined_last_rows, df.tail(1)])
    table_html = combined_last_rows.to_html(classes='table table-primary table-striped', index=False)
    script_url = "static/JS/clickableRows.js"
    script_tag = f'<script src="{script_url}"></script>'
    return render_template('index.html', table=table_html + script_tag)

@app.route('/table', methods=['GET', 'POST'])
def table():
    if request.method == 'POST':
        data = request.get_json()  # Retrieve JSON data
        stock_name = data.get('stock_name')  # Extract stock_name
        if stock_name:
            print(stock_name)
            return jsonify({"selected_stock": stock_name})
        return jsonify({"error": "Stock name not provided"}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
