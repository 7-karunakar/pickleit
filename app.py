from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

def generate_summary(df):
    summary = {
        "columns": list(df.columns),
        "shape": df.shape,
        "description": df.describe(include='all').fillna("NaN").to_dict()
    }
    return summary

def find_numeric_columns(df):
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) >= 2:
        return numeric_cols[:2]
    elif len(numeric_cols) == 1:
        return numeric_cols[0], numeric_cols[0]
    else:
        return None, None

def plot_matplotlib(df, x_col, y_col):
    fig, ax = plt.subplots()
    df.plot.scatter(x=x_col, y=y_col, ax=ax)
    plt.title('Matplotlib Visualization')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def plot_seaborn(df, x_col, y_col):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    plt.title('Seaborn Visualization')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return f"Error reading file: {e}", 400
        
        # Generate Summary for Original Data
        original_summary = generate_summary(df)
        
        # Find numeric columns for visualization
        x_col, y_col = find_numeric_columns(df)
        if x_col is None or y_col is None:
            return "No suitable numeric columns found for visualization", 400
        
        # Visualize Original Data with Plotly
        try:
            if x_col not in df.columns or y_col not in df.columns:
                return f"Columns {x_col} or {y_col} not found in the dataset", 400
            
            fig_original = px.scatter(df, x=x_col, y=y_col)
            graph_json_original = fig_original.to_json()
        except Exception as e:
            return f"Error generating original Plotly visualization: {e}", 400
        
        # Generate Matplotlib and Seaborn visualizations
        try:
            matplotlib_img = plot_matplotlib(df, x_col, y_col)
            seaborn_img = plot_seaborn(df, x_col, y_col)
        except Exception as e:
            return f"Error generating Matplotlib or Seaborn visualizations: {e}", 400

        # Data Cleaning Example: Remove NaN values
        df_cleaned = df.dropna()
        
        # Generate Summary for Cleaned Data
        cleaned_summary = generate_summary(df_cleaned)
        
        # Visualize Cleaned Data with Plotly
        try:
            if x_col not in df_cleaned.columns or y_col not in df_cleaned.columns:
                return f"Columns {x_col} or {y_col} not found in the cleaned dataset", 400
            
            fig_cleaned = px.scatter(df_cleaned, x=x_col, y=y_col)
            graph_json_cleaned = fig_cleaned.to_json()
        except Exception as e:
            return f"Error generating cleaned Plotly visualization: {e}", 400

        # Return summaries and visualizations
        return jsonify(
            original_summary=original_summary, 
            cleaned_summary=cleaned_summary, 
            original_visualization=graph_json_original, 
            cleaned_visualization=graph_json_cleaned,
            matplotlib_visualization=matplotlib_img,
            seaborn_visualization=seaborn_img
        )

if __name__ == '__main__':
    app.run(debug=True)
