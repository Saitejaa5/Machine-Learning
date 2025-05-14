from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd

# Load the saved models
kmeans_model = joblib.load('customer_segmentation.pkl')
pca_model = joblib.load('pca_model.pkl')


app = Flask(__name__)

# Generate cluster visualization
def plot_clusters():
    plt.figure(figsize=(10, 6))
    
    # Generate sample data points (since we don't have original dataset here)
    sample_data = np.random.rand(100, 3)  
    reduced_data = pca_model.transform(sample_data)
    predictions = kmeans_model.predict(reduced_data)

    # Plot clusters
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=predictions, palette='viridis')
    plt.title("Customer Segments")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Convert plot to base64 image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    segment = None
    if request.method == 'POST':
        try:
            # Get user input from form
            input_features = [float(request.form[key]) for key in request.form.keys()]
            input_features = np.array(input_features).reshape(1, -1)

            # Transform input using PCA
            transformed_input = pca_model.transform(input_features)

            # Predict segment
            segment = kmeans_model.predict(transformed_input)[0]
        except Exception as e:
            segment = f"Error: {str(e)}"

    plot_url = plot_clusters()
    return render_template('index.html', plot_url=plot_url, segment=segment)

if __name__ == '__main__':
    app.run(debug=True)
