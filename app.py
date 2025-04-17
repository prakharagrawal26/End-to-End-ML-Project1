from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        form_data = request.form.to_dict()

        try:
            data = CustomData(
                gender=form_data.get('gender'),
                race_ethnicity=form_data.get('ethnicity'),
                parental_level_of_education=form_data.get('parental_level_of_education'),
                lunch=form_data.get('lunch'),
                test_preparation_course=form_data.get('test_preparation_course'),
                reading_score=float(form_data.get('reading_score')),
                writing_score=float(form_data.get('writing_score'))
            )

            pred_df = data.get_data_as_frame()
            print("Input DataFrame:", pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0], input_data=form_data)

        except Exception as e:
            print("Error during prediction:", str(e))
            return render_template('home.html', results="Error: Check your input values.", input_data=form_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
