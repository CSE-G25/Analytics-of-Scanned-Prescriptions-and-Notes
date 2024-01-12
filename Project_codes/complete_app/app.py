# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from main_code import input_image1
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

#disease_dic= ["Benign","Melignant"]



#from model_predict  import pred_leaf_disease

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Prescription Reader'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Prescription Reader'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)
            file = request.files.get('file')

            print(file)
        #if not file:
         #   return render_template('disease.html', title=title)
        #try:
            img1 = file.read()
            with open('input.png', 'wb') as f:
                    f.write(img1)
            #print(img)
            from application_both_models import all_task
            prediction=all_task()

            print("this is the final sentence returned ",prediction)
            from fuzzywuzzy import fuzz, process

            # Load actual medicine names from the text file
            with open('medicine_names.txt', 'r') as file:
                actual_medicine_names = [line.strip() for line in file]

            # Predicted medicine names from your DL model
            predicted_medicine_names = prediction

            # Function to get the best match and similarity score
            def get_best_match(predicted_name, actual_names):
                matches = process.extractOne(predicted_name, actual_names, scorer=fuzz.token_sort_ratio)
                return matches

            # Replace old names with the most similar names
            improved_predictions = []
        
            for predicted_name in predicted_medicine_names:
                best_match, similarity_score = get_best_match(predicted_name, actual_medicine_names)
                
                # Only consider the match with the maximum similarity score
                improved_predictions.append(best_match)

            print("Improved Predictions:", improved_predictions)



            return render_template('disease-result.html', prediction="The Medicine Detected are As Follows: ",precaution=improved_predictions,title="medical prescription detection")
        #except:
         #   pass
    return render_template('disease.html', title=title)


# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
