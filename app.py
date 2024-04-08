from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('adv_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def predict_price():
    prediction_result = None
    if request.method == 'POST':
        # Extract input data from the form
        TV_spend = int(request.form['TV_spend'])
        Radio_spend = int(request.form['Radio_spend'])
        Newspaper_spend = int(request.form['Newspaper_spend'])

        # Prepare the input data for prediction
        new_data = [[TV_spend, Radio_spend, Newspaper_spend]]

        # Perform prediction
        predicted_price = model.predict(new_data)[0]
        prediction_result = round(predicted_price, 2)

    return render_template('index.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)