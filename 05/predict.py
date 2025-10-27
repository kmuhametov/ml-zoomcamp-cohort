import pickle
input_file = 'pipeline_v1.bin'
with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)
lead = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
X = dv.transform([lead])
y_pred = model.predict_proba(X)[0, 1]
print('input:', lead)
print('output:', y_pred)



