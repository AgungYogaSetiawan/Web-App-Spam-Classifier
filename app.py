from flask import Flask, render_template, url_for, request
from sklearn.svm import SVC
import pickle

# load model from pickle file
filename = 'final_model.pkl'
with open(filename, 'rb') as fm:
    vec, tf_trans, model_svc = pickle.load(fm)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        count_vec = vec.transform(data).toarray()
        tfTrans = tf_trans.transform(count_vec)
        prediction = model_svc.predict(tfTrans)
        return render_template('home.html',prediction=prediction)
    
if __name__ == '__main__':
    app.run(debug=True)










# #spam_clf(['hey dita good morning'])

# iface = gr.Interface(
#     fn=spam_clf,
#     inputs=gr.inputs.Dataframe(type='array', headers=['Message'], col_count=1,  row_count=1, col_width=350),
#     outputs='text'
# )
# print(iface.launch())





