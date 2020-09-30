import joblib
from flask import Flask, request, render_template

app = Flask(__name__)
Model = joblib.load('modelIris')
# 127.0.0.1:5000/

# facebook.com/

@app.route('/')
def iris():
    # return "Selamat Datang"
    return render_template('iris.html')

# Request ==> Response 
# GET, POST, PATCH, DELETE, dll

@app.route('/klasifikasi', methods=['POST'])
def hasil():
    if request.method == 'POST':
    
        DataUser = request.form
        sl = float(DataUser['sl'])
        sw = float(DataUser['sw'])
        pl = float(DataUser['pl'])
        pw = float(DataUser['pw'])
#         sl = float(input['sl'])
#         sw = float(input['sw'])
#         pl = float(input['pl'])
#         pw = float(input['pw'])   
        # pred = Model.predict([[float(DataUser['sl']), float(DataUser['sw']), float(DataUser['pl']), float(DataUser['pw'])]])[0]

        pred = Model.predict([[sl, sw, pl, pw]])[0]

        return render_template('hasil.html', input=DataUser, predict = pred)


angka = [1,2,5,4,8]
joblib.dump(angka, "angka")

if __name__ == "__main__":

    app.run(debug=True)