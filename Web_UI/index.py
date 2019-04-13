from flask import Flask, render_template
import tablib
import os
 
app = Flask (__name__)
dataset = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'../Attendance.csv')) as f:
    dataset.csv = f.read()
 
@app.route("/")
def index():
    data = dataset.html
    #return dataset.html
    return render_template('index.html', data=data)
 
if __name__ == "__main__":
    app.run()
