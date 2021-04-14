from flask import Flask,abort,render_template,request,redirect,url_for
from werkzeug import secure_filename
import os
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
@app.route('/')
def index():
    return redirect(url_for('hello'))

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name = None):
    return render_template('hello.html',name=name)

@app.route('/upload/',methods = ['GET','POST'])
def upload_file():
    if request.method =='POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            return hello()
    return render_template('file_upload.html')


if __name__ == '__main__':
    app.run(debug = True)