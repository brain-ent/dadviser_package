import re
import chardet
from core.utils import DAdviser
import logging

# from flask_session import Session
from flask import Flask, render_template, request, session, redirect, url_for, flash


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = open('secret').read()
# app.config['SESSION_TYPE'] = 'filesystem'

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S.%p', 
    filename='out.log', 
    encoding='utf-8'
)


# Session(app)

rus = re.compile('[^а-яА-ЯёЁ]')

documents_path = "core/texts"
adviser = DAdviser(documents_path)


@app.errorhandler(Exception)
def handle_error(e):
    return redirect(url_for('check'))
"""
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')

        with open('pass') as file:
            user, pswd = file.read().split(':')

        if username == user and password == pswd:
            session['auth'] = True
            return redirect(url_for('check'))

    return render_template("login.html")
"""

@app.route('/', methods=['GET','POST'])
def check():
    result = {}
    logging.info("New request!")
    
    if request.method == "POST":
        file = request.files.get("files")
        
        logging.info("File was upload")    

        if file and file.filename.endswith('.txt'):
            data = file.stream.read()
            encoding = chardet.detect(data)['encoding']
            if encoding:
                data = rus.sub(' ', data.decode(encoding).lower())
                if len(data.split()) > 20:
                    logging.info("SUCCESS!")
                    result = adviser.get_similarity(data, toplist=10)
                else:
                    logging.info("WARN: a few russian words...")
                    flash("В файле мало русских слов (<20)", 'error')
            else:
                logging.info("WARN: bad encoding...")
                flash("Неизвестная кодировка файла", 'error')
        else:
            logging.info("WARN: non txt file...")
            flash("Формат файла не .txt", 'error')

    return render_template('check.html', result=result)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)


