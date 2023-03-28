import re
import chardet
from core.utils import DAdviser

from flask_session import Session
from flask import Flask, render_template, request, session, redirect, url_for


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = open('secret').read()
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)

rus = re.compile('[^а-яА-ЯёЁ]')

documents_path = "core/texts"
adviser = DAdviser(documents_path)


@app.errorhandler(Exception)
def handle_error(e):
    return redirect(url_for('check'))

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

@app.route('/', methods=['GET','POST'])
def check():
    if not session.get('auth'):
        return redirect(url_for('login'))

    if request.method == "POST":
        files = request.files.getlist("files")
        for file in files:
            if file.filename.endswith('.txt'):
                data = file.stream.read()
                encoding = chardet.detect(data)['encoding']
                data = rus.sub(' ', data.decode(encoding).lower())
                result = adviser.get_similarity(data, toplist=10)

                return render_template('check.html', result=result)

    return render_template('check.html')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)


