FROM python:3.10

WORKDIR /app

RUN pip install numpy chardet nltk gensim pymorphy2 flask

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]