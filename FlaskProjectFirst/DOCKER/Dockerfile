FROM python:3.11

EXPOSE 5000

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

ADD . /FlaskProjectFirst/