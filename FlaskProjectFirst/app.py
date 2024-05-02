import json

from datetime import datetime, timedelta
from flask import render_template, redirect, request, url_for, session
from flask import Flask
import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
import pymysql
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:rootroot1@db/flaskDB'
app.config['SECRET_KEY'] = ("7dY5syEJUMA6zT8eEMWUA6g6hwyEJUMWUwe88yEJ80mM9Yi39KrY4yA6e880mM9Yi39K"
                            "rYY57XezpsCRIBLYCwtiryEJUA6EYNfdJob1kxDThmJv5Wb6zSFBiHAcSnEcVPmj5d1LL")
db = SQLAlchemy(app)


class POSTS(db.Model):
    __tablename__ = 'Posts'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), unique=False, nullable=False)
    datetime = db.Column(db.String(20), unique=False, nullable=False)
    author = db.Column(db.String(200), unique=False, nullable=False)
    text = db.Column(db.String(3000), unique=False, nullable=False)


class USERS(db.Model):
    __tablename__ = 'Users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), unique=True, nullable=False)
    email = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(1000), unique=False, nullable=False)
    repeat_password = db.Column(db.String(1000), unique=False, nullable=False)
    admin = db.Column(db.Integer, unique=False, nullable=False, default=0)


class DATASET(db.Model):
    __tablename__ = 'Dataset'
    id = db.Column(db.Integer, primary_key=True)
    baths = db.Column(db.String(100), unique=False, nullable=False)
    bedrooms = db.Column(db.String(100), unique=False, nullable=False)
    area = db.Column(db.String(100), unique=False, nullable=False)
    price = db.Column(db.String(100), unique=False, nullable=False)
    predict_price = db.Column(db.String(100), unique=False, nullable=False)
    error = db.Column(db.String(100), unique=False, nullable=False)

class DATASET_GRAD(db.Model):
    __tablename__ = 'Dataset_grad'
    id = db.Column(db.Integer, primary_key=True)
    longitude = db.Column(db.String(100), unique=False, nullable=False)
    latitude = db.Column(db.String(100), unique=False, nullable=False)
    housing_median_age = db.Column(db.String(100), unique=False, nullable=False)
    total_rooms = db.Column(db.String(100), unique=False, nullable=False)
    total_bedrooms = db.Column(db.String(100), unique=False, nullable=False)
    population = db.Column(db.String(100), unique=False, nullable=False)
    households = db.Column(db.String(100), unique=False, nullable=False)
    median_income = db.Column(db.String(100), unique=False, nullable=False)
    weights = db.Column(db.Text, unique=False, nullable=True)


with app.app_context():
    db.create_all()
    admins = USERS.query.filter_by(email='annplug@mail.ru').first()
    if admins is None:
        db.session.add(
            USERS(username='annplug', email='annplug@mail.ru',
                  password=generate_password_hash('Annplug'),
                  repeat_password=generate_password_hash('Annplug'),
                  admin=1
                  )
        )
        db.session.commit()


async def status(username):
    user = USERS.query.filter_by(username=username).first()
    if user is not None:
        return user.admin
    session.pop('username', None)
    return 0


@app.route('/', methods=['GET', 'POST'])
async def main_fun():
    page = request.args.get('page', 1, type=int)
    posts = POSTS.query.paginate(page=page, per_page=5)
    return render_template("main_view.html",
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           posts=posts)


@app.route('/aboutblog/', methods=['GET', 'POST'])
async def aboutblog_fun():
    return render_template("about_view.html",
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           )


@app.route('/aboutpost/<int:post_id>', methods=['GET', 'POST'])
async def aboutpost_fun(post_id):
    post = POSTS.query.filter_by(id=post_id).first()
    if post is not None:
        return render_template("aboutpost_view.html",
                               status=await status(session.get('username')),
                               username=session.get('username'),
                               post=post)
    return redirect(url_for('main_fun'))


@app.route('/search/', methods=['POST', 'GET'])
async def search():
    search = f"%{request.args.get('search')}%"
    page = request.args.get('page', 1, type=int)
    posts = POSTS.query.filter(
        POSTS.title.like(search) | POSTS.author.like(search) |
        POSTS.text.like(search) | POSTS.datetime.like(search)).paginate(page=page, per_page=5)
    return render_template("main_view.html",
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           posts=posts)


@app.route('/createpost/', methods=['GET', 'POST'])
async def createpost_fun():
    title = request.form.get('title')
    text = request.form.get('text')
    username = session.get('username')
    date = str(datetime.datetime.now().date())
    if title is not None and text is not None and username is not None:
        db.session.add(POSTS(title=title, text=text, author=username, datetime=date))
        db.session.commit()
    return redirect(url_for('main_fun'))


@app.route('/login/', methods=['GET', 'POST'])
async def login_fun():
    email = request.form.get('email')
    password = request.form.get('password')
    if email is not None and password is not None:
        user = USERS.query.filter_by(email=email).first()
        if user is not None:
            if check_password_hash(user.password, password):
                session['username'] = user.username
    return redirect(url_for('main_fun'))


@app.route('/registration/', methods=['GET', 'POST'])
async def registration_fun():
    username = request.form.get('username')
    email = request.form.get('email')
    password1 = request.form.get('password1')
    password2 = request.form.get('password2')
    if username is not None and email is not None and password1 is not None and password2 is not None:
        if password1 == password2:
            ps = generate_password_hash(password1)
            db.session.add(USERS(username=username,
                                 email=email,
                                 password=ps,
                                 repeat_password=ps,
                                 )
                           )
            db.session.commit()
            session['username'] = username
    return redirect(url_for('main_fun'))


@app.route('/logout/')
async def logout_fun():
    session.pop('username', None)
    return redirect(url_for('main_fun'))


# _______________________
@app.route('/polreg/', methods=['GET', 'POST'])
async def polreg_fun():
    return render_template('polreg_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           data=DATASET.query.all()
                           )


@app.route('/polreg/start/', methods=['GET', 'POST'])
async def polreg_start_fun():
    DATASET.query.delete()
    data = pd.read_csv('FlaskProjectFirst/DOCKER/DATASET.csv', index_col=0)
    X = data.drop('price', axis=1)[:10000]
    Y = data['price'][:10000]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=42)

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X_train.values)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_train.values)

    X_new_poly = poly_features.transform(X_test.values)
    y_new = lin_reg.predict(X_new_poly)

    for i in range(len(y_new)):
        db.session.add(
            DATASET(baths=str(X_test.values[i][0]),
                    bedrooms=str(X_test.values[i][1]),
                    area=str(X_test.values[i][2]),
                    price=str(int(y_test.values[i])),
                    predict_price=str(int(y_new[i])),
                    error=str(int(y_test.values[i]) - int(y_new[i])),
                    ))
    db.session.commit()
    return redirect(url_for('polreg_fun'))

@app.route('/gradbust/', methods=['GET', 'POST'])
async def gradbust_fun():
    return render_template('grad_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           data=DATASET_GRAD.query.all()
                           )

@app.route('/gradbust/start1/', methods=['GET', 'POST'])
async def gradbust_start_fun():
    DATASET_GRAD.query.delete()
    data = pd.read_csv('FlaskProjectFirst/DOCKER/DATASET_GRAD.csv', index_col=0)
    start_date = datetime.strptime('2023-01-01', "%Y-%m-%d %H:%M")
    end_date = start_date + datetime.timedelta(hours=2)
    if data.empty:
        print("Нет доступных данных для указанной даты.")
        return

    features = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    model = xgb.XGBRegressor()
    model.load_model('mymodel_mysql/my_xgb_model.json')
    predictions = model.predict(features_scaled)

    weights = model.get_booster().get_dump()

    for i in range(len(predictions)):
        db.session.add(
            DATASET_GRAD(longitude=str(data['longitude'].iloc[i]),
                         latitude=str(data['latitude'].iloc[i]),
                         housing_median_age=str(data['housing_median_age'].iloc[i]),
                         total_rooms=str(data['total_rooms'].iloc[i]),
                         total_bedrooms=str(data['total_bedrooms'].iloc[i]),
                         population=str(data['population'].iloc[i]),
                         households=str(data['households'].iloc[i]),
                         median_income=str(data['median_income'].iloc[i]),
                         weights=json.dumps(weights)
                         )
        )
    db.session.commit()
    return redirect(url_for('gradbust_fun'))


@app.route('/neyronka/', methods=['GET', 'POST'])
async def neyronka_fun():
    return render_template('neyronka_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           data=DATASET_GRAD.query.all()
                           )

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[['total_bedrooms', 'households', 'population', 'median_income']].values)
    return scaled_features, scaler

def make_predictions(model, X_last, forecast_hours):
    predictions = []
    for _ in range(forecast_hours):
        pred = model.predict(X_last)[0][0]
        predictions.append([pred])
        X_last = np.roll(X_last, -1, axis=1)
        X_last[0, -1, :] = np.append(pred, X_last[0, -2, 1:])
    return predictions


def inverse_transform(predictions, scaler):
    predictions_scaled = scaler.inverse_transform(np.hstack([predictions, np.zeros((len(predictions), 5))]))[:, 0]
    return predictions_scaled

@app.route('/neyronka/start2/', methods=['GET', 'POST'])
async def neyronka_start_fun():
    DATASET_GRAD.query.delete()
    data = pd.read_csv('FlaskProjectFirst/DOCKER/DATASET_GRAD.csv', index_col=0)

    if data.empty:
        print("Нет доступных данных для указанной даты.")
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(data[['total_bedrooms', 'households', 'population', 'median_income']].values)

    model_path = 'mymodel/my_rnn_model'
    db_path = 'air_quality.db'
    time_step = 100

    start_date_str = input("2021-03-01 19:00")
    forecast_hours = int(input("24"))

    try:
        model = load_model(model_path)
        start_date = pd.to_datetime(start_date_str)
        if data.empty:
            return "Нет доступных данных для указанной даты."

        scaled_features, scaler = scale_data(data)
        X_last = scaled_features[-time_step:].reshape(1, time_step, 6)

        predictions = make_predictions(model, X_last, forecast_hours)

        predictions_scaled = inverse_transform(predictions, scaler)

        for i in range(len(predictions)):
            db.session.add(
                DATASET_GRAD(longitude=str(data['longitude'].iloc[i]),
                             latitude=str(data['latitude'].iloc[i]),
                             housing_median_age=str(data['housing_median_age'].iloc[i]),
                             total_rooms=str(data['total_rooms'].iloc[i]),
                             total_bedrooms=str(data['total_bedrooms'].iloc[i]),
                             population=str(data['population'].iloc[i]),
                             households=str(data['households'].iloc[i]),
                             median_income=str(data['median_income'].iloc[i]),
                             weights=json.dumps(predictions_scaled)
                             )
            )
        db.session.commit()

        return redirect(url_for('neyronka_fun'))
    except ValueError as e:
        return str(e)




# _______________________

@app.route('/adminpanel/', methods=['POST', 'GET'])
async def adminpanel_fun():
    return render_template('adminpanel_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           )


@app.route('/adminpanel/posts/', methods=['POST', 'GET'])
async def adminpanel_posts_fun():
    page = request.args.get('page', 1, type=int)
    posts = POSTS.query.paginate(page=page, per_page=5)
    users = USERS.query.all()
    return render_template('adminpanel_posts_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           posts=posts,
                           authors=[user.username for user in users]
                           )


@app.route('/adminpanel/users/', methods=['POST', 'GET'])
async def adminpanel_users_fun():
    page = request.args.get('page', 1, type=int)
    users = USERS.query.paginate(page=page, per_page=5)
    return render_template('adminpanel_users_view.html',
                           status=await status(session.get('username')),
                           username=session.get('username'),
                           users=users,
                           )


@app.route('/adminpanel/operations/changetitle/<int:post_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_changetitle_fun(post_id):
    input = request.args.get('input')
    if input is not None:
        post = POSTS.query.filter_by(id=post_id).first()
        if post is not None:
            post.title = input
            db.session.commit()
    return redirect(url_for('adminpanel_posts_fun'))


@app.route('/adminpanel/operations/changeauthor/<int:post_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_changeauthor_fun(post_id):
    input = request.args.get('input')
    if input is not None:
        post = POSTS.query.filter_by(id=post_id).first()
        if post is not None:
            post.author = input
            db.session.commit()
    return redirect(url_for('adminpanel_posts_fun'))


@app.route('/adminpanel/operations/changetext/<int:post_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_changetext_fun(post_id):
    input = request.args.get('input')
    if input is not None:
        post = POSTS.query.filter_by(id=post_id).first()
        if post is not None:
            post.text = input
            db.session.commit()
    return redirect(url_for('adminpanel_posts_fun'))


@app.route('/adminpanel/operations/deletepost/<int:post_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_deletepost_fun(post_id):
    db.session.delete(POSTS.query.filter_by(id=post_id).first())
    db.session.commit()
    return redirect(url_for('adminpanel_posts_fun'))


@app.route('/adminpanel/operations/changeusername/<int:user_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_changeusername_fun(user_id):
    input = request.args.get('input')
    if input is not None:
        user = USERS.query.filter_by(id=user_id).first()
        if user is not None:
            user.username = input
            db.session.commit()
    return redirect(url_for('adminpanel_users_fun'))


@app.route('/adminpanel/operations/deleteuser/<int:user_id>/', methods=['POST', 'GET'])
async def adminpanel_operations_deleteuser_fun(user_id):
    db.session.delete(USERS.query.filter_by(id=user_id).first())
    db.session.commit()
    return redirect(url_for('adminpanel_users_fun'))


if __name__ == '__main__':
    app.run(host="0.0.0.0")
