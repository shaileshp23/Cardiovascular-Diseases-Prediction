from flask import Flask, render_template, request, redirect, session, url_for
from flask_mysqldb import MySQL
import MySQLdb.cursors
from flask_mail import Mail, Message
from random import randrange
import pickle
import math
import numpy as np
filename = 'final2.pkl'

model = pickle.load(open(filename, 'rb'))


app = Flask(__name__)
app.secret_key = "abcdef"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'shailesh@12'
app.config['MYSQL_DB'] = 'cardio'
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USERNAME"] = "heartokayy@gmail.com"
app.config["MAIL_PASSWORD"] = "xbos hsbn zmlr tadf"
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False


mail = Mail(app)
mysql = MySQL(app)

@app.route("/")
def home():
	if 'username' in session:
		return render_template("home.html", name = session['username'])
	else:
		return redirect(url_for('signup'))

@app.route("/find")
def find():
	if 'username' in session:
		return render_template("find.html", name = session['username'])
	else:
		return redirect(url_for('home'))
	
@app.route('/check', methods=['POST'])
def check():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        bp_lo = int(request.form['bp_lo'])
        bp_hi = int(request.form['bp_hi'])
        cholesterol = int(request.form['cholesterol'])
        heartrate = int(request.form['heartrate'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])
        cardio = int(request.form['cardio'])

        input_data = [age, gender, height, weight, bp_lo, bp_hi, cholesterol, heartrate, smoke, alco, active, cardio]

        score = 0

        if age <= 40:
            score += 1
        elif age >= 41:
            score += 2

        if gender == 1:
            score += 2
        else:
            score += 1

        h = height / 100
        bmi = weight / (h * h)

        if bmi >= 40:
            score += 2
        elif bmi >= 30:
            score += 1

        if bp_hi == 120 and bp_lo == 80:
            score += 0
        elif bp_lo >= 80 and bp_lo <= 90 and bp_hi >= 120 and bp_hi <= 130:
            score += 1
        elif bp_hi >= 130 and bp_lo >= 90:
            score += 2
        elif (bp_lo < 80 and bp_hi < 120) or (bp_lo > 80 and bp_hi > 120):
            score += 3
        elif (
            (bp_lo < 80 and bp_hi > 120)
            or (bp_hi < 120 and bp_lo > 80)
            or (bp_lo == 80 and bp_hi > 120)
            or (bp_lo > 80 and bp_hi == 120)
            or (bp_lo == 80 and bp_hi < 120)
            or (bp_lo < 80 and bp_hi == 120)
        ):
            score += 4

        sc = 0

        if age >= 30 and age <= 50:
            if heartrate in range(170, 180):
                sc = sc + 1
            else:
                sc = sc + 2
        elif age >= 50 and age <= 60:
            if heartrate in range(160, 170):
                sc = sc + 1
            else:
                sc = sc + 2
        elif age >= 70 and age <= 80:
            if heartrate in range(150, 160):
                sc = sc + 1
            else:
                sc = sc + 2
        elif age >= 80 and age <= 90:
            if heartrate in range(140, 150):
                sc = sc + 1
            else:
                sc = sc + 2

        ch = 0

        if cholesterol <= 200:
            ch = ch + 1
        elif cholesterol >= 201 and cholesterol <= 239:
            ch = ch + 2
        elif cholesterol >= 240:
            ch = ch + 3

        score += (ch + sc + smoke + alco + active + cardio)
        percentage = (score / 19) * 100

        prediction = model.predict([input_data])[0]

        # Assuming 'prediction' is the result from the model.predict() statement

        if prediction == 0:
            result_statement = "The person is predicted not to have a heart diseases."
        else:
            result_statement = "The person may have or may develop a heart disease in the future if this lifestyle is continued."

        result_statement += f"\nChances of cardiovascular disease: {math.ceil(percentage * 100) / 100}%"

        return render_template('result.html', result_statement=result_statement)


@app.route("/signup", methods = ["GET", "POST"])
def signup():
	if request.method == "POST":
		em = request.form["em"]
		un = request.form["un"]
		pw = ""
		text = "0123456789"
		for i in range(6):
			pw = pw + text[randrange(len(text))]
		print(pw)
		msg = Message("Welcome to Heart Okayy!", sender = "heartokayy@gmail.com", recipients = [em])
		msg.body = "Greetings from Heart Okayy! Your password is " + str(pw)
		mail.send(msg)
		con = None
		try:
			cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
			cursor.execute('insert into user values(%s, %s, %s)',(un, em, pw))
			mysql.connection.commit()
			return render_template("login.html", msg = "Password has been mailed to you")
		except Exception as e:
			cursor.rollback()
			return render_template("signup.html", msg = "User already exists" + str(e))
	else:
		return render_template("signup.html")

@app.route("/login", methods = ["GET", "POST"])
def login():
	if request.method == "POST":
		un = request.form["un"]
		pw = request.form["pw"]
		con = None
		try:
			cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
			cursor.execute('select * from user where username = %s and password = %s',(un,pw))
			data = cursor.fetchall()
			if len(data) == 0:
				return render_template("login.html", msg = "invalid login")
			else:	
				session['loggedin'] = True
				session['username'] = un
				return redirect( url_for('home'))
		
		except Exception as e:
			msg = "Issue " + str(e)
			return render_template("login.html", msg = msg)
	else:
		return render_template("login.html")
	
@app.route("/changepassword", methods = ['GET', 'POST'])
def changepassword():
	if request.method == 'POST':
		em = request.form['em']
		pw = request.form['pw']
		rpw = request.form['rpw']

		if pw == rpw:
			cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
			cursor.execute('update user set password=%s where emailid=%s',(rpw,em))
			msg = 'Password changed successfully!'
			mysql.connection.commit()
			return render_template('changepassword.html', msg = msg)
		else:
			msg = 'Password does not matched!'
			return render_template('changepassword.html', msg = msg)
	else:
		msg = 'Please try again!'
	return render_template('changepassword.html', msg = msg)

@app.route("/forgot", methods = ["GET", "POST"])
def forgot():
	if request.method == "POST":
		un = request.form["un"]
		em = request.form["em"]
		con = None
		try:
			cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
			cursor.execute('select * from user where username = %s',(un,))
			data = cursor.fetchall()
			if len(data) == 0:
				return render_template("forgot.html", msg = "invalid login")
			else:	
				session['username'] = un
				pw1 = ""
				text = "0123456789"
				for i in range(6):
					pw1 = pw1 + text[randrange(len(text))]
				print(pw1)
				msg = Message("Hello again from Heart Okayy", sender = "heartokayy@gmail.com", recipients = [em])
				msg.body = "Greetings from Heart Okayy! Seems like you forgot your password. Your new password is " + str(pw1)
				mail.send(msg)
				try:
					cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
					cursor.execute("update user set password=%s where username=%s",(pw1,un))
					mysql.connection.commit()
					return render_template("login.html", msg = "Password has been mailed to you")
				except Exception as e:
					con.rollback()
					return render_template("forgot.html", msg = "Some Issue: " + str(e))
		except Exception as e:
			msg = "Issue " + str(e)
			return render_template("forgot.html", msg = msg)	
	else:
		return render_template("forgot.html")			

@app.route('/logout')
def logout():
	session.pop('username', None)
	session.clear()	
	return redirect(url_for("login"))

if __name__ == "__main__":
	app.run(debug = True)
