from flask import Flask,render_template

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('map_out.html')


# 챗봇 엔진 query 전송 API
#@app.route('/load_wind', methods=['GET'])
@app.route('/load_wind', methods=['GET'])
def load_wind():
    return render_template('wind_out.html')


if __name__ == '__main__':
    app.debug = True
    app.run()