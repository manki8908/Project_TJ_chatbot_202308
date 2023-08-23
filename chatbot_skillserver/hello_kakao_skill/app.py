from flask import Flask, request, jsonify
app = Flask(__name__)


# 카카오톡 텍스트형 응답
@app.route('/api/sayHello', methods=['POST'])
#@app.route('/api/sayHello', methods=['GET','POST'])
def sayHello():
    body = request.get_json()
    print(body)
    print(body['userRequest']['utterance'])

    responseBody = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "안녕 hello I'm Ryan"
                    }
                }
            ]
        }
    }

    return responseBody
    #return jsonify(responseBody)


# 카카오톡 이미지형 응답
@app.route('/api/showHello', methods=['POST'])
#@app.route('/api/showHello', methods=['GET','POST'])
def showHello():
    body = request.get_json()
    print(body)
    print(body['userRequest']['utterance'])

    responseBody = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleImage": {
                        "imageUrl": "https://t1.daumcdn.net/friends/prod/category/M001_friends_ryan2.jpg",
                        "altText": "hello I'm Ryan"
                    }
                }
            ]
        }
    }

    return responseBody
    #return jsonify(responseBody)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    #app.run(debug=True)
