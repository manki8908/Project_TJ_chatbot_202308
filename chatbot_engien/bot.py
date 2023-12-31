import threading
import json

from config.DatabaseConfig import *
from utils.Database import Database
from utils.BotServer import BotServer
from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel
from utils.FindAnswer import FindAnswer
from models.mtn_load.plot_load import find_load


# 등산로 로드

geo_path = '../DATA/FRT000801/moutain_load.geojson'
search_class = find_load(geo_path)

# 전처리 객체 생성
p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin',
               userdic='utils/mtn_user_dict.tsv')

#head = r'C:\workspace\VScode_project\project2\chatbot_book_ex'
# 의도 파악 모델
intent = IntentModel(model_name='models/intent/intent_model_use_mtndat_mtndic.h5', proprocess=p)

# 개체명 인식 모델
ner = NerModel(model_name='models/ner/ner_model_use_cpsdic_mtndic_mtndat.h5', proprocess=p)


def to_client(conn, addr, params):
    db = params['db']

    try:
        db.connect()  # 디비 연결

        # 데이터 수신
        read = conn.recv(2048)  # 수신 데이터가 있을 때 까지 블로킹, 2048은 수신데이터 최대크기 byte
        print('===========================')
        print('Connection from: %s' % str(addr))

        if read is None or not read:
            # 클라이언트 연결이 끊어지거나, 오류가 있는 경우
            print('클라이언트 연결 끊어짐')
            exit(0)


        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']

        # 의도 파악
        intent_predict = intent.predict_class(query)
        intent_name = intent.labels[intent_predict]

        # 개체명 파악
        ner_predicts = ner.predict(query)
        ner_tags = ner.predict_tags(query)


        # 답변 검색
        try:
            # 답변 검색
            f = FindAnswer(db)
            answer_text, answer_image = f.search(intent_name, ner_tags)
            answer = f.tag_to_word(ner_predicts, answer_text)

            # 등산로 이미지 생성
            latlon_load = search_class.get_latlon(query)
            search_class.plot_load(latlon_load)

        except:
            answer = "죄송해요 무슨 말인지 모르겠어요. 조금 더 공부 할게요."
            answer_image = None

        send_json_data_str = {
            "Query" : query,
            "Answer": answer,
            "AnswerImageUrl" : answer_image,
            "Intent": intent_name,
            "NER Predicts": str(ner_predicts),
            "NER tags": str(ner_tags)
        }
        message = json.dumps(send_json_data_str)
        conn.send(message.encode())

    except Exception as ex:
        print(ex)

    finally:
        if db is not None: # db 연결 끊기
            db.close()
        conn.close()


# 이하 책에 설명 없음
if __name__ == '__main__':

    # 질문/답변 학습 디비 연결 객체 생성
    db = Database(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME
    )
    print("DB 접속")

    # 소켓서버 설정
    port = 5050  # 소켓포트
    listen = 100  # 쓰레드 개수

    # 봇 서버 동작
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start")

    while True:
        conn, addr = bot.ready_for_client()
        params = {
            "db": db
        }

        client = threading.Thread(target=to_client, args=(
            conn,  # 클라이언트 연결 소켓
            addr,  # ㅇ클라이언트 연결 주소 정보
            params # 쓰레드 함수 파라미터
        ))
        client.start()
