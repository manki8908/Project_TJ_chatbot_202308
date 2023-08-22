import sys
sys.path.insert(0, '../')

from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin',
              #userdic='../utils/user_dic.tsv')
              userdic='../utils/mtn_user_dict.tsv')

#intent = IntentModel(model_name='../models/intent/intent_model.h5', proprocess=p)
intent = IntentModel(model_name='../models/intent/intent_model_use_mtndat_mtndic.h5', proprocess=p)
query1 = "관악산 장군봉"
query2 = "노악산 주록리구간"
query3 = "봉개민오름 송정동구간"
query4 = "10시 30분"
query5 = "1번"
query_list = [query1,query2,query3,query4,query5]

for i in query_list:
    predict = intent.predict_class(i)
    predict_label = intent.labels[predict]

    print(i)
    print("의도 예측 클래스 : ", predict)
    print("의도 예측 레이블 : ", predict_label)


# intent 실험 결과 정리
# 1. 훈련 corpus dict + user dict, mtn_data_set
#    테스트 corpus dict + user dict
#    결과: 다 잘됨    
# 2. corpus dict + mtn_user dict, mtn_data_set: 
#    테스트 corpus dict + user dict
#    결과: 봉개민오름 --> 욕설, 나머지 잘됨
# 3. corpus dict + mtn_user dict, mtn_data_set: 
#    테스트 corpus dict + mtn_user dict
#    결과: 다 잘됨

# 질문 
# corpus에 '봉개민오름'이 없는 어떻게 인덱싱 되었을까?

