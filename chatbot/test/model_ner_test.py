import sys
sys.path.insert(0, '../')
from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin',
               #userdic='../utils/user_dic.tsv')
               userdic='../utils/mtn_user_dict.tsv')


#ner = NerModel(model_name='../models/ner/ner_model_basicset.h5', proprocess=p)
#ner = NerModel(model_name='../models/ner/ner_model_use_cpsdic_usrdic_mtndat.h5', proprocess=p)
ner = NerModel(model_name='../models/ner/ner_model_use_cpsdic_mtndic_mtndat.h5', proprocess=p)


query = '오늘 오전 13시 2분에 탕수육 주문 하고 싶어요'
query1 = "관악산 장군봉"
query2 = "노악산 주록리구간"
query3 = "봉개민오름 송정동구간"
query4 = "10시 30분"
query5 = "1번"
query_list = [query1,query2,query3,query4,query5]

for i in query_list:
    predicts = ner.predict(i)
    tags = ner.predict_tags(i)

    print(i)
    print(predicts)
    print(tags)


# 결과정리
# 1. 훈련: corpus dict + user dict, ner_data
#    테스트: corpus dict + user dict
#    결과: 시간 빼고, 다 안됨
#    질문: 관악산은 왜 단어 그대로 인지되는지?
#          노악산은 노악으로 분리됨, 노악 -> corpus에 없음
#    테스트: corpus dict + mtn user dict
#    결과: 1결과와 동일, 단어분리는 mtn user dict로 되나 예측은 안됨
#
# 2. 훈련: corpus dict + user dict, mtn_ner_data
#    테스트: corpus dict + user dict
#    결과: LC f1 성능이 올라감, 숫자 뺴고 잘나오는데, 단어 분리 못함, 
#          index_to_ner 바뀌어서 수동으로 바꿔줌
#    테스트: corpus dict + mtn user dict
#    결과: 숫자빼고 잘됨
#
# 3. 훈련: corpus dict + mtn user dict, mtn_ner_data
#    테스트: corpus dict + user dict
#    결과: TI, PS 성능이 떨어짐, LC는 0.01오름
#          단어가 분리 되었는데도 일부가 LC가 나와 결과는 잘나옴
#    테스트: corpus dict + mtn user dict
#    결과: 숫자빼고 잘됨

# 결론
# 최적 세팅
# 사전: 원본 corpus
# 모델 훈련시 mtn user, mtn_ner_data
# 모델 테스트 mtn user