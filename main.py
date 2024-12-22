# from fastapi import FastAPI
# from emergencymy import audio2text, text2summary, predict, recommend_hospital3
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import json
# import emergencymy as em

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# # 전역 변수 초기화
# path = "./"
# map_key = json.loads(open(path + "map_key.txt", "r").read())
# c_id, c_key = map_key["c_id"], map_key["c_key"]

# emergency = pd.read_csv(path + "응급실 정보.csv")
# emergency_list = emergency.to_dict("records")

# save_directory = path + "fine_tuned_bert_my"
# model = AutoModelForSequenceClassification.from_pretrained(save_directory)
# tokenizer = AutoTokenizer.from_pretrained(save_directory)

# @app.get("/hospital_by_module")
# async def get_hospital(request: str, latitude: float, longitude: float):
#     """
#     병원 추천 API
#     - 요약된 텍스트 (request), 현재 위치 (위도와 경도)를 입력받아 병원을 추천
#     """
#     # Step 1: 응급 등급 예측
#     predicted_class, _ = predict(request, model, tokenizer)

#     # Step 2: 병원 추천
#     if predicted_class <= 2:
#         result = recommend_hospital3(emergency_list, latitude, longitude, c_id, c_key)
#         # 결과를 JSON 형태로 반환
#         return result.to_dict(orient="records")
#     else:
#         return {"message": "응급 상황이 아니므로 병원 추천이 필요 없습니다."}


#####################################################################
# # 이건 오디오랑 오디오파일명, 위도, 경도 받는것(미완성, 수정 필요)
# # FastAPI 초기화
# from fastapi import FastAPI
# from emergencymy import audio2text, text2summary, predict, recommend_hospital3
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import json
# import emergencymy as em

# app = FastAPI()

# # 데이터 및 모델 초기화
# path = "./"
# openai_api_key = em.load_file(path + "api_key.txt")
# map_key = json.loads(em.load_file(path + "map_key.txt"))
# c_id, c_key = map_key["c_id"], map_key["c_key"]

# emergency = pd.read_csv(path + "응급실 정보.csv")
# emergency_list = emergency.to_dict("records")

# save_directory = path + "fine_tuned_bert"
# model = AutoModelForSequenceClassification.from_pretrained(save_directory)
# tokenizer = AutoTokenizer.from_pretrained(save_directory)

# @app.get("/process")
# async def process_pipeline(audio_path: str, filename: str, start_lat: float, start_lng: float):
#     # Step 1: 오디오에서 텍스트 추출
#     transcript = audio2text(audio_path, filename)

#     # Step 2: 텍스트 요약
#     summary = text2summary(transcript)

#     # Step 3: 응급 등급 예측
#     predicted_class, _ = predict(summary, model, tokenizer)

#     # Step 4: 병원 추천
#     if predicted_class <= 2:
#         result = recommend_hospital3(emergency_list, start_lat, start_lng, c_id, c_key)
#         return result.to_dict(orient="records")
#     else:
#         return {"message": "응급 상황이 아니므로 병원 추천이 필요 없습니다."}


##############################################################################3


from fastapi import FastAPI
from emergencymy import audio2text, text2summary, predict, recommend_hospital3
from db_manger import insert_log
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import emergencymy as em

import os
import openai

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# 전역 변수 초기화
path = "./"
openai.api_key = em.load_file(path + "api_key.txt")
os.environ["OPENAI_API_KEY"] = openai.api_key
map_key = json.loads(open(path + "map_key.txt", "r").read())
c_id, c_key = map_key["c_id"], map_key["c_key"]
#c_id, c_key = 'hqev5yhpp0', '0hKV6LdcBYambAIlHkVtAMuxCuRD6ypCVQrSGjne'

emergency = pd.read_csv(path + "응급실 정보.csv")
emergency_list = emergency.to_dict("records")

save_directory = path + "fine_tuned_bert_my"
model = AutoModelForSequenceClassification.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

@app.get("/hospital_by_module")
async def get_hospital(request: str, latitude: float, longitude: float):
    """
    병원 추천 API
    - 텍스트 (request), 현재 위치 (위도와 경도)를 입력받아 병원을 추천
    """
    # Step 1: 텍스트 요약
    summary = text2summary(request)
    
    # Step 2: 응급 등급 예측
    predicted_class, _ = predict(summary, model, tokenizer)

    # Step 3: 병원 추천
    if predicted_class <= 2:
        result = recommend_hospital3(emergency_list, latitude, longitude, c_id, c_key)
        
        # Step 4: 데이터베이스에 결과 저장
        insert_log(request, latitude, longitude, predicted_class, result.to_dict(orient="records"))
        
        # 결과를 JSON 형태로 반환
        return result.to_dict(orient="records")
        # return {
        #     "예측된 응급실 등급": predicted_class + 1,
        #     "추천된 병원 응급실": result.to_dict(orient="records")
        # }
    else:
        return {"message": "응급 상황이 아니므로 병원 추천이 필요 없습니다."}