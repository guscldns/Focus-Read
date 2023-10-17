# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pymysql
from werkzeug.utils import secure_filename
import os
from PIL import Image
from google.cloud import vision
import numpy as np
import fitz
import zipfile
import cv2
import io
import torch
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration, BartForCausalLM
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from keybert import KeyBERT
from kiwipiepy import Kiwi
from transformers import BertModel
from collections import Counter

# 모델과 토크나이저 로드 (오래걸려서 미리 로드)
summary_model = BartForConditionalGeneration.from_pretrained('/home/alpaco/Web_project/kobart_summary')
summary_tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    
# 파일 저장할 폴더 생성
if not os.path.exists("./upload_file"):
    os.mkdir("./upload_file")
if not os.path.exists("./image_file"):
    os.mkdir("./image_file")

# 플라스크 객체 생성
app = Flask(__name__)

# 데이터베이스 연결 객체 생성
db_conn = pymysql.connect(  
                        host = 'localhost',
                        port = 3306,
                        user = 'alpaco',
                        passwd ='1234',
                        db = 'webproject',
                        charset='utf8')
                        
print(db_conn)

#cursor = db_conn.cursor()

# OCR 함수 부분 ==================================================================================================================
def detect_paragraphs(image_path):
    from google.cloud import vision
    # API키 가져오기
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/alpaco/Web_project/api-project-397750607032-5ddc025931cd.json"
    # API 가져오기
    client = vision.ImageAnnotatorClient()
    # 주석을 추가할 이미지 파일 이름
    file_name = os.path.abspath(image_path)
    # 이미지 로드
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    # 이미지 OCR
    response = client.document_text_detection(image=image)
    # 이미지 OCR 텍스트 전문
    full_text = response.full_text_annotation.text
    # 이미지 OCR 후 결과 (bbox, word 등)
    pages = response.full_text_annotation.pages
    # 텍스트 주석
    # 참고 : https://cloud.google.com/dotnet/docs/reference/Google.Cloud.Vision.V1/latest/Google.Cloud.Vision.V1.TextAnnotation.Types.DetectedBreak.Types.BreakType
    # 참고 : https://googleapis.github.io/googleapis/java/grpc-google-cloud-vision-v1/0.1.5/apidocs/com/google/cloud/vision/v1/TextAnnotation.DetectedBreak.BreakType.html
    breaks = vision.TextAnnotation.DetectedBreak.BreakType
    paragraphs = []
    lines = []
    for page in pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                para = "★문단시작★"
                line = ""
                for word in paragraph.words:
                    for symbol in word.symbols:
                        line += symbol.text
                        # breaks.SPACE : 공백
                        if symbol.property.detected_break.type == breaks.SPACE:
                            line += ' '
                        # breaks.EOL_SURE_SPACE : 줄 바꿈
                        if symbol.property.detected_break.type == breaks.EOL_SURE_SPACE:
                            line += ' '
                            lines.append(line)
                            para += line
                            line = ''
                        # breaks.LINE_BREAK : 단락을 끝내는 줄바꿈
                        if symbol.property.detected_break.type == breaks.LINE_BREAK:
                            lines.append(line)
                            para += line
                            line = ''
                paragraphs.append(para)
    return full_text, paragraphs
#  ================================================================================================================== OCR 함수 끝

text_full = ''

# OCR, 요약 파이프라인 infer 함수 ==================================================================================================================
def infer(file_path):
    def pdf_to_png(files):
        path = f"{files}"
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            img = page.get_pixmap()
            img.save(f"./image_file/{i}.png")
    # pdf 변환
    pdf_to_png(file_path)

    # 함수 시험 # 숫자 count로 추가작업해야함★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    image_path = "./image_file/0.png"
    
    global text_full
    
    # PDF 1페이지 OCR 단계 -> 전문, 문단스플릿 리스트 리턴
    text_full, paragraphs = detect_paragraphs(image_path)
    
    print(text_full)
    print(paragraphs)
    
    

    # summary 함수 정의
    def summaryfn(paragraphs_arg):  
        # 문단 리스트를 512 토큰으로 나누기
        # 입력문장 리스트 : paragraphs
        # 모델 입력 데이터 생성
        input_data = []
       
        # 현재까지의 토큰 수 초기화
        current_token_count = 0
        
        print("paragraphs",paragraphs) 
        
        for paragraph in paragraphs:
            # '★문단시작★'를 제거하고 문단을 토큰화
            paragraph_tokens = summary_tokenizer(paragraph.replace('★문단시작★', ''), return_tensors='pt', add_special_tokens=True).input_ids
            # 현재 문단을 추가해도 512 토큰을 넘지 않으면 추가
            if current_token_count + len(paragraph_tokens[0]) <= 512:
                if not input_data:
                    input_data.append(paragraph_tokens)
                    current_token_count += len(paragraph_tokens[0])
                else:
                    # 이미 문단이 추가되어 있는 경우 이어서 추가
                    input_data[-1] = torch.cat((input_data[-1], paragraph_tokens), dim=-1)
                    current_token_count += len(paragraph_tokens[0])
            else:
                # 256 토큰을 넘어가면 새로운 입력으로 시작
                input_data.append(paragraph_tokens)
                current_token_count = len(paragraph_tokens[0])
                
        print("input_data",input_data) 
         # 요약 데이터 생성
        summary =[]
        for inp in input_data:
            output = summary_model.generate(inp, eos_token_id=1, max_length=512, num_beams=4)
            t_output = summary_tokenizer.decode(output[0], skip_special_tokens=True)
            summary.append(t_output)
        return summary    
                
    # 요약단계
    summary_result = summaryfn(paragraphs)
    
    return summary_result
#  ================================================================================================================== OCR, 요약 파이프라인 infer 함수 끝


# Flask 주소 매핑 함수들 부분 ==================================================================================================================

# url 주소에 따른 실행
@app.route("/")
def main():
    # 어떤 화면(index.html)을 보여줘라
    return render_template("main.html",flag=True)

keywords = '' 
@app.route("/summary", methods=["POST"])
def summary_post():
    user_sum = request.form['user_sum']
    
    # 출력 및 시간 test
    model = BertModel.from_pretrained('skt/kobert-base-v1')
    kw_model = KeyBERT(model) # 로드까지 26초
   
    def fulltext_keywords(full_text):
        kiwi = Kiwi()
        kiwi.analyze(full_text)
        global keywords
        keywords = kw_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=10)
        # 명사 키워드 추출 함수 정의
        def extract_nouns(keyword_list):
            noun_keywords = []
            keywordscore = []
            for keyword, score in keywords:
                # 형태소 분석 수행
                tokens = kiwi.analyze(keyword)
                # print(tokens)
                for tk in tokens[0][0]:
                    # print(tk)
                    if tk.tag == "NNG":
                        if tk.form not in noun_keywords:
                            noun_keywords.append(tk.form)
                            keywordscore.append((tk.form,score))
            return noun_keywords, keywordscore
        
        noun_keywords, keywordscore = extract_nouns(keywords)
        
        return noun_keywords, keywordscore

    noun_keywords, keywordscore = fulltext_keywords(text_full)
    
    keywordscores=[keywordscore]
    noun_keyword=[noun_keywords]
    
    def keyword_sum(noun_keyword, n):
        
        # 키워드 리스트 하나로 합치기
        word_list = sum(noun_keyword,[])
        # 단어 빈도수 계산
        word_counts = Counter(word_list)
        # 가장 많이 중복된 단어 5개 선택
        top_n_words = [word for word, count in word_counts.most_common(n)]
        return top_n_words
    
    top_n_words =  keyword_sum(noun_keyword, 5)
    
    final_keyword = []
    
    for i in keywords:
        if i[0] in top_n_words:
            final_keyword.append(i)
    
    
    return render_template("result.html", result_summary = result_summary_list[0] , user_sum_context= user_sum, keywords_result = keywordscores[0])

result_summary_list=''

@app.route("/list", methods=["POST"])
def test():
    print("우와")
    
    file = request.files['fileInput']
    
    image_path= './image_src'
    filename = secure_filename(file.filename)
    
    if filename == '':
        return render_template("main.html", flag = False)
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    file.save(os.path.join(image_path, filename))
    global result_summary_list
    result_summary_list = infer(os.path.join(image_path, filename)) # 요약 결과
    
    
    
    return render_template("test_page.html", words=text_full.split(' '))


#  ================================================================================================================== Flask 주소 매핑 함수들 부분 끝

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = True)