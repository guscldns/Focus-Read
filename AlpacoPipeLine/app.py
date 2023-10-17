# piapp.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
from IPython.display import clear_output
import pymysql
import pandas as pd

app = Flask(__name__)

@app.after_request
def set_response_headers(response):
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

# DB 연동
db_conn = pymysql.connect(
    host='112.175.29.231',
    port=3302,
    user='alpaco',
    passwd='1234',
    db='webproject',
    charset='utf8'
)
print(db_conn)
# cursor = db_conn.cursor()
# query = "select * from vocab"
# cursor.execute(query)

chatbot_data = pd.read_csv("word_chatbot.csv")
chat_dic = {}
row = 0
for rule in chatbot_data['rule']:
    chat_dic[row] = rule.split('|')
    row += 1

def process_chat(user_request):
    responses = []
    response2s = []
    for k, v in chat_dic.items():
        if user_request in v:
            responses.append(chatbot_data['response'][k])
            response2s.append(chatbot_data['response2'][k])
    if responses:
        response_str = f"'{user_request}'의 뜻과 예문은 다음과 같습니다."
        for i, (res, res2) in enumerate(zip(responses, response2s), 1):
            response_str += f"\n{i}.{res}\n{res2}"
        return response_str
    else:
        no_response_msg = '어휘 사전에 없는 단어입니다.'
        return no_response_msg

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/chatbotsql')
def chatbotsql():
    cursor = db_conn.cursor()
    query = "select * from vocab"
    cursor.execute(query)
    result = []
    for _, row in chatbot_data.iterrows():
     temp = {'request': row['request'], 'rule': row['rule'], 'response': row['response'], 'response2': row['response2']}
     result.append(temp)
    db_conn.close()
    return render_template('chatbotsql.html', result_table=result)
    
@app.route('/login_check', methods=['POST'])
def login_check():
    id = request.form['id']
    pwd = request.form['pwd']
    cursor = db_conn.cursor()
    query = f"select * from user_info where user_id like '{id}' and user_pwd like '{pwd}'"
     
    cursor.execute(query)

    if len(list(cursor)) > 0:
        return redirect(url_for('chat'))  # 로그인 성공 후 챗봇 페이지로 리다이렉트

    else:
        return render_template('main.html')

@app.route('/signup')
def signup():
       return render_template('signup.html')

@app.route('/signup_add', methods=['POST'])
def signup_add():
    id = request.form['id']
    pwd = request.form['pwd']
    name = request.form['name']
    email = request.form['email']
    if id != '' and pwd != '' and name != '' and email != '':
        cursor = db_conn.cursor()
        query_check_duplicate = f"SELECT COUNT(*) FROM user_info WHERE user_id='{id}'"
        cursor.execute(query_check_duplicate)
        result_count = cursor.fetchone()[0]
        if result_count > 0:
            return render_template('error.html', message='중복된 아이디입니다.')
        query_insert_user = f"INSERT INTO user_info (user_id, user_name,user_pwd, user_email) VALUES ('{id}', '{name}', '{pwd}', '{email}')"
        cursor.execute(query_insert_user)
        db_conn.commit()
        return render_template('main.html')
    else:
        return render_template('error.html')
    
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        req = request.form.get('request')
        if req is None:
            return jsonify(result='Invalid request')
        processed_output = process_chat(req)
        clear_output(wait=True)
        return jsonify(result=processed_output)
    else:
        return render_template('chatbot.html')
if __name__ == "__main__":
    app.run(debug=True)