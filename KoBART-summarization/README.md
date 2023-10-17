# FocusRead-summarization

## Load KoBART summarization
- seujung의 KoBART-summarization
```
!git clone https://github.com/seujung/KoBART-summarization
```

## Requirements
```
torch==2.0.1
transformers==4.32.1
tokenizers==0.13.3
lightning==2.0.8
streamlit==1.26.0
wandb==0.15.9
```

## Data
- AI Hub '요약문 및 레포트 생성 데이터' 문학 데이터 활용
  https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582
- 생성형 AI 서비스들을 이용하여 생성 후 검수한 자체 요약문 데이터
- tsv 형태로 데이터를 변환
- Data 구조
    - Train Data : 9,905
    - Test Data : 1,276


## How to Train
- KoBART summarization fine-tuning
```bash
pip install -r requirements.txt

!CUDA_VISIBLE_DEVICES=0 python /home/alpaco/hw/KoBART-summarization/train.py --gradient_clip_val 1.0 \
                --train_file '/home/alpaco/hw/KoBART-summarization/data/train.tsv' \
                --test_file '/home/alpaco/hw/KoBART-summarization/data/test.tsv' \
                --max_epochs 201 \
                --checkpoint checkpoint \
                --accelerator gpu \
                --num_gpus 1 \
                --lr 0.00005 \
                --batch_size 26 \
                --num_workers 64
```



## 모델 저장
   - kobart_summary 디렉토리에 모델 저장
   - hparams: logs 하위 디렉토리에서 사용할 모델의 버전 골라 hparams.yaml set
   - model_binary: logs 하위 디렉토리에서 사용할 체크포인트 골라 *.ckpt set
```
!python get_model_binary.py --hparams /home/alpaco/hw/KoBART-summarization/lightning_logs/version_60/hparams.yaml --model_binary checkpoint/last-v6.ckpt
```

## Inference
```
import torch
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained('/home/alpaco/hw/KoBART-summarization/kobart_summary')
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

text = """
운수 좋은날 현진건 새침하게 흐린 품이 눈이 올 듯하더니 눈은 아니 오고 얼다가 만 비가 추 적추적 내리는 날이었다. 이날이야말로 동소문 안에서 인력거꾼 노릇을 하는 김첨지에게는 오래간만 에도 닥친 운수 좋은 날이었다. 문안에 거기도 문밖은 아니지만 들어간답 시는 앞집 마마님을 전찻길까지 모셔다 드린 것을 비롯으로 행여나 손님이 있을까 하고 정류장에서 어정어정하며 내리는 사람 하나하나에게 거의 비는 듯한 눈결을 보내고 있다가 마침내 교원인 듯한 양복쟁이를 동광학교(東光 學校)까지 태워다 주기로 되었다. 첫 번에 삼십전, 둘째 번에 오십전 - 아침 댓바람에 그리 흉치 않은 일이 었다. 그야말로 재수가 옴불어서 근 열흘 동안 돈을 보지도 못한 김첨지는 십 전짜리 백동화 서 푼, 또는 다섯 푼이 찰깍 하고 손바닥에 떨어질 제 거의 눈물을 흘릴 만큼 기뻤었다. 더구나 이날 이때에 이 팔십 전이라는 돈이 그 에게 얼마나 유용한지 몰랐다. 컬컬한 목에 모주 한 잔도 적실 수 있거니와 그보다도 앓는 아내에게 설렁탕 한 그릇도 사다 줄 수 있음이다. 그의 아내가 기침으로 쿨룩거리기는 벌써 달포가 넘었다. 조밥도 굶기를 먹다시피 하는 형편이니 물론 약 한 첩 써본 일이 없다. 구태여 쓰려면 못 쓸 바도 아니로되 그는 병이란 놈에게 약을 주어 보내면 재미를 붙여서 자 꾸 온다는 자기의 신조(信)에 어디까지 충실하였다. 따라서 의사에게 보 인 적이 없으니 무슨 병인지는 알 수 없으되 반듯이 누워 가지고 일어나기 는 새로 모로도 못 눕는 걸 보면 중증은 중증인 듯. 병이 이대도록 심해지 기는 열흘전에 조밥을 먹고 체한 때문이다. 인력거꾼 김첨지가 오래간만에 돈 을 얻어서 좁쌀 한 뇌와 십 전짜리 나무 한 단을 사다 주었더니 김첨지의 말에 의지하면 그 오라질 년이 천방지축으로 냄비에 대고 끓였다.
"""

if text:
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=4)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)
```

## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)

