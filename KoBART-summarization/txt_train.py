import argparse
import numpy as np
import pandas as pd
from loguru import logger
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dataset import KobartSummaryModule
from model import KoBARTConditionalGeneration
from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser(description='KoBART Summarization')

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/test.tsv',
                            help='train file')
        parser.add_argument('--test_file',
                            type=str,
                            default='data/test.tsv',
                            help='test file')
        parser.add_argument('--batch_size',
                            type=int,
                            default=28,
                            help='')
        parser.add_argument('--checkpoint',
                            type=str,
                            default='checkpoint',
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        parser.add_argument('--max_epochs',
                            type=int,
                            default=10,
                            help='train epochs')
        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')
        parser.add_argument('--accelerator',
                            type=str,
                            default='gpu',
                            choices=['gpu', 'cpu'],
                            help='select accelerator')
        parser.add_argument('--num_gpus',
                            type=int,
                            default=1,
                            help='number of gpus')
        parser.add_argument('--gradient_clip_val',
                            type=float,
                            default=1.0,
                            help='gradient_clipping')
        return parser

if __name__ == '__main__':
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummaryModule.add_model_specific_args(parser)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    args = parser.parse_args()
    logger.info(args)
    
    dm = KobartSummaryModule(args.train_file,
                        args.test_file,
                        tokenizer,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        num_workers=args.num_workers)
    dm.setup("fit")
    
    model = KoBARTConditionalGeneration(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=args.checkpoint,
                                          filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                          verbose=True,
                                          save_last=True,
                                          mode='min',
                                          save_top_k=3)
    
    wandb_logger = WandbLogger(project="KoBART-summ")
    # add
    model.eval()

    def generate_summary(input_text):
        # tokenize input
        input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=args.max_len, truncation=True)

        # summ with model
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

        # decoding
        generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print("Generated Summary:", generated_summary)

    # test data
    test_input_text = """눈보라 ＊"동아일보"(1929년 09월 21일~10.1)에동업자란 제목으로 발표한 작품.단편집"태형"(대조사, 1946)에서 눈보라로 제목을 고쳐 수록함. 조선은 빽빽한 곳이었습니다. 어떤 사립학교에서 교사 노릇을 하던 홍 선생은 그 학교가 총무부 지정 학 교가 되는 바람에 쫓겨 나왔습니다. 제아무리 실력이 있다 할지라도 교원 면허증이라 하는 종잇조각이 없으면 교사질도 하지 말라 합니다. 그러나 이 제 다시 산술이며 지리 역사를 복습해가지고 교원검정시험을 치를 용기는 없었습니다. 일본 어떤 사립중학과 대학을 우유배달과 신문배달을 하면서 공부를 하느 라고 얼마나 애를 썼던가. 겨울, 주먹을 쥐면 손이 모두 터져서 손등에서 피가 줄줄 흐르는 그런 손으로 필기를 하여 공부한 자기가 아니었던가. 주 린 배를 움켜쥐고 학교 시간 전에 신문배달을 끝내려고 눈앞이 보이지 않는 것을 씩씩거리며 뛰어다니던 그 쓰라림은 얼마나 하였던가. 그리고 시간을 경제하느라고 우유 구루마를 끌고 책을 보며 다니다가 돌이라도 차고 넘어 졌다가 다시 일어날 때에 벙글 웃던 그 웃음은 얼마나 상쾌하였던가. 이것 도 장래의 나의 일화의 한 페이지가 되려니. 아아, 생각지 않으리라. 그 모든 고생이며 애도 오늘날의 영광을 기대하는 바람이 있었기에 무서운 참을성으로 참고 지내지 안했나. 그러나, 그 애, 그 노력도 모두 물거품으로 돌아가버렸습니다. 7년 동안의 끔찍이 쓴 노력도 조선 돌아와서 소학 교사 하나를 해먹을 수가 없었습니 다. 7년 동안을 머릿속에 잡아넣은 지식은 헛되이 썩어날 뿐 활용해볼 길이 없었습니다."""
    trainer = L.Trainer(max_epochs=args.max_epochs,
                        accelerator=args.accelerator,
                        devices=args.num_gpus,
                        gradient_clip_val=args.gradient_clip_val,
                        callbacks=[checkpoint_callback],
                        logger=wandb_logger
                        )
    
    # 에폭 당 프레딕션 실행
    for epoch in range(args.max_epochs):
        trainer.fit(model, dm)
        generate_summary(test_input_text)
