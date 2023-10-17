import argparse
from loguru import logger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import PreTrainedTokenizerFast
from dataset import KobartSummaryModule
from model import KoBARTConditionalGeneration

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
        parser.add_argument('--generate_summaries',
                            action='store_true',
                            help='Generate summaries using the trained model')
        parser.add_argument('--reference_file',
                            type=str,
                            default='data/reference_summaries.tsv',
                            help='Reference summaries file for evaluation')
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
                                          save_top_k=101)

    wandb_logger = WandbLogger(project="KoBART-summ")

    if args.generate_summaries:
        # 요약문 생성 코드
        gen_summaries = []  # 생성된 요약문 저장
        ref_summaries = []  # 참조 요약문 저장

        # 모델을 사용하여 요약문 생성 및 ref_summaries를 채워넣는 코드 작성

        # Rouge 점수 계산 및 출력

    else:
        # 훈련 코드
        trainer = pl.Trainer(max_epochs=args.max_epochs,
                            accelerator=args.accelerator,
                            gradient_clip_val=args.gradient_clip_val,
                            callbacks=[checkpoint_callback],
                            logger=wandb_logger
                            )
        trainer.fit(model, dm)
