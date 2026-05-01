# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : main.py
# Time       ：6/8/2024 10:15 pm
# Author     ：XXXXXXX
# version    ：python 
# Description：disease
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import time
import torch
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, SampleEHRDataset, eICUDataset, OMOPDataset
from utils import split_by_patient, load_pickle, save_pickle, set_random_seed, generate_rare_disease, ana_rare_disease, get_tokenizers, generate_rare_patient
from data import drug_recommendation_mimic3_fn_wc, drug_recommendation_mimic4_fn_wc, drug_recommendation_eicu_fn_wc, re_generate_dataset, convert_dataset, load_custom_dataset
from data import drug_recommendation_omop_fn_wc, drug_recommendation_pic_fn_wc, drug_recommendation_omix_fn_wc

from loader import get_dataloader, raremed_mask_nsp_collate_fn, collate_fn_dict
from config import config
from pretrain import run_pretrain_pcf, run_pretrain_drl, aug_inference, load_pretrain_pcf, load_pretrain_plm, load_pretrain_drl, evaluate_pcf
from pic_parase import PICDataset
from omix_parse import OMIXDataset

set_random_seed(config['SEED'])



# @profile(precision=4, stream=open("memory_profiler.log", "w+"))
def run_single_config(pretrain=False, tuning=False,  exp_num=''):
    # GPU 占用
    a = torch.ones((10000, 50000)).to('cuda:' + config['GPU'])
    print("GPU Memory Usage", torch.cuda.memory_allocated('cuda:' + config['GPU']) / 1024 / 1024 / 1024, "GB")
    
    # load datasets
    # STEP 1: load data
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    root_to = os.path.join(os.path.dirname(_src_dir), 'data', config['TASK'], config['DATASET'], 'processed') + '/'
    if not os.path.exists(root_to + 'datasets_pre_stand.pkl'):
        if config['DATASET'] == 'MIII':
            base_dataset = MIMIC3Dataset(
                root="/home/xxx/HyperHealth/data/physionet.org/files/mimiciii/1.4",
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 这里graphcare的ATC-level是3；和我们在data阶段有差别
                dev=False,
                refresh_cache=False,
            )
            base_dataset.stat()
            # set task
            sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn_wc) # 按照task_fn进行处理
            sample_dataset.stat()
        elif config['DATASET'] == 'eICU':
            base_dataset = eICUDataset(
                root="/home/xxxx/HyperHealth/data/physionet.org/files/eicu-crd/2.0",
                tables=["diagnosis", "medication", "physicalExam", "treatment", "admissionDx"],
                dev=False,
                refresh_cache=False,
            )
            base_dataset.stat()
            # set task
            sample_dataset = base_dataset.set_task(drug_recommendation_eicu_fn_wc) # 按照task_fn进行处理
            sample_dataset.stat()
        elif config['DATASET'] == 'MIV':
            base_dataset = MIMIC4Dataset(
                root="/home/xxxx/HyperHealth/data/physionet.org/files/mimiciv/2.0/hosp",
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 4
                dev=False,
                refresh_cache=False,
            )
            base_dataset.stat()
            # set task
            sample_dataset = base_dataset.set_task(drug_recommendation_mimic4_fn_wc)
            sample_dataset.stat()
        elif config['DATASET'] == 'CUSTOM':
            RECORDS_PATH = os.path.join(config['DRUGDOCTOR_DIR'], 'data', 'output', 'records_final.pkl')
            VOC_PATH     = os.path.join(config['DRUGDOCTOR_DIR'], 'data', 'output', 'voc_final.pkl')
            samples = load_custom_dataset(RECORDS_PATH, VOC_PATH)
            save_pickle(samples, root_to + 'datasets_pre_stand.pkl')
            print("Custom dataset saved! {} samples total".format(len(samples)))
            print("Please run again!")
            return
        else:
            print("No such dataset!")
            return


        samples = sample_dataset.samples
        # disease_group = generate_rare_disease(samples, config['RARE_THRES'], root_to, task=config['TASK'])
        # generate_rare_patient(samples, disease_group['group_disease'], root_to)

        save_pickle(samples, root_to + 'datasets_pre_stand.pkl')

        print("initial dataset done!")
        print("Please run again!")
        return
    else:
        start = time.time()
        samples = load_pickle(root_to + 'datasets_pre_stand.pkl')
        # load_code_convert(dataset=config['DATASET'], samples=samples) # 如果这里要弄，必须要重新生成rare disease
        disease_group = generate_rare_disease(samples, config['RARE_THRES'], root_to, task=config['TASK'])
        generate_rare_patient(samples, disease_group['group_disease'], root_to)

        disease = load_pickle(root_to + 'rare.pkl')
        patient = load_pickle(root_to + 'rare_patient.pkl')

        rare_disease = disease['rare_disease']# rare_disease， filter_drl_items
        y_grouped = disease['group_disease'].values()
        p_grouped = patient # patient

        samples = ana_rare_disease(samples, rare_disease)


        end = time.time()
        print("Load data done! Cost time {} s".format(end-start))

        if config['DEV']:
            print("DEV train mode: 1000 patient")
            samples = samples[:3000]
            sample_dataset = SampleEHRDataset( # 这个贼耗时
                samples,
                dataset_name=config['DATASET'],
                task_name=config['TASK'],
            )
            train_dataset, val_dataset, test_dataset = split_by_patient(
                sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                train_ratio=1.0,  # Train test split
                seed=528
            )
            del samples

            endt = time.time()
            print('Dataset done!, Cost {} s'.format(endt - end))
        else:
            sample_dataset = convert_dataset(samples, dataset_name=config['DATASET'], task_name=config['TASK']) # 需要查看是否需要load_code_convert
            train_dataset, _, test_dataset = split_by_patient(
                sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                train_ratio=1.0,  # Train test split
                warm_cold=False,
                seed=config['SEED']
            ) # 这样似乎更快，固定随机种子的时候是一样的；

            tokenizer = get_tokenizers(sample_dataset, special_tokens=[])['conditions']  # 和label一致
            y_grouped = tokenizer.batch_encode_2d(y_grouped, padding=False, truncation=False)

            del samples

            endt = time.time()
            print('Train Dataset done!, Cost {} s'.format(endt - end))


    # STEP 2: load dataloader
    if config['PCF_MODEL'] == 'RAREMeds': # 不要用了，使用cls embedding问题很大。
        collate_fn = raremed_mask_nsp_collate_fn
    else:
        collate_fn = collate_fn_dict
    train_dataloader = get_dataloader(train_dataset, batch_size=config['PCF_CONFIG']['BATCH'], shuffle=True, drop_last=True, collate_fn=collate_fn) # 得明确一下其是否是经过standarlized
    # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=config['PCF_CONFIG']['BATCH'], shuffle=True, drop_last=True, collate_fn=collate_fn) # config['BATCH']
    load_dataloader = time.time()
    print('Dataloader done!, Cost {} s'.format(load_dataloader - endt))
    del a
    #
    # test_dataloader = iter(test_dataloader)
    # data = next(test_dataloader)
    # print(data)
    #
    # cache clear
    torch.cuda.empty_cache()
    gc.collect()

    if pretrain:
        # pretrain model
        print("===============Pretrain PCF!===============")
        start = time.time()
        # print(sample_dataset.get_all_tokens(key='conditions'))
        pcf_model = run_pretrain_pcf(sample_dataset, train_dataloader, test_dataloader, config,
                                     y_grouped=y_grouped, p_grouped=p_grouped, special_input=train_dataset, exp_num=exp_num)
        end = time.time()
        print("Pretrain PCF done! Cost {} s".format(end - start))

        # pretrain drl
        print("===============Pretrain DRL!===============")
        # print(sample_dataset.get_all_tokens(key='conditions'))
        start = time.time()
        # joint时候要注释掉
        pcf_model = load_pretrain_pcf(sample_dataset, config, special_input=train_dataset, exp_num=exp_num) # 这里为啥不能同时sampledataset 一旦initial两次？，rarmed的时候，这里会出问题；直接注释掉上面的
        plm_model, drl_model = run_pretrain_drl(disease, sample_dataset, train_dataset, test_dataloader, pcf_model, config, exp_num=exp_num)
        end = time.time()
        print("Pretrain DRL done! Cost {} s".format(end - start))
    else:
        pcf_model = load_pretrain_pcf(sample_dataset, config, special_input=train_dataset, exp_num=exp_num) # 因为只训一次
        plm_model = load_pretrain_plm(config, exp_num=exp_num)
        drl_model = load_pretrain_drl(pcf_model, plm_model, config, exp_num=exp_num)
        print("Load Pretrain Done!")

    # aug inference
    print("===============Aug Inference!===============")
    print("=====原始:")
    evaluate_pcf(pcf_model, test_dataloader, config, y_grouped, p_grouped, exp_num='0')
    # 
    # import logging # disable掉部分logger
    # logging.shutdown()


    print("=====修正:")
    model=aug_inference(sample_dataset, pcf_model, plm_model, drl_model, train_dataloader, test_dataloader, config, y_grouped,  p_grouped, special_input=train_dataset, tuning=tuning, exp_num=exp_num)
    print("===============Aug Inference Done!===============")





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--tuning',   action='store_true', default=False)
    parser.add_argument('--exp_num',  type=str, default='0')
    parser.add_argument('--gpu',      type=str, default=None)
    parser.add_argument('--thres',    type=float, default=None,
                        help='prediction threshold (overrides config THRES); '
                             'appended to exp_num so results land in a separate dir')
    args = parser.parse_args()

    if args.gpu is not None:
        config['GPU'] = args.gpu
    if args.thres is not None:
        config['THRES'] = args.thres
        args.exp_num = args.exp_num + '_t{}'.format(str(args.thres).replace('.', ''))

    config['JOINT'] = True
    print("Hi, This is UDC Health!")
    print("You are running on", config['DATASET'], "dataset! THRES={}".format(config['THRES']))
    run_single_config(pretrain=args.pretrain, tuning=args.tuning, exp_num=args.exp_num)
    print("All Done!")
