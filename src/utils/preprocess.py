import os
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse


def pad_sequence(seq_diagnosis_codes, maxlen, maxcode, n_code):
    lengths = len(seq_diagnosis_codes)
    diagnosis_codes = np.zeros((maxlen, maxcode), dtype=np.int64) + n_code
    seq_mask_code = np.zeros((maxlen, maxcode), dtype=np.int8)
    seq_mask = np.zeros((maxlen), dtype=np.int8)
    seq_mask_final = np.zeros((maxlen), dtype=np.int8)
    for pid, subseq in enumerate(seq_diagnosis_codes):
        for tid, code in enumerate(subseq):
            diagnosis_codes[pid, tid] = code
            seq_mask_code[pid, tid] = 1
    seq_mask[:lengths] = 1
    seq_mask_final[lengths - 1] = 1
    return diagnosis_codes, seq_mask_code, seq_mask, seq_mask_final


def preprocess_ehr(data_dict_d, max_visits_length, max_code_len, n_code):
    new_data_dict_d = {}
    for sample_id, data in tqdm(data_dict_d.items()):
        data_dict_new = {}
        pad_seq, seq_mask_code, seq_mask, seq_mask_final = pad_sequence(data['seq'], max_visits_length, max_code_len, n_code)
        data_dict_new['seq'] = pad_seq
        data_dict_new['admitdate'] = data['admitdate']
        data_dict_new['timedelta'] = np.pad(data['timedelta'], pad_width=(0, max_visits_length - len(data['timedelta'])), mode='constant', constant_values=100000)
        data_dict_new['seq_mask'] = seq_mask
        data_dict_new['seq_mask_final'] = seq_mask_final
        data_dict_new['seq_mask_code'] = seq_mask_code
        data_dict_new['label'] = data['label']       
        new_data_dict_d[sample_id] = data_dict_new
    return new_data_dict_d
    

if __name__ == '__main__':
    # Load data
    parser = argparse.ArgumentParser(description='preprocess for mimiciv')
    parser.add_argument('--path', type=str, default='./data/', help='path to data')
    parser.add_argument('--save_path', type=str, default='./data/', help='path to save')

    args = parser.parse_args()
    
    with open(os.path.join(args.path, 'code2idx.pkl'), 'rb') as f:
        dtype_dict = pickle.load(f)
    f.close()

    with open(os.path.join(args.path, 'data_dict_preprocess_maxlen50.pkl'), 'rb') as f:
        data_dict_d = pickle.load(f)
    f.close()
    
    length_list = []
    sample_list = []
    code_length_list = []
    for sample_id, visits in tqdm(data_dict_d.items()):
        # 레이블 추가
        length_list.append(len(visits['seq']))
        sample_list.append(sample_id)
        code_length_list.append(max([len(seq)for seq in visits['seq']]))
        
    n_code = len(dtype_dict) + 1
    max_visits_length = max(length_list)
    max_code_len = max(code_length_list)
    print('n_code:', n_code)
    print('max_visit:', max_visits_length)
    print('max_code_len:', max_code_len)
    
    date_str = datetime.now().strftime("%Y%m%d")
    new_data_dict_d = preprocess_ehr(data_dict_d, max_visits_length, max_code_len, n_code)
    with open(os.path.join(args.save_path, f'preprocessed_nd_{date_str}.pkl'), 'wb') as f:
        pickle.dump(new_data_dict_d, f)
    f.close()
        
    