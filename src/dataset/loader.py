import torch
from torch.utils.data import Dataset

# Dataset 클래스 정의
class EHRDataset(Dataset):
    def __init__(self, data):
        self.data = []
        self.patient_ids = []
        self.visit_lengths = []
        self.labels = []
        self.time_deltas = []
        # 각 환자별로 모든 방문을 하나의 샘플로 저장
        for patient_id, patient_data in data.items():
            self.patient_ids.append(patient_id)  # 환자 ID 저장
            self.data.append([torch.tensor(visit) for visit in patient_data['seq']])  # 방문 기록을 텐서로 저장
            self.visit_lengths.append(patient_data['visit_length'])  # 환자의 라벨 저장
            self.time_deltas.append(torch.tensor(patient_data['timedelta'], dtype=torch.long))
            self.labels.append(patient_data['label'])  # 환자의 라벨 저장

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.patient_ids[idx], self.data[idx], self.visit_lengths[idx], self.time_deltas[idx], self.labels[idx]

# 패딩 값을 인자로 받는 collate_fn 정의
def collate_fn(batch, padding_value=0):
    patient_id = [item[0] for item in batch]  # 환자 ID 추출
    visit_data = [item[1] for item in batch]  # 방문 기록만 추출 (각 환자의 방문 리스트)
    visit_lengths = [item[2] for item in batch]  # 방문 길이 추출
    time_deltas = [item[3] for item in batch]
    labels = [torch.tensor(item[4], dtype=torch.float32) for item in batch]  # 라벨 추출
    
    # 각 방문 내 시퀀스(진단 코드 리스트)를 먼저 패딩하여 동일한 길이로 만듦
    max_code_length = max(max(len(visit) for visit in visits) for visits in visit_data)
    padded_visits = [
        [torch.cat([visit, torch.full((max_code_length - len(visit),), padding_value)]) for visit in visits]
        for visits in visit_data
    ]
    
    # 환자의 방문 리스트에도 패딩을 적용하여 방문 횟수도 맞춤
    max_num_visits = max(len(visits) for visits in padded_visits)
    for i in range(len(padded_visits)):
        while len(padded_visits[i]) < max_num_visits:
            padded_visits[i].append(torch.full((max_code_length,), padding_value))
            
    padded_tds = [
        torch.cat([td, torch.full((max_num_visits - len(td),), 100000)]) for td in time_deltas
    ]
    
    # 텐서를 3D로 변환
    patient_id = torch.tensor(patient_id)
    padded_visits_seq = torch.stack([torch.stack(visits) for visits in padded_visits])
    visit_lengths = torch.tensor(visit_lengths, dtype=torch.long)
    padded_tds = torch.stack(padded_tds)
    seq_mask = (padded_tds != 100000).float()
    seq_mask_final = (padded_tds == 0).float()
    seq_mask_code = (padded_visits_seq != 0).float()
    labels = torch.stack(labels)
    return {'patient_id': patient_id, 
            'visit_seq': padded_visits_seq, 
            'length': visit_lengths, 
            'time_delta': padded_tds,
            'seq_mask': seq_mask,
            'seq_mask_final': seq_mask_final, 
            'seq_mask_code': seq_mask_code, 
            'labels': labels}
    
class EHRDatasetNew(Dataset):
    def __init__(self, data):
        self.data = []
        self.code_types = []
        self.patient_ids = []
        self.visit_lengths = []
        self.labels = []
        self.time_deltas = []
        # 각 환자별로 모든 방문을 하나의 샘플로 저장
        for patient_id, patient_data in data.items():
            self.patient_ids.append(patient_id)  # 환자 ID 저장
            self.data.append([torch.tensor(visit) for visit in patient_data['seq_idx']])  # 방문 기록을 텐서로 저장
            self.code_types.append([torch.tensor(ct) for ct in patient_data['code_types']])
            self.visit_lengths.append(patient_data['visit_length'])  # 환자의 라벨 저장
            self.time_deltas.append(torch.tensor(patient_data['timedelta'], dtype=torch.long))
            self.labels.append(patient_data['label'])  # 환자의 라벨 저장

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.patient_ids[idx], self.data[idx], self.code_types[idx],
                self.visit_lengths[idx], self.time_deltas[idx], self.labels[idx])

# 패딩 값을 인자로 받는 collate_fn 정의
def collate_fn_new(batch, padding_value=0):
    patient_id = [item[0] for item in batch]  # 환자 ID 추출
    visit_data = [item[1] for item in batch]  # 방문 기록만 추출 (각 환자의 방문 리스트)
    code_types = [item[2] for item in batch]
    visit_lengths = [item[3] for item in batch]  # 방문 길이 추출
    time_deltas = [item[4] for item in batch]
    labels = [torch.tensor(item[5], dtype=torch.float32) for item in batch]  # 라벨 추출
    
    # 각 방문 내 시퀀스(진단 코드 리스트)를 먼저 패딩하여 동일한 길이로 만듦
    max_code_length = max(max(len(visit) for visit in visits) for visits in visit_data)
    padded_visits = [
        [torch.cat([visit, torch.full((max_code_length - len(visit),), padding_value)]) for visit in visits]
        for visits in visit_data
    ]
    padded_code_types = [
        [torch.cat([visit_ct, torch.full((max_code_length - len(visit_ct),), padding_value)]) for visit_ct in visits_ct]
        for visits_ct in code_types
    ]
    
    # 환자의 방문 리스트에도 패딩을 적용하여 방문 횟수도 맞춤
    max_num_visits = max(len(visits) for visits in padded_visits)
    for i in range(len(padded_visits)):
        while len(padded_visits[i]) < max_num_visits:
            padded_visits[i].append(torch.full((max_code_length,), padding_value))
            padded_code_types[i].append(torch.full((max_code_length,), padding_value))
            
    padded_tds = [
        torch.cat([td, torch.full((max_num_visits - len(td),), 100000)]) for td in time_deltas
    ]
    
    # 텐서를 3D로 변환
    patient_id = torch.tensor(patient_id)
    padded_visits_seq = torch.stack([torch.stack(visits) for visits in padded_visits])
    visit_lengths = torch.tensor(visit_lengths, dtype=torch.long)
    visit_code_types = torch.stack([torch.stack(visits_ct) for visits_ct in padded_code_types])
    padded_tds = torch.stack(padded_tds)
    seq_mask = (padded_tds != 100000).float()
    seq_mask_final = (padded_tds == 0).float()
    seq_mask_code = (padded_visits_seq != 0).float()
    labels = torch.stack(labels)
    return {'patient_id': patient_id, 
            'visit_seq': padded_visits_seq, 
            'visit_code_types': visit_code_types,
            'length': visit_lengths, 
            'time_delta': padded_tds,
            'seq_mask': seq_mask,
            'seq_mask_final': seq_mask_final, 
            'seq_mask_code': seq_mask_code, 
            'labels': labels}
    
    
# class EHRDatasetGraph(Dataset):