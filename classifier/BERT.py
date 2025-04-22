import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
import os,time
import numpy as np
from sklearn.metrics import classification_report, matthews_corrcoef  
import pandas as pd
from transformers import get_linear_schedule_with_warmup
import random
from nltk.corpus import stopwords
import string
import json
from transformers import RobertaTokenizer, RobertaModel
from sklearn.ensemble import RandomForestClassifier 
from BertAdam import BertAdam

SEED = 42  # 您可以选择任何常数值作为种子  
# Python 的内置随机生成器  
random.seed(SEED)  
# NumPy 随机生成器  
np.random.seed(SEED)  
# PyTorch 随机生成器  
torch.manual_seed(SEED)  
# 如果使用 GPU  
if torch.cuda.is_available():  
    torch.cuda.manual_seed(SEED)  
    torch.cuda.manual_seed_all(SEED)  # 如果使用多个 GPU  
# CuDNN 的确定性设置，这可能会影响性能  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False

start_time = time.time()
# 打印CUDA信息
print(torch.cuda.is_available())  # 如果返回 True，则表示 CUDA 可用
print(torch.__version__)  # 查看 PyTorch 版本
print(torch.version.cuda)


functional_keys = [
    "功能需求-初始化", 
    "功能需求-数据采集", 
    "功能需求-控制输出", 
    "功能需求-遥测", 
    "功能需求-控制计算-姿态控制",
    "功能需求-控制计算-姿态确定", 
    "功能需求-控制计算-模式管理", 
    "功能需求-控制计算-轨道计算", 
    "功能需求-遥控", 
    "功能需求-控制计算-运行时保障-故障诊断和处理", 
    "功能需求-控制计算-运行时保障-数据有效性判断", 
    # "非功能需求-可靠性需求", 
    # "非功能需求-安全性需求", 
    # "非功能需求-性能需求-精度需求", 
    # "非功能需求-性能需求-空间需求", 
    # "非功能需求-性能需求-时间需求", 
    # "非功能需求-可复用性需求", 
    # "非功能需求-可维护性需求"
]

def create_hierarchical_prompt(text, label, feature, functional_keys):  
    # prompt = f"给定航天嵌入式软件需求分类目录结构及类别标签如下： {', '.join(functional_keys[:-1])}。根据目录结构及类别标签对以下需求描述进行分类，该需求可能属于多个需求类别。需求描述：{text}"  
    # prompt = text
    indices_not_one = [i for i, x in enumerate(label) if x == 1]
    filtered_labels = [functional_keys[index] for index in indices_not_one] 
    prompt = text + "\n需求类别标签：" + ", ".join(filtered_labels) 
    prompt = text + feature
    return prompt  

def chinese_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('chinese'))
    filtered_text = ' '.join([word for word in text.split() if word not in stop_words])
    return filtered_text

### 自定义截断策略
class TextEncoder:  
    def __init__(self, tokenizer, max_len, head_len=128, tail_len=None):  
        self.tokenizer = tokenizer  
        self.max_len = max_len  
        self.head_len = head_len  
        self.tail_len = tail_len if tail_len else max_len - head_len - 2  # Exclude [CLS] and [SEP]  

    def encode_text(self, text, test, label, feature, truncation_strategy='head_tail'):  
        # Tokenize the input text  
        if test == False:
            text = create_hierarchical_prompt(text, label, feature, functional_keys)
        tokens = self.tokenizer.tokenize(text)  
        if len(tokens) > self.max_len - 2:  # Adjust for special tokens  
            if truncation_strategy == 'head_only':  
                truncated_tokens = tokens[:self.max_len-2]  
            elif truncation_strategy == 'tail_only':  
                truncated_tokens = tokens[-(self.max_len-2):]  
            elif truncation_strategy == 'head_tail':  
                truncated_tokens = tokens[:self.head_len] + tokens[-self.tail_len:]
            elif truncation_strategy == 'feature':
                truncated_tokens = self.tokenizer.tokenize(feature)
            else:  
                raise ValueError(f"Unknown truncation strategy: {truncation_strategy}")  
        else:  
            truncated_tokens = tokens  

        truncated_tokens = ['[CLS]'] + truncated_tokens + ['[SEP]']  
        input_ids = self.tokenizer.convert_tokens_to_ids(truncated_tokens)  
        attention_mask = [1] * len(input_ids)  

        # Pad sequences  
        padding_length = self.max_len - len(input_ids)  
        input_ids.extend([0] * padding_length)  
        attention_mask.extend([0] * padding_length)  

        # 返回 PyTorch 张量  
        return {  
            'input_ids': torch.tensor(input_ids),  # 转换为张量  
            'attention_mask': torch.tensor(attention_mask)  # 转换为张量  
        }  

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, test = False, strategy='feature'):
        self.texts = texts
        self.labels = labels
        # self.features = features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.strategy = strategy
        self.test = test

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        # feature = self.features[index]
        ## 默认截断策略
        if len(text) > 512:
            encoding = self.tokenizer.encode_plus(
                # chinese_process(feature),
                chinese_process(text),
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        else:
            ## 自定义截断策略
            encoding = self.tokenizer.encode_plus(
                # chinese_process(text+feature),
                chinese_process(text),
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        ## 自定义截断策略
        # encoder = TextEncoder(self.tokenizer, max_len=512)
        # encoding = encoder.encode_text(text, self.test, label, feature, truncation_strategy=self.strategy)   
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)  
        }
    
# 加载数据函数
def load_data(file_path,fold,train_or_test):
    df = pd.read_csv(file_path)
    # with open(f'splits/fold_{fold}_{"train" if train_or_test=="train" else "test"}_req_with_refined_core_classes.json','r',encoding='utf-8') as file:
    #     req_with_refined_core_classes=json.load(file)
    # texts = df['Requirements Description'].values  # 第一列为 "Requirements Description"
    # texts_new = []
    # for item in texts:
    #     for item2 in req_with_refined_core_classes:
    #         if item==item2["requirement"]:
    #             text = "Requirement:"+item2["requirement"]+"\n"+"Core classes:"+item2["refined_core_classes"]
    #             texts_new.append(text)
    texts = df['Requirements Description'].values  # 第一列为 "Requirements Description"
    labels = df.iloc[:, 1:len(functional_keys)+1].values  # 从第二列到第十九列为标签（12个数字）
    # features = df['feature'].values  # 从第二十列到最后一列为特征（18个数字）
    weights = []
    for i in range(1, len(functional_keys)+1):
        weights = np.ones(len(functional_keys))
    return texts, labels, weights

# 设置训练函数
def train_model(model, data_loader, optimizer, scheduler, device, loss_fn):
    model = model.train()
    total_loss = 0
    i = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = loss_fn(logits, labels)  # 使用 BCEWithLogitsLoss
        total_loss += loss.item()
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"batch{i}:loss:{loss},total_loss:{total_loss}")
        i+=1

    return total_loss / len(data_loader)

# 评估函数
def eval_model(model, data_loader, device,fold_dir, k=3):
    model = model.eval()
    predictions_list = []
    labels_list = []
    pass_at_1_list = []
    pass_at_k_list = []
    pass_at_k_best_list = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

           # 将logits应用sigmoid激活
            predictions_raw = torch.sigmoid(logits)
            predictions = (predictions_raw > 0.5).float()  # Apply threshold 0.5 for multi-label classification
            # 收集预测值和标签
            predictions_list.append(predictions.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
            # 计算 pass@1 和 pass@k
            pass_at_1_batch = torch.zeros_like(predictions_raw)  # (batch_size, num_labels)
            pass_at_k_batch = torch.zeros_like(predictions_raw)

            for i in range(predictions_raw.size(0)):  # Loop over the batch size
                # 获取每行的最大值位置 (pass@1)
                top_1_idx = predictions_raw[i].argsort()[-1].item()  # Top 1 prediction index
                pass_at_1_batch[i, top_1_idx] = 1  # Mark the top 1

                # 获取每行前k个最大值的位置 (pass@k)
                top_k_idx = predictions_raw[i].argsort()[-k:].cpu().numpy()  # Top k prediction indices
                pass_at_k_batch[i, top_k_idx] = 1  # Mark the top 3

                # pass_at_k_best_batch = [predictions_raw[i]]  # Top k prediction indices

            pass_at_1_list.append(pass_at_1_batch.cpu().numpy())
            pass_at_k_list.append(pass_at_k_batch.cpu().numpy()) 
            pass_at_k_best_list.append(predictions_raw.cpu().numpy())        

    # 将所有预测和标签合并为一个 DataFrame
    predictions_array = np.concatenate(predictions_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    # 将 pass@1 和 pass@k 保存为 excel 文件
    pass_at_1_array = np.concatenate(pass_at_1_list, axis=0)
    pass_at_k_array = np.concatenate(pass_at_k_list, axis=0)
    pass_at_k_best_array = np.concatenate(pass_at_k_best_list, axis=0)
    
    # 保存为 Excel 文件
    predictions_df = pd.DataFrame(predictions_array, columns=[f"label_{i+1}" for i in range(predictions_array.shape[1])])
    labels_df = pd.DataFrame(labels_array, columns=[f"label_{i+1}" for i in range(labels_array.shape[1])])
    pass_at_1_df = pd.DataFrame(pass_at_1_array, columns=[f"label_{i+1}" for i in range(pass_at_1_array.shape[1])])
    pass_at_k_df = pd.DataFrame(pass_at_k_array, columns=[f"label_{i+1}" for i in range(pass_at_k_array.shape[1])])
    pass_at_k_best_df = pd.DataFrame(pass_at_k_best_array, columns=[f"label_{i+1}" for i in range(pass_at_k_best_array.shape[1])])

    predictions_df.to_excel(os.path.join(fold_dir, 'llm_predict_baseline.xlsx'), index=False)
    labels_df.to_excel(os.path.join(fold_dir, 'labels_baseline.xlsx'), index=False)
    pass_at_1_df.to_excel(os.path.join(fold_dir, 'pass_at_1.xlsx'), index=False)
    pass_at_k_df.to_excel(os.path.join(fold_dir, f'pass_at_{k}.xlsx'), index=False)
    pass_at_k_best_df.to_excel(os.path.join(fold_dir, f'pass_at_{k}_best.xlsx'), index=False)

    
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 主训练函数，处理每个fold的训练和评估
def cross_validation_training(data_dir, n_splits=10):
    EPOCHS = 100
    k=3
    for fold in range(1, n_splits):
        if fold==2:
            break
        print(f"Training fold {fold}...")
        train_file = os.path.join(data_dir, f"fold_{fold}", "train.csv")
        test_file = os.path.join(data_dir, f"fold_{fold}", "test.csv")
        
        # 加载训练集和测试集
        # train_texts, train_labels, train_feature, train_weight = load_data(train_file)
        # test_texts, test_labels, test_feature, test_weight = load_data(test_file)
        train_texts, train_labels, train_weight = load_data(train_file,fold,"train")
        test_texts, test_labels, test_weight = load_data(test_file,fold,"test")
        # 加载tokenizer和二分类模型
        # config = BertConfig.from_pretrained('google-bert/bert-base-chinese')  
        # config.hidden_dropout_prob = 0.2  # 设置隐藏层的 Dropout 率  
        # config.attention_probs_dropout_prob = 0.2  # 设置注意力概率的 Dropout 率  
        tokenizer = AutoTokenizer.from_pretrained("D:/LLM-code/LLM_Models/bert-base-chinese")
        model = AutoModelForSequenceClassification.from_pretrained("D:/LLM-code/LLM_Models/bert-base-chinese", num_labels=len(functional_keys))
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = model.to(device)
        
        # 创建数据集和数据加载器
        train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len=512, test=True)
        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_len=512, test=False)

        # 使用 BCEWithLogitsLoss
        pos_weight = torch.tensor(train_weight, dtype=torch.float32).to(device)  
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # loss_fn = torch.nn.BCEWithLogitsLoss()

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))
        
        # 设置优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,  weight_decay=1e-2)
        optimizer = BertAdam(model.parameters(), lr=2e-5,  warmup=.1)
        total_steps = len(train_loader) * EPOCHS   # 训练步数
        scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=len(train_loader)//2,
                    num_training_steps=total_steps)  # 调度器
        
        best_precision,best_recall,best_f1=0,0,0
        # 训练模型
        for epoch in range(30):
            train_loss = train_model(model, train_loader, optimizer, scheduler, device, loss_fn)
            eval_model(model, test_loader, device, os.path.join(data_dir, f"fold_{fold}"), k)
            pass_at_1_list=pd.read_excel(os.path.join(data_dir, f'fold_{fold}/pass_at_1.xlsx')).values
            pass_at_k_list=pd.read_excel(os.path.join(data_dir, f'fold_{fold}/pass_at_{k}.xlsx')).values
            pass_at_k_best_list=pd.read_excel(os.path.join(data_dir, f'fold_{fold}/pass_at_{k}_best.xlsx')).values
            labels_array=pd.read_excel(os.path.join(data_dir, f'fold_{fold}/labels_baseline.xlsx')).values
             # 计算 pass@1 和 pass@k
            pass_at_1_count = 0
            pass_at_k_count = 0
            pass_at_k_best_count = 0
            
            total_samples = len(pass_at_1_list)
            # 针对每一行
            for i in range(total_samples):
                true_labels = labels_array[i]
                
                # pass@1 计算
                pass_at_1 = pass_at_1_list[i]
                if any(pass_at_1[j] == 1 and true_labels[j] == 1 for j in range(len(true_labels))):
                    pass_at_1_count += 1

                # pass@k 计算
                pass_at_k = pass_at_k_list[i]
                true_number=0
                predict_true_number=0
                for j in range(len(pass_at_k)):
                    if pass_at_k[j] == 1 and true_labels[j] == 1:
                        predict_true_number += 1
                    if true_labels[j]==1:
                        true_number+=1
                pass_at_k_count+=predict_true_number/min(true_number,k)

                # pass@k_best 计算
                true_number=0
                predict_true_number=0
                current_k = sum(true_labels[j] == 1 for j in range(len(true_labels)))
                top_k_idx = pass_at_k_best_list[i].argsort()[-current_k:]
                pass_at_k_best=np.zeros(len(true_labels))
                for idx in top_k_idx:
                    pass_at_k_best[idx] = 1
                for j in range(len(true_labels)):
                    if pass_at_k_best[j] == 1 and true_labels[j] == 1:
                        predict_true_number += 1
                    if true_labels[j]==1:
                        true_number+=1
                pass_at_k_best_count+=predict_true_number/min(true_number, total_labels = len(functional_keys))

            pass_at_1 = pass_at_1_count / total_samples
            pass_at_k = pass_at_k_count / total_samples
            pass_at_k_best = pass_at_k_best_count / total_samples
            # 加载预测和标签数据
            y_true = pd.read_excel(os.path.join(data_dir, f'fold_{fold}/labels_baseline.xlsx'))
            y_predict = pd.read_excel(os.path.join(data_dir, f'fold_{fold}/llm_predict_baseline.xlsx'))
            
            # 计算 classification_report
            report = classification_report(y_true, y_predict, zero_division=0)
            print(report)

            # 提取样本平均 precision、recall 和 f1-score
            report_dict = classification_report(y_true, y_predict, zero_division=0, output_dict=True)
            precision = report_dict['samples avg']['precision']
            recall = report_dict['samples avg']['recall']
            f1 = report_dict['samples avg']['f1-score']

            mcc_per_label = [matthews_corrcoef(y_true.iloc[:, i], y_predict.iloc[:, i]) for i in range(y_true.shape[1])]    
            mcc_average = np.mean(mcc_per_label) 

            # 保存到文本文件
            output_file = os.path.join(data_dir,f'fold_{fold}/bert_result.txt')
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f'第{epoch+1}轮:\n')
                f.write(report)
                f.write('\n')
                f.write(f'pass@1:{pass_at_1}\n')
                f.write(f'pass@3:{pass_at_k}\n')
                f.write(f'pass@k:{pass_at_k_best}\n')
                f.write(f"MCC per label: {mcc_per_label}\n")  
                f.write(f'Average MCC: {mcc_average}\n')
                f.write(f'loss:{train_loss}\n')

            print(f'Epoch {epoch + 1}, Loss: {train_loss}')
            # 比较指标是否符合条件，并保存最好的模型
            if (
                # 任意两个指标占优势
                (precision > best_precision and recall > best_recall) or
                (precision > best_precision and f1 > best_f1) or
                (recall > best_recall and f1 > best_f1) or
                # 一个指标增幅大于另外两个指标的下降
                (f1 - best_f1 > (best_recall - recall + best_precision - precision)) or
                (precision - best_precision > (best_recall - recall + best_f1 - f1)) or
                (recall - best_recall > (best_precision - precision + best_f1 - f1))
            ):
                # 保存模型
                torch.save(model, os.path.join(data_dir, f"fold_{fold}/best_model_epoch_{epoch+1}.pt"))
                best_precision, best_recall, best_f1 = precision, recall, f1
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f'Best Precision:{best_precision}\n')
                    f.write(f'Best Recall:{best_recall}\n')
                    f.write(f'Best F1:{best_f1}\n')
                    f.write(f'Best epoch:{epoch+1}\n')

                print("Best Precision: ", best_precision)
                print("Best Recall: ", best_recall)
                print("Best F1: ", best_f1)


# 设置数据文件夹路径
data_dir = 'splits'

# 执行十倍交叉验证训练
cross_validation_training(data_dir)
end_time = time.time()