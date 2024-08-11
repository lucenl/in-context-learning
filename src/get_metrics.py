import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter

def extract_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    predictions, targets = [], []
    
    for result in data['results']:
        predictions.append(result['pred'])
        targets.append(result['_target'])
    
    metrics = data['metrics']
    class_metrics = metrics['class_metrics']
    
    return {
        'predictions': predictions,
        'targets': targets,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'f1': metrics['f1_score'],
        'recall': metrics['recall'],
        'harmful_precision': class_metrics['Harmful']['precision'],
        'harmful_f1': class_metrics['Harmful']['f1_score'],
        'harmful_recall': class_metrics['Harmful']['recall'],
        'harmless_precision': class_metrics['Harmless']['precision'],
        'harmless_f1': class_metrics['Harmless']['f1_score'],
        'harmless_recall': class_metrics['Harmless']['recall']
    }

def cal_micro_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = Counter(y_pred).items()
    return {
        'micro_accuracy': accuracy * 100,
        'micro_precision': precision * 100,
        'micro_recall': recall * 100,
        'micro_f1': f1 * 100,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

def cal_macro_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'macro_accuracy': accuracy * 100,
        'macro_precision': precision * 100,
        'macro_recall': recall * 100,
        'macro_f1': f1 * 100,
    }
    
def calculate_manual_metrics(tn, fp, fn, tp):
    if not isinstance(tn, (int, np.integer)):
        return {
        'manual_accuracy': tn,
        'manual_precision': tn,
        'manual_recall': tn,
        'manual_f1': tn
    }
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'manual_accuracy': accuracy * 100,
        'manual_precision': precision * 100,
        'manual_recall': recall * 100,
        'manual_f1': f1 * 100
    }
    
    
def process_directory(root_dir):
    results = []
    nonbi = set()
    for n_shots_dir in os.listdir(root_dir):
        if not n_shots_dir.endswith('_shots'):
            continue
        
        n_shots = n_shots_dir.split('_')[0]
        n_shots_path = os.path.join(root_dir, n_shots_dir)
        
        for selector_type in os.listdir(n_shots_path):
            selector_type_path = os.path.join(n_shots_path, selector_type)
            if not os.path.isdir(selector_type_path):
                continue
            
            for selector_name in os.listdir(selector_type_path):
                selector_name_path = os.path.join(selector_type_path, selector_name)
                if not os.path.isdir(selector_name_path):
                    continue
                
                for item in os.listdir(selector_name_path):
                    if item == 's0':
                        s0_path = os.path.join(selector_name_path, item)
                        for model in os.listdir(s0_path):
                            model_path = os.path.join(s0_path, model)
                            if not os.path.isdir(model_path):
                                continue
                            
                            json_file = os.path.join(model_path, 'test.json')
                            if os.path.exists(json_file):
                                
                                data = extract_metrics(json_file)
                                print("number of nonbinary:")
                                count = 0
                                for i, pre in enumerate(data['predictions']):
                                    if pre != 'Harmless' and pre != 'Harmful':
                                        print(f"non-binary prediction: {pre}")
                                        nonbi.add(json_file)
                                        # data['predictions'][i] = 'Harmful'  # Modify the list element directly
                                        count += 1
                                print(count)
                                micro_metrics = cal_micro_metrics(data['targets'], data['predictions'])
                                macro_metrics = cal_macro_metrics(data['targets'], data['predictions'])
                                manual_metrics = calculate_manual_metrics(micro_metrics['tn'], micro_metrics['fp'], micro_metrics['fn'], micro_metrics['tp'])
                                results.append({
                                    'n_shots': n_shots,
                                    'selector_type': selector_type,
                                    'selector_name': selector_name,
                                    'model': model,
                                    **data,
                                    **micro_metrics,
                                    **macro_metrics,
                                    **manual_metrics
                                })

    print(nonbi)
    return pd.DataFrame(results)

# Specify the root directory
root_dir = '../results/caption_mistral_zero/YOUTUBE/test'  # Current directory

# Process the directory and get the DataFrame
df = process_directory(root_dir)

# Save to Excel
df.to_excel('caption_mistral_zeroresults.xlsx', index=False)