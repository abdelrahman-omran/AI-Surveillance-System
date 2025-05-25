import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import torch.nn.functional as F
import json

#Evaluation Phase: Predict on test set
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    model.load_state_dict(torch.load('finetuned_facenet.pth', map_location=device))

    test_df = pd.read_csv('eval_set.csv')
    mtcnn_eval = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True)
    model.eval()
    
    results = []
    confidence_threshold = 0.8  # Tune this threshold based on your validation set
    
    # Inverse mapping from class index to person label
    # Load label mapping
    with open('label_mapping.json', 'r') as f:
        person2idx = json.load(f)

    # Create inverse mapping
    idx2person = {int(idx): person for person, idx in person2idx.items()}

    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            image_path = 'test/' + row['image_path']
            try:
                img = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                pred = "doesn't_exist"
                results.append({'gt': pred, 'image': image_path})
                continue
            face = mtcnn_eval(img)
            if face is None:
                pred = "doesn't_exist"
            else:
                face = face.unsqueeze(0).to(device)
                logits = model(face)
                probs = F.softmax(logits, dim=1)
                max_prob, pred_idx = torch.max(probs, dim=1)
                if max_prob.item() < confidence_threshold:
                    pred = "doesn't_exist"
                else:
                    pred = idx2person[pred_idx.item()]
            results.append({'gt': pred, 'image': image_path})
    
    submission_df = pd.DataFrame(results)
    submission_df.to_csv('face_output.csv', index=False)
    print("Output file saved as face_output.csv")

if __name__ == '__main__':
    main()
