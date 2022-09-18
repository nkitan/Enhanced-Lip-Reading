import argparse
import json
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from django.http import HttpResponse

from .lipreading.model import Lipreading
from .preprocessing.transform import crop_img, warp_img

STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
START_IDX = 48
STOP_IDX = 68
CROP_WIDTH = CROP_HEIGHT = 96


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def load_model(config_path: Path, numClasses):
    with config_path.open() as fp:
        config = json.load(fp)
    tcn_options = {
        'num_layers': config['tcn_num_layers'],
        'kernel_size': config['tcn_kernel_size'],
        'dropout': config['tcn_dropout'],
        'dwpw': config['tcn_dwpw'],
        'width_mult': config['tcn_width_mult'],
    }
    return Lipreading(
        num_classes=int(numClasses),
        tcn_options=tcn_options,
        backbone_type=config['backbone_type'],
        relu_type=config['relu_type'],
        width_mult=config['width_mult'],
        extract_feats=False,
    )


def visualize_probs(vocab, probs, col_width=4, col_height=300):
    num_classes = len(probs)
    out = np.zeros((col_height, num_classes * col_width + (num_classes - 1), 3), dtype=np.uint8)
    for i, p in enumerate(probs):
        x = (col_width + 1) * i
        cv2.rectangle(out, (x, 0), (x + col_width - 1, round(p * col_height)), (255, 255, 255), 1)
    top = np.argmax(probs)
    print(f'Prediction: {vocab[top]}')
    print(f'Confidence: {probs[top]:.3f}')
    cv2.putText(out, f'Prediction: {vocab[top]}', (10, out.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,color=(255, 255, 255))
    cv2.putText(out, f'Confidence: {probs[top]:.3f}', (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,color=(255, 255, 255))
    return out


def getPrediction(response, puType, numClasses, modelPath, configPath, wordListPath, videoPath):
    fullWordListPath = 'lip_reader_ai/labels/' + wordListPath
    fullConfigPath = 'lip_reader_ai/configs/' + configPath
    fullModelPath = 'lip_reader_ai/models/' + modelPath
    configPath = Path(fullConfigPath)
    modelPath = Path(fullModelPath)
    deviceType = 'cpu' if int(puType) == 0 else 'cuda'
    print(deviceType)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=deviceType)
    model = load_model(configPath, numClasses)
    model.load_state_dict(torch.load(modelPath, map_location=deviceType)['model_state_dict'])
    model = model.to(deviceType)

    mean_face_landmarks = np.load(Path('lip_reader_ai/landmarks/words_mean_face.npy'))

    with Path(fullWordListPath).open() as fp:
        vocab = fp.readlines()

    with VideoCapture(videoPath) as cap:
        occurrences = {}
        highest_confidence = {}
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        queueLength = length
        queue = deque(maxlen=queueLength)
        print("Length of video: ", length)
        while True:
            ret, image_np = cap.read()
            if not ret:
                break
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            all_landmarks = fa.get_landmarks(image_np)
            if all_landmarks:
                landmarks = all_landmarks[0]

                # BEGIN PROCESSING

                trans_frame, trans = warp_img(landmarks[STABLE_PNTS_IDS, :], mean_face_landmarks[STABLE_PNTS_IDS, :], image_np, STD_SIZE)
                trans_landmarks = trans(landmarks)
                patch = crop_img(trans_frame, trans_landmarks[START_IDX:STOP_IDX], CROP_HEIGHT // 2, CROP_WIDTH // 2)

                patch_torch = to_tensor(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)).to(deviceType)
                queue.append(patch_torch)

                print(len(queue))
                if len(queue) >= queueLength:
                    with torch.no_grad():
                        model_input = torch.stack(list(queue), dim=1).unsqueeze(0)
                        logits = model(model_input, lengths=[queueLength])
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        probs = probs[0].detach().cpu().numpy() if int(puType) == 0 else probs[0].detach().cuda().cpu().numpy()
                    top = np.argmax(probs)

                    print(f'Prediction: {vocab[top]}')
                    print(f'Confidence: {probs[top]:.3f}')
                    
                    if vocab[top] in occurrences:
                        occurrences[vocab[top]] = occurrences[vocab[top]] + 1
                    else:
                        occurrences.update({vocab[top]:1})
                    
                    if vocab[top] in highest_confidence:
                        if (highest_confidence[vocab[top]] < probs[top]):
                            highest_confidence[vocab[top]] = probs[top]
                    else:
                        highest_confidence.update({vocab[top]:probs[top]})

                for x, y in landmarks:
                    cv2.circle(image_np, (int(x), int(y)), 2, (0, 0, 255))
        
        print("Prediction Occurrences: ", occurrences)
        if(len(occurrences) > 0):
            highest_occurrence = max(occurrences, key=occurrences.get)
            print("Prediction with highest occurrence: ",highest_occurrence)

        print("Added confidence: ", highest_confidence)
        highest_confidence_prediction = 0

        if(len(highest_confidence) > 0):
            highest_confidence_prediction = max(highest_confidence, key=highest_confidence.get)
            print("Prediction with highest added confidence: ",highest_confidence_prediction)  
            highest_confidence = highest_confidence[highest_confidence_prediction]
        
    cv2.destroyAllWindows()
    return HttpResponse(json.dumps({'videoName': videoPath, 'confidence': str(highest_confidence), 'prediction': highest_confidence_prediction}))