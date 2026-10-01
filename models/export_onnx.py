import os
import torch
import torch.nn as nn
import time
import numpy as np

from models.multi_task_heads import MultiModalNeuroLensModel

def export_and_benchmark_onnx():
    print("=== Exporting NeuroLens PyTorch Model to ONNX & Measuring CPU Latency ===")
    os.makedirs('checkpoints', exist_ok=True)
    
    model = MultiModalNeuroLensModel()
    model.eval()
    
    # Create dummy input batch tensors
    dummy_hw_img = torch.randn(1, 1, 64, 64)
    dummy_hw_ts = torch.randn(1, 50, 4)
    dummy_gaze_img = torch.randn(1, 3, 128, 128)
    dummy_gaze_feats = torch.randn(1, 6)
    dummy_speech_feats = torch.randn(1, 8)
    dummy_drawing_img = torch.randn(1, 3, 128, 128)
    dummy_eeg_feats = torch.randn(1, 5)
    
    dummy_batch = {
        'handwriting_img': dummy_hw_img,
        'handwriting_ts': dummy_hw_ts,
        'gaze_img': dummy_gaze_img,
        'gaze_feats': dummy_gaze_feats,
        'speech_feats': dummy_speech_feats,
        'drawing_img': dummy_drawing_img,
        'eeg_feats': dummy_eeg_feats
    }
    
    # Measure PyTorch CPU Inference Latency
    num_runs = 50
    latencies = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(dummy_batch)
            
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_batch)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"PyTorch CPU Inference Latency : {avg_latency:.2f} ms ± {std_latency:.2f} ms per child assessment sample")
    
    # Export ONNX for Speech Encoder
    onnx_path = 'checkpoints/neurolens_speech_encoder.onnx'
    try:
        torch.onnx.export(
            model.speech_enc,
            dummy_speech_feats,
            onnx_path,
            input_names=['speech_feats'],
            output_names=['speech_token'],
            dynamic_axes={'speech_feats': {0: 'batch_size'}},
            opset_version=14
        )
        file_size_kb = os.path.getsize(onnx_path) / 1024.0
        print(f"ONNX Speech Encoder Exported successfully to {onnx_path} ({file_size_kb:.1f} KB)")
    except Exception as e:
        print(f"ONNX export notice: {e}")
        
    return avg_latency

if __name__ == '__main__':
    export_and_benchmark_onnx()
