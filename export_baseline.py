import torch
from rois.estimator.unet_resnet18 import TemporalUnet

print("1. Initializing TemporalUnet architecture...")
model = TemporalUnet(gru_hidden=64, gru_kernel=3, insertion_point=5)

print("2. Loading baseline weights...")
state = torch.load('weights/airport_baseline_batch_8/full_model_best.pt', map_location='cpu')
state = {k: v for k, v in state.items() if 'bottleneck_gru' not in k}
model.load_state_dict(state, strict=False)
model.eval()

print("3. Tracing baseline to TorchScript...")
# Monkey-patch the forward method directly to avoid adding a "model." prefix!
model.forward = model.forward_without_gru

dummy_img = torch.randn(1, 3, 448, 768)
with torch.no_grad():
    traced_model = torch.jit.trace(model, dummy_img)

save_path = 'weights/airport_baseline_batch_8/full_model_best.torchscript.pt'
traced_model.save(save_path)
print(f"Successfully saved TorchScript model to: {save_path}")
