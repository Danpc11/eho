"""Mide el tiempo de un paso de entrenamiento del EHO en CPU y en cada GPU."""
import sys, time
import torch

sys.path.insert(0, ".")
from eho.data.synthetic import make_dataset
from eho.model import EHO, sample_losses


def step_time(device, n_points=512, batch=32, reps=5):
    torch.manual_seed(0)
    m = EHO().to(device)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    b = make_dataset(batch, "dt2", n_points=n_points)
    for i in range(reps + 1):
        if i == 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()
        out = m(b)
        loss = sample_losses(out["fields"], out["scalars"], b).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.time() - t0) / reps


if __name__ == "__main__":
    torch.set_num_threads(min(26, torch.get_num_threads()))
    print(f"CPU ({torch.get_num_threads()} hilos): {step_time(torch.device('cpu')):.3f} s/paso")
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}: {step_time(torch.device(f'cuda:{i}')):.3f} s/paso")
