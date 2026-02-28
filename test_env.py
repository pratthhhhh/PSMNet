import torch
from models.stackhourglass import PSMNet as StackPSM

def run_smoke_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = StackPSM(192).to(device)
    model.eval()

    B, C, H, W = 1, 3, 256, 512  # reduce H,W if you have limited GPU memory
    left = torch.randn(B, C, H, W, device=device)
    right = torch.randn(B, C, H, W, device=device)

    try:
        with torch.no_grad():
            out = model(left, right)
        print('Forward pass successful. Output type:', type(out))
        if isinstance(out, torch.Tensor):
            print('Output shape:', out.shape)
        elif isinstance(out, (list, tuple)):
            print('Tuple lengths/shapes:', [x.shape if isinstance(x, torch.Tensor) else str(type(x)) for x in out])
    except Exception as e:
        print('Forward failed:', type(e).__name__, e)

if __name__ == '__main__':
    run_smoke_test()
