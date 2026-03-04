import torch

def main():
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_version:", torch.version.cuda)
        print("gpu:", torch.cuda.get_device_name(0))
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        print("gpu_matmul_ok:", y.mean().item())
    else:
        x = torch.randn(1000, 1000)
        y = x @ x
        print("cpu_matmul_ok:", y.mean().item())

if __name__ == "__main__":
    main()