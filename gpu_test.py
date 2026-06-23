import torch

def check_cuda():
    try:
        print("Torch version:", torch.__version__)
        print("CUDA version compiled with Torch:", torch.version.cuda)
        print("Is CUDA available? ->", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("GPU name:", torch.cuda.get_device_name(0))
            print("Total devices:", torch.cuda.device_count())
        else:
            print("CUDA not available. Likely CPU-only build or driver missing.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check_cuda()