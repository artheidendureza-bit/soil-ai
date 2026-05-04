import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

start = time.time()

if device.type == "cuda":
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b) 
    torch.cuda.synchronize() 

end = time.time()
print("Time taken:", end - start, "seconds")

size = 20000
print(f"creating {size}x{size} matrices...")
a = torch.randn(size, size)
b = torch.randn(size, size)

print("Multiplying...")
start = time.time()
c = torch.matmul(a,b)

end = time.time()
print("Time taken:", end-start, "seconds")

if device.type == "cuda":
    print("\nGPU is screaming fast!")
else:
    print("\nCPU is chillin'...")