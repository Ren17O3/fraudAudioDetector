import psutil, os
print(psutil.Process(os.getpid()).memory_info().rss / 1024**2, "MB")
