import sys
from huggingface_hub import snapshot_download
for m in sys.argv[1:]:
    print("FETCHING", m, flush=True)
    snapshot_download(m)
    print("DONE", m, flush=True)
print("ALL_DONE", flush=True)
