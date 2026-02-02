# rate_limiter.py
import time
from collections import defaultdict, deque

RATE_LIMIT = 15
WINDOW_SECONDS = 60
request_log = defaultdict(deque)

def check_rate_limit(api_key: str):
    now = time.time()
    q = request_log[api_key]

    while q and q[0] < now - WINDOW_SECONDS:
        q.popleft()

    if len(q) >= RATE_LIMIT:
        return False

    q.append(now)
    return True

from fastapi import HTTPException

def rate_limit(api_key: str):
    if not check_rate_limit(api_key):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
