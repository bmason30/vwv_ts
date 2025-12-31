"""
Rate limiting functionality to comply with SEC's 10 requests/second limit
"""
import time
import threading
from collections import deque
from functools import wraps
from typing import Callable, Any
from .config import MAX_REQUESTS_PER_SECOND
from .exceptions import RateLimitExceeded


class RateLimiter:
    """
    Thread-safe rate limiter using a sliding window algorithm
    Ensures no more than MAX_REQUESTS_PER_SECOND requests in any 1-second window
    """

    def __init__(self, max_requests: int = MAX_REQUESTS_PER_SECOND, time_window: float = 1.0):
        """
        Initialize the rate limiter

        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.lock = threading.Lock()

    def _clean_old_requests(self, current_time: float) -> None:
        """Remove request timestamps older than the time window"""
        cutoff_time = current_time - self.time_window
        while self.request_times and self.request_times[0] <= cutoff_time:
            self.request_times.popleft()

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire permission to make a request

        Args:
            blocking: If True, wait until a slot is available. If False, raise exception immediately.

        Returns:
            True if request is allowed

        Raises:
            RateLimitExceeded: If blocking=False and rate limit is exceeded
        """
        with self.lock:
            current_time = time.time()
            self._clean_old_requests(current_time)

            if len(self.request_times) < self.max_requests:
                # We have capacity, record this request
                self.request_times.append(current_time)
                return True

            if not blocking:
                raise RateLimitExceeded()

            # Calculate how long to wait
            oldest_request = self.request_times[0]
            wait_time = oldest_request + self.time_window - current_time

        # Wait outside the lock to allow other threads to proceed
        if wait_time > 0:
            time.sleep(wait_time)

        # Try again after waiting
        return self.acquire(blocking=True)

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to rate-limit a function

        Usage:
            rate_limiter = RateLimiter()

            @rate_limiter
            def api_call():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            self.acquire(blocking=True)
            return func(*args, **kwargs)
        return wrapper


# Global rate limiter instance for the entire module
_global_rate_limiter = RateLimiter(max_requests=MAX_REQUESTS_PER_SECOND, time_window=1.0)


def rate_limited(func: Callable) -> Callable:
    """
    Convenience decorator using the global rate limiter

    Usage:
        @rate_limited
        def fetch_data():
            ...
    """
    return _global_rate_limiter(func)


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    return _global_rate_limiter
