"""
Custom exceptions for the EDGAR module
"""

class EDGARException(Exception):
    """Base exception for all EDGAR module errors"""
    pass


class RateLimitExceeded(EDGARException):
    """Raised when SEC rate limit (10 requests/second) is exceeded"""
    def __init__(self, message="SEC rate limit exceeded. Maximum 10 requests per second allowed."):
        self.message = message
        super().__init__(self.message)


class CIKNotFound(EDGARException):
    """Raised when a CIK cannot be found for a given ticker"""
    def __init__(self, ticker, message=None):
        self.ticker = ticker
        self.message = message or f"CIK not found for ticker: {ticker}"
        super().__init__(self.message)


class InvalidCIK(EDGARException):
    """Raised when a CIK format is invalid"""
    def __init__(self, cik, message=None):
        self.cik = cik
        self.message = message or f"Invalid CIK format: {cik}"
        super().__init__(self.message)


class APIError(EDGARException):
    """Raised when SEC API returns an error"""
    def __init__(self, status_code, message=None):
        self.status_code = status_code
        self.message = message or f"SEC API error: HTTP {status_code}"
        super().__init__(self.message)


class DataNotAvailable(EDGARException):
    """Raised when requested data is not available"""
    def __init__(self, message="Requested data is not available"):
        self.message = message
        super().__init__(self.message)


class ParseError(EDGARException):
    """Raised when data parsing fails"""
    def __init__(self, message="Failed to parse SEC data"):
        self.message = message
        super().__init__(self.message)


class InvalidFormType(EDGARException):
    """Raised when an invalid form type is specified"""
    def __init__(self, form_type, message=None):
        self.form_type = form_type
        self.message = message or f"Invalid form type: {form_type}"
        super().__init__(self.message)


class NetworkError(EDGARException):
    """Raised when network connectivity issues occur"""
    def __init__(self, message="Network error occurred while accessing SEC data"):
        self.message = message
        super().__init__(self.message)
