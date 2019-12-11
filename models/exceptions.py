class BaseRDEException(Exception):
    """
    Base exception class
    """
class NotAnArrowException(BaseRDEException):
    """
    Raised when arrow recognition algorithm confirms a line is not an arrow.
    """