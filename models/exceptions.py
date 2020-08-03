class BaseRDEException(Exception):
    """
    Base exception class
    """
class NotAnArrowException(BaseRDEException):
    """
    Raised when arrow recognition algorithm confirms a line is not an arrow.
    """

class NoArrowsFoundException(BaseRDEException):
    """
    Raised when no arrows have been found in an image
    """

class AnchorNotFoundException(BaseRDEException):
    """
    Raised when a text line cannot be anchored in a figure or crop.
    """