class MatchError(Exception):
    """
    Exception for when we didn't get a match.
    """
    def __init__(self, msg: str):
        super().__init__(msg)


class BindError(MatchError):
    """
    Exception class for binding errors.
    """
    pass
    
