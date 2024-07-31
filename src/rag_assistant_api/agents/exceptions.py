class TokenLengthExceedsMaxTokenNumber(Exception):
    """The token number exceeds the maximum token number the model can handle."""


class ModelNotIncluded(Exception):
    """The model name is not supported"""


class NoAnswerError(Exception):
    """The model didn't respond with an answer."""
