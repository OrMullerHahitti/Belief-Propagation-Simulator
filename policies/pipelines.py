from typing import List

from toolz import pipe

from bp_base.components import Message
from DCOP_base import Policy
from policies.abstract import MessagePolicy


def message_pipline(policies: List[MessagePolicy],message:Message) -> Message:
    return pipe(message,policies)


