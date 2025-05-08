from typing import List

from toolz import pipe

from bp_base.components import Message


class MessagePolicy:
    pass


def message_pipline(policies: List[MessagePolicy],message:Message) -> Message:
    return pipe(message,policies)


