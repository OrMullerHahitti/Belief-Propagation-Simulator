from __future__ import annotations
from typing import Dict, List, Optional, Protocol, Any, Union, TypeVar, Generic
import numpy as np
from functools import singledispatch, singledispatchmethod
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, TypeAlias, TYPE_CHECKING

from DCOP_base import Agent

if TYPE_CHECKING:
    from bp_base.agents import BPAgent

CostTable: TypeAlias = np.ndarray

class Message:
    '''
    Represents a message in the BP algorithm. (Base class) will be extended for different kinds of messages.
    '''
    def __init__(self, data: np.ndarray, sender: Agent, recipient: Agent):
        self.data = data
        self.sender = sender
        self.recipient = recipient
    
    def copy(self) -> Message:
        '''
        Create a copy of this message with a new data array.
        :return: A new Message object with same sender, recipient and a copy of data
        '''
        return Message(
            data=np.copy(self.data),
            sender=self.sender,
            recipient=self.recipient
        )
        
    def __hash__(self):
        return hash((self.sender, self.recipient))
    def __eq__(self, other):
        return self.sender == other.sender and self.recipient == other.recipient
    def __ne__(self, other):
        return not self == other
    def __str__(self):
        return f"Message from {self.sender.name} to {self.recipient.name}: {self.data}"
    def __repr__(self):
        return self.__str__()
    def __add__(self, other):
        if not isinstance(other, Message):
            raise TypeError("Can only add Message to Message")
        if self.sender.name != other.sender.name and self.recipient.name != other.recipient.name:
            raise ValueError("Cannot add messages from different senders or to different recipients")
        return Message(
            data=self.data + other.data,
            sender=self.sender,
            recipient=self.recipient
        )
    def __mul__(self, other):
        if  isinstance(other, Message):
            if self.sender != other.recipient or self.recipient != other.sender:
                raise ValueError("Cannot multiply messages from different senders or to different recipients")
            else:
                return Message(
                data=self.data * other.data,
                sender=self.sender,
                recipient=self.recipient
            )
        elif isinstance(other, (int, float)):
            return Message(
                data=self.data * other,
                sender=self.sender,
                recipient=self.recipient
            )
        else:
            raise TypeError("Can only multiply Message by int, float or Message")







class MailHandler:
    def __init__(self,_domain_size: int):
        self._message_domain_size = _domain_size
        self._clear_after_staging = True
        self._incoming: List[Message] = []
        self._outgoing: List[Message] = []



    def set_first_message(self,owner: BPAgent,neighbor:BPAgent) -> Optional[Message]:
        """Add a message to the mailbox, replacing any from the same sender."""
        self._incoming.append(Message(np.zeros(self._message_domain_size,), neighbor, owner))
        # for idx, existing in enumerate(self._incoming):
        #     if existing.sender == to_send.sender:
        #         self._incoming[idx] = to_send
        #         return
        # self._incoming.append(to_send)

    #TODO: decide where the mailbox should be cleared
    #TODO: decide where THE SENDER BECOMES THE RECIEVER ETC HERE? IN THE BP AGENT? IN THE COMPUTATOR? I THINK THE BEST IS IN THE AGENT WITH A PRIVATE FUNCTION
    #TODO : might be better to save all incmoing out going in a dict so i can hash them and retrieve them faster
    @singledispatchmethod
    def receive_messages(self, message: Message):
        """Handle a single Message."""
        self._incoming.append(message)

    @receive_messages.register(list)
    def _(self, messages: list[Message]):
        """Handle a list of Messages."""
        for message in messages:
            self.receive_messages(message)
    def send(self):
        for message in self._outgoing:
            message.recipient.mailer.receive_messages(message)

    def stage_sending(self, messages: List[Message]):##i.e staging meaning computing the messages in the inbox
        """Add a message, replacing any from the same sender."""
        self._outgoing= messages.copy()



    def prepare(self):
            self._outgoing.clear()
    #if i want to use from outside the class
    def clear_inbox(self): #important!!!  for variables never clear the inbox
        self._incoming.clear()

    def clear_outgoing(self):
        self._outgoing.clear()

    @property
    def inbox(self) -> List[Message]:
        return self._incoming

    @property
    def outbox(self) -> List[Message]:
        return self._outgoing

    @property
    def clear(self):
        return self._clear_after_staging

    @clear.setter
    def clear(self, value: bool):
        self._clear_after_staging = value

    @singledispatchmethod
    def _receiver_sender_turnaround(self, arg: Union[Message, List[Message]]):
        raise NotImplementedError(f"Cannot handle {type(arg)}")

    @_receiver_sender_turnaround.register
    def _(self, arg: Message) -> Message:
        return Message(
            data=arg.data,
            sender=arg.recipient,
            recipient=arg.sender
        )

    @_receiver_sender_turnaround.register(list)
    def _(self, arg: List[Message]) -> List[Message]:
        return [
            Message(
                data=message.data,
                sender=message.recipient,
                recipient=message.sender
            )
            for message in arg
        ]


    def __getitem__(self, sender_name:str) -> Optional[Message]:
        for msg in self._incoming:
            if msg.sender.name == sender_name:
                return msg
        return None
    def __setitem__(self,sender_name:str, message: Message):
        pass

    def __len__(self):
        return len(self._incoming)

    def __iter__(self):
        return iter(self._incoming)




class BPMessage(Message):
    pass

