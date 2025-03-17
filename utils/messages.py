class Messages:
    @staticmethod
    def get_message(messages:List[Message],sender:Agent,recipient:Agent)->Message:
        for message in messages:
            if message.sender == sender and message.recipient == recipient:
                return message
        return None
