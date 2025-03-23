def validate_message_direction(func):
    """
    Decorator that validates the direction of messages and applies appropriate transformations.
    - If data is from variable 1 to 2 with name F12, processes normally
    - Otherwise transposes the data first
    """

    def wrapper(self, message, *args, **kwargs):
        # Extract information from data
        sender_name = message.sender.name
        recipient_name = message.recipient.name

        # Check if this follows the expected pattern (e.g., F12 from var1 to var2)
        if sender_name.startswith('V') and recipient_name.startswith('F'):
            # Variable to Factor direction
            var_id = sender_name[1:]  # Extract variable ID number
            factor_name = recipient_name

            # Check if the factor name doesn't match expected pattern (F followed by var IDs)
            if not factor_name.startswith('F') or var_id not in factor_name[1:]:
                # Transpose data if direction doesn't match naming convention
                message.data = message.data.T

        elif sender_name.startswith('F') and recipient_name.startswith('V'):
            # Factor to Variable direction
            var_id = recipient_name[1:]  # Extract variable ID
            factor_name = sender_name

            # Check if variable ID isn't in the expected position in factor name
            if var_id not in factor_name[1:]:
                # Transpose data
                message.data = message.data.T

        # Call the original function with possibly transformed data
        return func(self, message, *args, **kwargs)