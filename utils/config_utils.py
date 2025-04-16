import pickle
import os
from datetime import datetime



def save_object_as_pickle(data_object, directory, filename):
  """
  Saves a Python object to a pickle file in the specified directory.

  Args:
    data_object: The Python object to save.
    directory: The path to the directory where the file should be saved.
               The directory will be created if it doesn't exist.
    filename: The name for the pickle file (e.g., 'my_data.pkl').
              It's recommended to use a .pkl or .pickle extension.

  Returns:
    None. Prints a success message or raises an exception on error.

  Raises:
    pickle.PicklingError: If the object cannot be pickled.
    OSError: If there's an issue creating the directory or writing the file.
  """
  # Ensure the filename has a reasonable extension if not provided
  if not (filename.endswith('.pkl') or filename.endswith('.pickle')):
      print(f"Warning: Filename '{filename}' doesn't have a standard .pkl or .pickle extension.")

  # Construct the full file path
  full_path = os.path.join(directory, filename)

  try:
    # Create the directory if it doesn't exist
    # exist_ok=True prevents an error if the directory already exists
    os.makedirs(directory, exist_ok=True)

    # Open the file in binary write mode ('wb')
    # Using 'with' ensures the file is properly closed even if errors occur
    with open(full_path, 'wb') as file_handle:
      # Serialize the object and write it to the file
      pickle.dump(data_object, file_handle)

    print(f"Object successfully saved to: {full_path}")

  except pickle.PicklingError as pe:
    print(f"Error: Failed to pickle object. {pe}")
    raise # Re-raise the exception if needed
  except OSError as oe:
    print(f"Error: Could not create directory or write file at {full_path}. {oe}")
    raise # Re-raise the exception if needed
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    raise # Re-raise the exception if needed


def load_object_from_pickle(directory, filename):
  """
  Loads a Python object from a pickle file in the specified directory.

  Args:
    directory: The path to the directory where the file is located.
    filename: The name of the pickle file (e.g., 'my_data.pkl').

  Returns:
    The loaded Python object, or None if the file is not found or
    an error occurs during loading.

  Raises:
    FileNotFoundError: If the specified file does not exist.
    pickle.UnpicklingError: If the file content cannot be unpickled.
    EOFError: If the file is empty or truncated.
    OSError: If there's an issue reading the file.
  """
  # Construct the full file path
  full_path = os.path.join(directory, filename)

  try:
    # Check if the file exists before trying to open it
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Error: Pickle file not found at {full_path}")

    # Open the file in binary read mode ('rb')
    with open(full_path, 'rb') as file_handle:
      # Deserialize the object from the file
      loaded_object = pickle.load(file_handle)
      print(f"Object successfully loaded from: {full_path}")
      return loaded_object

  except FileNotFoundError as fnfe:
      print(fnfe)
      raise # Re-raise the specific error
  except (pickle.UnpicklingError, EOFError) as ue:
      print(f"Error: Failed to unpickle file. It might be corrupted or empty. {ue}")
      raise # Re-raise the specific error
  except OSError as oe:
      print(f"Error: Could not read file at {full_path}. {oe}")
      raise # Re-raise the specific error
  except Exception as e:
      print(f"An unexpected error occurred during loading: {e}")
      raise # Re-raise the specific error




