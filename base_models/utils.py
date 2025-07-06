"""
Utility functions for common patterns in the belief propagation simulator.
These functions encapsulate repetitive control structures to promote
bottom-up programming and code reuse.
"""

from typing import List, Dict, Callable, Any, TypeVar, Optional
import numpy as np

T = TypeVar('T')
R = TypeVar('R')


def process_items_with_validation(
    items: List[T], 
    validator: Callable[[T], bool], 
    processor: Callable[[T], R],
    error_logger: Optional[Callable[[str, T], None]] = None
) -> List[R]:
    """
    Process a list of items with validation and error handling.
    
    Args:
        items: List of items to process
        validator: Function to validate each item
        processor: Function to process valid items
        error_logger: Optional function to log validation errors
        
    Returns:
        List of processed results from valid items
    """
    results = []
    
    for item in items:
        if validator(item):
            results.append(processor(item))
        elif error_logger:
            error_logger("Validation failed for item", item)
            
    return results


def filter_and_transform(
    items: List[T], 
    filter_func: Callable[[T], bool], 
    transform_func: Callable[[T], R]
) -> List[R]:
    """
    Filter items and transform the remaining ones.
    
    Args:
        items: List of items to process
        filter_func: Function to filter items
        transform_func: Function to transform filtered items
        
    Returns:
        List of transformed items that passed the filter
    """
    return [transform_func(item) for item in items if filter_func(item)]


def batch_process(
    items: List[T], 
    batch_size: int, 
    processor: Callable[[List[T]], List[R]]
) -> List[R]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        processor: Function to process each batch
        
    Returns:
        Flattened list of all batch results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = processor(batch)
        results.extend(batch_results)
        
    return results


def safe_dict_lookup(
    dictionary: Dict[str, T], 
    keys: List[str], 
    default_factory: Callable[[], T] = lambda: None
) -> List[T]:
    """
    Safely lookup multiple keys in a dictionary with default handling.
    
    Args:
        dictionary: Dictionary to lookup from
        keys: List of keys to lookup
        default_factory: Function to create default values
        
    Returns:
        List of values or defaults
    """
    return [dictionary.get(key, default_factory()) for key in keys]


def validate_and_extract(
    data_dict: Dict[str, Any], 
    required_keys: List[str],
    validators: Dict[str, Callable[[Any], bool]] = None
) -> Dict[str, Any] | None:
    """
    Validate and extract data from a dictionary.
    
    Args:
        data_dict: Dictionary to validate
        required_keys: Keys that must be present
        validators: Optional validators for specific keys
        
    Returns:
        Validated data dictionary or None if validation fails
    """
    validators = validators or {}
    
    # Check required keys
    missing_keys = [key for key in required_keys if key not in data_dict]
    if missing_keys:
        return None
        
    # Validate specific keys
    for key, validator in validators.items():
        if key in data_dict and not validator(data_dict[key]):
            return None
            
    return {key: data_dict[key] for key in required_keys}


def create_index_mapping(
    items: List[T], 
    key_extractor: Callable[[T], str]
) -> Dict[str, T]:
    """
    Create a mapping from extracted keys to items.
    
    Args:
        items: List of items to map
        key_extractor: Function to extract key from item
        
    Returns:
        Dictionary mapping keys to items
    """
    return {key_extractor(item): item for item in items}


def conditional_aggregate(
    data: np.ndarray, 
    conditions: List[Callable[[np.ndarray], bool]], 
    aggregator: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray | None:
    """
    Apply aggregation conditionally based on data validation.
    
    Args:
        data: Input data array
        conditions: List of validation conditions
        aggregator: Function to aggregate data if conditions pass
        
    Returns:
        Aggregated data or None if conditions fail
    """
    if not all(condition(data) for condition in conditions):
        return None
        
    return aggregator(data)


def build_nested_structure(
    flat_items: List[T], 
    grouper: Callable[[T], str], 
    transformer: Callable[[List[T]], R]
) -> Dict[str, R]:
    """
    Build nested structure from flat items by grouping and transforming.
    
    Args:
        flat_items: List of flat items
        grouper: Function to extract group key from item
        transformer: Function to transform grouped items
        
    Returns:
        Dictionary mapping group keys to transformed groups
    """
    groups = {}
    
    for item in flat_items:
        group_key = grouper(item)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(item)
        
    return {key: transformer(items) for key, items in groups.items()}