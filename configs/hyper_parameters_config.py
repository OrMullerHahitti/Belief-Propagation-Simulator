# configs for FactorAgents
from bp_base.computators import MaxSumComputator
from utils import ct_utils

# message parameters
MESSAGE_DOMAIN_SIZE = 3
# Cost Table Creation Function and parameters
CT_CREATION_FUNCTION = ct_utils.create_random_int_table
CT_CREATION_PARAMS = {
    "low": 0,
    "high": 10,
}

# computator
COMPUTATOR = MaxSumComputator()
