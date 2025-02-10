
from abc import ABC, abstractmethod
class Measurments():
    ''' abstract base class that gets key argument of functions amd values of the parameters needed for the function
    :param kwargs: key value pairs of the parameters needed for the function
    :returns: the value of the functions as a List'''
    def __init__(self,*args):
        self.measures = [*args]
    def get_measures(self) ->list:
        '''return the measures as a list'''
