# schedulers.py
import random
from abc import ABC, abstractmethod

from DCOP_base.interfaces import Updator
class Updator(ABC):
    @abstractmethod
    def schedule_updates(self, Q_keys, R_keys, iteration: int):
        """
        Given the lists of edges for Q and R, return the order in which they should be updated.
        In synchronous mode, might return something like [('Q', vName, fName), ...] for all Q
        then [('R', fName, vName), ...], or do them in random order, etc.

        Must produce an iterable of tuples of the form:
           ('Q', vName, fName) or ('R', fName, vName)
        indicating the update type and the node names.
        """
        pass



class SynchronousScheduler(Updator):
    """
    Updates all Q first, then all R, in a fixed order (like a standard flooding schedule).
    """
    def schedule_updates(self, Q_keys, R_keys, iteration: int):
        # Q_keys is a list of (vName, fName)
        # R_keys is a list of (fName, vName)
        update_order = []
        # 1) Q updates
        for (v, f) in Q_keys:
            update_order.append(('Q', v, f))
        # 2) R updates
        for (f, v) in R_keys:
            update_order.append(('R', f, v))
        return update_order


class AsynchronousScheduler(Updator):
    """
    Randomly shuffles all edges (both Q and R) each iteration.
    """
    def schedule_updates(self, Q_keys, R_keys, iteration: int):
        updates_q = [('Q', v, f) for (v, f) in Q_keys]
        updates_r = [('R', f, v) for (f, v) in R_keys]
        combined = updates_q + updates_r
        random.shuffle(combined)
        return combined
