# schedulers.py
import random
from interfaces import Updator

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
