from bp_base.DCOP_base import NeighborAddingPolicy


class CustomNeighborAddingPolicy(NeighborAddingPolicy):
    def add_neighbors(self, node, neighbor) -> bool:
        # Custom logic for adding neighbors
        if neighbor not in node.neighbors:
            node.neighbors.append(neighbor)
            return True
        else:
            return False


class BPNeighborAddingPolicy(NeighborAddingPolicy):
    "" "Add neighbors only if they are of different type" ""

    def add_neighbors(self, node, neighbor):
        "" "Add neighbors only if they are of different type" ""
        # Custom logic for adding neighbors
        if neighbor not in node.neighbors and node.type != neighbor.type:
            node.neighbors.append(neighbor)
            return True
        else:
            return False
