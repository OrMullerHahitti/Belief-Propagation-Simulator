import { create } from 'zustand';
import { 
  addEdge, 
  applyNodeChanges, 
  applyEdgeChanges,
  type Node, 
  type Edge, 
  type Connection, 
  type OnNodesChange, 
  type OnEdgesChange 
} from '@xyflow/react';

// Types mirror the backend models roughly
export interface VariableData {
  name: string;
  domainSize: number;
  currentBelief?: number[];
  assignment?: number;
  unaryCost?: number[]; // unary constraint costs per domain value
}

export interface FactorData {
  name: string;
  costTable: number[][] | number[]; // 2D or 1D
  costLabels?: string[]; // [row, col]
  neighbors?: string[]; // ordered names
}

export interface Snapshot {
  step: number;
  Q: Record<string, number[]>; // "x->f"
  R: Record<string, number[]>; // "f->x"
  assignments: Record<string, number>;
  global_cost?: number;
  cost_tables?: Record<string, number[][]>;
  cost_labels?: Record<string, string[]>; // factor_name -> [row_var, col_var]
}

export type DampingMode = 'q' | 'r' | 'both';
export type MessageType = 'Q' | 'R';

export interface SelectedMessage {
  type: MessageType;
  key: string; // "src->dst"
}

// highlight mode: index of target variable in sorted connected vars, or -1 for "both"
// e.g., for factor connected to [x1, x2]: 0 = →x1, 1 = →x2, -1 = both
export type HighlightMode = number;

interface AppState {
  nodes: Node[];
  edges: Edge[];

  // Graph building
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: (connection: Connection) => void;
  addVariable: () => void;
  addFactor: () => void;
  updateNodeData: (id: string, data: any) => void;

  // Simulation config
  dampingFactor: number;
  dampingMode: DampingMode;
  maxIters: number;
  setDampingFactor: (value: number) => void;
  setDampingMode: (mode: DampingMode) => void;
  setMaxIters: (value: number) => void;

  // Simulation state
  runId: string | null;
  snapshots: Snapshot[];
  currentStep: number;
  isPlaying: boolean;

  // Selection
  selectedNodeId: string | null;
  selectedMessage: SelectedMessage | null;

  // Factor highlight modes (factor id -> target var index, -1 = both)
  factorHighlightModes: Record<string, HighlightMode>;
  setFactorHighlightMode: (factorId: string, mode: HighlightMode) => void;

  setSnapshots: (id: string, snaps: Snapshot[]) => void;
  setCurrentStep: (step: number) => void;
  togglePlay: () => void;
  resetSimulation: () => void;

  setSelectedNode: (id: string | null) => void;
  setSelectedMessage: (msg: SelectedMessage | null) => void;
}

export const useStore = create<AppState>((set, get) => ({
  nodes: [
    { id: 'x1', type: 'variable', position: { x: 100, y: 100 }, data: { name: 'x1', domainSize: 2 } },
    { id: 'f1', type: 'factor', position: { x: 300, y: 100 }, data: { name: 'f1', costTable: [[0, 1], [1, 0]] } },
    { id: 'x2', type: 'variable', position: { x: 500, y: 100 }, data: { name: 'x2', domainSize: 2 } },
  ],
  edges: [
    { id: 'e1', source: 'x1', target: 'f1' },
    { id: 'e2', source: 'f1', target: 'x2' },
  ],
  
  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },
  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },
  onConnect: (connection) => {
    const { nodes, edges } = get();
    const newEdges = addEdge(connection, edges);

    // find the factor node in this connection (if any)
    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);

    const factorNode = sourceNode?.type === 'factor' ? sourceNode :
                       targetNode?.type === 'factor' ? targetNode : null;

    if (factorNode) {
      // get all variables connected to this factor (after adding new edge)
      const connectedVarIds = newEdges
        .filter((e) => e.source === factorNode.id || e.target === factorNode.id)
        .map((e) => (e.source === factorNode.id ? e.target : e.source));

      const connectedVars = connectedVarIds
        .map((id) => nodes.find((n) => n.id === id))
        .filter((n) => n?.type === 'variable');

      // resize cost table if we have connected variables
      if (connectedVars.length > 0) {
        const domains = connectedVars.map((v) => (v?.data as unknown as VariableData)?.domainSize || 2);
        const currentTable = (factorNode.data as unknown as FactorData).costTable;

        // only resize if dimensions changed
        const needsResize =
          (connectedVars.length === 1 && Array.isArray(currentTable[0])) ||
          (connectedVars.length === 2 && (!Array.isArray(currentTable[0]) ||
            currentTable.length !== domains[0] ||
            (currentTable[0] as number[]).length !== domains[1]));

        if (needsResize) {
          let newTable: number[][] | number[];
          if (connectedVars.length === 1) {
            // unary factor
            newTable = Array(domains[0]).fill(0);
          } else {
            // binary factor (2D table)
            newTable = Array(domains[0]).fill(0).map(() => Array(domains[1]).fill(0));
          }

          // update factor's cost table and store neighbor order
          const neighborNames = connectedVars.map((v) => v?.data?.name || v?.id);
          set({
            edges: newEdges,
            nodes: nodes.map((n) =>
              n.id === factorNode.id
                ? { ...n, data: { ...n.data, costTable: newTable, costLabels: neighborNames } }
                : n
            ),
          });
          return;
        }
      }
    }

    set({ edges: newEdges });
  },
  
  addVariable: () => {
    const id = `x${get().nodes.length + 1}`;
    const newNode: Node = {
      id,
      type: 'variable',
      position: { x: Math.random() * 400, y: Math.random() * 400 },
      data: { name: id, domainSize: 2 },
    };
    set({ nodes: [...get().nodes, newNode] });
  },
  addFactor: () => {
    const id = `f${get().nodes.length + 1}`;
    const newNode: Node = {
      id,
      type: 'factor',
      position: { x: Math.random() * 400, y: Math.random() * 400 },
      data: { name: id, costTable: [[0],[1]] }, // Default unary-ish
    };
    set({ nodes: [...get().nodes, newNode] });
  },
  updateNodeData: (id, data) => {
    set({
      nodes: get().nodes.map((node) => 
        node.id === id ? { ...node, data: { ...node.data, ...data } } : node
      ),
    });
  },
  
  // Simulation config
  dampingFactor: 0.0,
  dampingMode: 'q',
  maxIters: 10,
  setDampingFactor: (value) => set({ dampingFactor: value }),
  setDampingMode: (mode) => set({ dampingMode: mode }),
  setMaxIters: (value) => set({ maxIters: value }),

  // Simulation state
  runId: null,
  snapshots: [],
  currentStep: 0,
  isPlaying: false,

  selectedNodeId: null,
  setSelectedNode: (id) => set({ selectedNodeId: id }),
  selectedMessage: null,
  setSelectedMessage: (msg) => set({ selectedMessage: msg }),

  // Factor highlight modes
  factorHighlightModes: {},
  setFactorHighlightMode: (factorId, mode) =>
    set({ factorHighlightModes: { ...get().factorHighlightModes, [factorId]: mode } }),

  setSnapshots: (id, snaps) => set({ runId: id, snapshots: snaps, currentStep: 0, selectedMessage: null }),
  setCurrentStep: (step) => set({ currentStep: step }),
  togglePlay: () => set({ isPlaying: !get().isPlaying }),
  resetSimulation: () =>
    set({ runId: null, snapshots: [], currentStep: 0, isPlaying: false, selectedMessage: null }),
}));
