# PropFlow Web - Implementation Details

This document describes the implementation details of the PropFlow Web visual editor and simulator.

**Build Status**: All TypeScript errors resolved, frontend builds successfully.

---

## Table of Contents

1. [Backend Simulation Fixes](#1-backend-simulation-fixes)
2. [Mini Cost Table Visualization](#2-mini-cost-table-visualization)
3. [Argmin Cell Highlighting](#3-argmin-cell-highlighting)
4. [Auto-Resize Cost Tables](#4-auto-resize-cost-tables)
5. [Unary Constraints](#5-unary-constraints)
6. [Damping](#6-damping)
7. [Multiple Factor Connections](#7-multiple-factor-connections)
8. [Message Flow Visualization on Edges](#8-message-flow-visualization-on-edges)

---

## 1. Backend Simulation Fixes

### Files Modified
- `src/propflow_server/simulation.py`

### Problem
The original backend had three critical bugs:
1. **Wrong Computator initialization**: `BPComputator(op_mode=op_mode)` - the `BPComputator` class doesn't accept `op_mode`
2. **Wrong SnapshotManager initialization**: `SnapshotManager(graph)` - `SnapshotManager` takes no constructor arguments
3. **Redundant snapshot capture**: Manual `snapshot_manager.capture_step()` with wrong signature

### Solution

**Computator Selection:**
```python
from propflow.bp.computators import MinSumComputator, MaxSumComputator, SumProductComputator

COMPUTATOR_MAP = {
    "min_sum": MinSumComputator,
    "max_sum": MaxSumComputator,
    "sum_product": SumProductComputator,
}

# Usage
computator_cls = COMPUTATOR_MAP.get(op_mode, MinSumComputator)
computator = computator_cls()
```

**Engine Initialization:**
```python
# BPEngine creates its own SnapshotManager internally
engine = BPEngine(graph, computator=computator)

# engine.step(i) captures snapshots internally at engine._snapshots[i]
for i in range(max_iters):
    engine.step(i)
    snap = engine._snapshots[i]  # retrieve captured snapshot
```

### How PropFlow Computators Work
- `BPComputator` base class takes `reduce_func` and `combine_func`
- `MinSumComputator`: `reduce_func=np.min`, `combine_func=np.add`
- `MaxSumComputator`: `reduce_func=np.max`, `combine_func=np.add`
- `SumProductComputator`: `reduce_func=np.sum`, `combine_func=np.multiply`

---

## 2. Mini Cost Table Visualization

### Files Created
- `src/propflow_web/src/components/MiniCostTable.tsx`

### Purpose
Display a compact visualization of factor cost tables directly on factor nodes in the graph editor.

### Component Interface
```typescript
interface MiniCostTableProps {
  costTable: number[][] | number[];  // 2D (binary) or 1D (unary)
  highlightedCells?: [number, number][];  // [row, col] pairs to highlight
  maxCellSize?: number;  // max pixels per cell (default: 14)
}
```

### Implementation Details
- Uses CSS Grid for layout with `gap-px` to create 1px separators
- Cells are ~12px each to fit within factor node bounds
- Highlighted cells use yellow background (`bg-yellow-300`)
- Non-highlighted cells use dark background (`bg-gray-800`)
- Hover tooltip shows `[row,col] = value`
- Values are formatted to fit: integers shown as-is, decimals truncated

### Styling
```css
/* Grid container */
grid-template-columns: repeat(cols, minmax(0, maxCellSize));
grid-template-rows: repeat(rows, minmax(0, maxCellSize));

/* Cell styling */
.cell-highlighted { background: yellow-300; color: gray-900; font-bold }
.cell-normal { background: gray-800; color: gray-300 }
```

---

## 3. Argmin Cell Highlighting

### Files Modified
- `src/propflow_web/src/components/FactorNode.tsx`
- `src/propflow_web/src/utils/bp_math.ts` (existing, used)

### Purpose
During simulation playback, highlight the cost table cells that achieved the argmin in the R message computation.

### How R Message Computation Works
For a binary factor f(x1, x2) computing R_{f→x1}:
```
R_{f→x1}[a] = min_b (C[a,b] + Q_{x2→f}[b])
```
For each value `a` of x1, there's a winning value `b*` that achieves the minimum. The highlighted cell is `(a, b*)`.

### Highlight Mode Toggle
Three modes, cycled by clicking the toggle button:
1. **→x1**: Show argmin cells for R message to first connected variable
2. **→x2**: Show argmin cells for R message to second connected variable
3. **both**: Show all argmin cells

### Implementation
```typescript
type HighlightMode = 0 | 1 | 2;  // first var, second var, both

const highlightedCells = useMemo(() => {
  // Get connected variables
  const connectedVars = edges.filter(...).map(...).sort();

  // Filter based on highlight mode
  const targetVars = highlightMode === 2
    ? connectedVars
    : [connectedVars[highlightMode]];

  // For each target, compute argmin cells using computeRMessageAndHighlight()
  for (const targetVar of targetVars) {
    const results = computeRMessageAndHighlight(
      costTable, costLabels, targetVar, incomingQ, 'min'
    );
    // Collect [cellRow, cellCol] from results
  }
}, [snapshot, highlightMode, ...]);
```

### The `computeRMessageAndHighlight` Function
Located in `src/propflow_web/src/utils/bp_math.ts`:
- Takes cost table, labels, target variable, incoming Q message
- Returns array of results with `cellRow`, `cellCol` for each argmin

---

## 4. Auto-Resize Cost Tables

### Files Modified
- `src/propflow_web/src/store.ts` (`onConnect` handler)

### Problem
When adding a new factor and connecting variables, the cost table remained at its default 2×1 size instead of resizing to match connected variable domains.

### Solution
Updated the `onConnect` handler to:
1. Detect when an edge connects to a factor
2. Count all connected variables after the new edge
3. Resize cost table to match variable domains
4. Store `costLabels` with variable names in connection order

### Implementation
```typescript
onConnect: (connection) => {
  const newEdges = addEdge(connection, edges);

  // Find factor node in connection
  const factorNode = sourceNode?.type === 'factor' ? sourceNode : targetNode;

  if (factorNode) {
    // Get all connected variables
    const connectedVars = newEdges
      .filter(e => e.source === factorNode.id || e.target === factorNode.id)
      .map(e => /* get variable node */);

    const domains = connectedVars.map(v => v.data.domainSize);

    // Resize based on connection count
    if (connectedVars.length === 1) {
      newTable = Array(domains[0]).fill(0);  // unary
    } else if (connectedVars.length === 2) {
      newTable = Array(domains[0]).fill(0).map(() =>
        Array(domains[1]).fill(0)
      );  // binary
    }

    // Update node with new table and labels
    updateNodeData(factorNode.id, {
      costTable: newTable,
      costLabels: connectedVars.map(v => v.data.name)
    });
  }
}
```

---

## 5. Unary Constraints

### Files Modified
- `src/propflow_web/src/store.ts` - Added `unaryCost` to `VariableData`
- `src/propflow_web/src/components/Inspector.tsx` - Added `UnaryCostEditor`
- `src/propflow_server/models.py` - Added `unary_cost` to `VariableSpec`
- `src/propflow_server/simulation.py` - Creates unary factors

### What Are Unary Constraints?
Unary constraints assign costs to individual variable values, independent of other variables. For example, `[0, 5]` means:
- Value 0 has cost 0 (preferred)
- Value 1 has cost 5 (penalized)

### Frontend Implementation

**Store Type:**
```typescript
interface VariableData {
  name: string;
  domainSize: number;
  unaryCost?: number[];  // cost per domain value
}
```

**UnaryCostEditor Component:**
- Toggle button to enable/disable unary costs
- Input fields for each domain value
- Quick-generate buttons: Zero, Random, Prefer 0

```typescript
const UnaryCostEditor = ({ domainSize, unaryCost, onChange }) => {
  const [enabled, setEnabled] = useState(!!unaryCost);

  // Toggle creates/removes the array
  const handleToggle = () => {
    if (enabled) {
      onChange(undefined);  // remove
    } else {
      onChange(Array(domainSize).fill(0));  // create
    }
  };

  // Generate presets
  const generate = (type) => {
    if (type === 'prefer0') {
      onChange([0, 1, 1, ...]);  // 0 for first, 1 for rest
    }
  };
};
```

### Backend Implementation

**Creating Unary Factors:**
```python
def build_graph(spec: GraphSpec) -> FactorGraph:
    # ... create regular factors ...

    # Create unary factors for variables with unary costs
    for v_spec in spec.variables:
        if v_spec.unary_cost is not None:
            unary_cost = np.array(v_spec.unary_cost, dtype=float)
            unary_factor = FactorAgent.create_from_cost_table(
                name=f"u_{v_spec.name}",  # prefix with u_
                cost_table=unary_cost
            )
            factor_agents.append(unary_factor)
            edges[unary_factor] = [var_map[v_spec.name]]
```

### Model Update
```python
class VariableSpec(BaseModel):
    name: str
    domain_size: int = 2
    unary_cost: Optional[List[float]] = None

class SnapshotJSON(BaseModel):
    # ... other fields ...
    cost_tables: Dict[str, Union[List[float], List[List[float]]]]  # 1D or 2D
```

---

## 6. Damping

### Files Modified
- `src/propflow_web/src/store.ts` - Added `dampingFactor` state
- `src/propflow_web/src/App.tsx` - Added damping slider UI
- `src/propflow_server/simulation.py` - Uses `DampingEngine`

### What Is Damping?
Damping is a message smoothing technique that prevents oscillations in BP by blending new messages with previous ones:

```
new_message = λ * previous_message + (1 - λ) * computed_message
```

Where `λ` is the damping factor:
- `λ = 0`: No damping (pure new message)
- `λ = 0.5`: Equal blend
- `λ = 0.9`: Heavy damping (90% previous, 10% new)

### Frontend Implementation

**Store State:**
```typescript
interface AppState {
  dampingFactor: number;  // 0 to 0.99
  maxIters: number;
  setDampingFactor: (value: number) => void;
  setMaxIters: (value: number) => void;
}

// Initial values
dampingFactor: 0.0,
maxIters: 10,
```

**UI Controls:**
```tsx
{/* Damping slider */}
<input
  type="range"
  min={0}
  max={0.99}
  step={0.01}
  value={dampingFactor}
  onChange={(e) => setDampingFactor(Number(e.target.value))}
/>
<div className="flex justify-between">
  <span>None</span>
  <span>Heavy</span>
</div>
```

**API Payload:**
```typescript
const spec = {
  variables,
  factors,
  config: {
    max_iters: maxIters,
    engine_type: 'min_sum',
    damping: dampingFactor  // passed to backend
  }
};
```

### Backend Implementation

**Engine Selection:**
```python
from propflow.bp.engines import DampingEngine

def run_simulation(spec: GraphSpec):
    # ... build graph and computator ...

    damping = spec.config.damping
    if damping > 0:
        engine = DampingEngine(
            graph,
            computator=computator,
            damping_factor=damping
        )
    else:
        engine = BPEngine(graph, computator=computator)
```

### How DampingEngine Works
`DampingEngine` extends `BPEngine` and overrides `post_var_compute()`:

```python
class DampingEngine(BPEngine):
    def __init__(self, factor_graph, damping_factor=0.9, **kwargs):
        super().__init__(factor_graph, **kwargs)
        self.damping_factor = damping_factor

    def post_var_compute(self, var):
        # Apply damping to each outgoing message
        damp(var, x=self.damping_factor)
```

The `damp()` function blends `var.mailer.outbox` messages with `var.last_iteration` messages.

---

## 7. Multiple Factor Connections

### Files Modified
- `src/propflow_web/src/components/VariableNode.tsx`

### Problem
Variable nodes only had 2 handles (top/bottom), limiting connections to 2 factors maximum.

### Solution
Added 8 handles to variable nodes (4 source + 4 target) at all positions:
- Top, Bottom, Left, Right for both source and target types

```tsx
{/* Multiple handles to allow connections to many factors */}
<Handle type="source" position={Position.Top} id="t" />
<Handle type="source" position={Position.Bottom} id="b" />
<Handle type="source" position={Position.Left} id="l" />
<Handle type="source" position={Position.Right} id="r" />
<Handle type="target" position={Position.Top} id="tt" />
<Handle type="target" position={Position.Bottom} id="tb" />
<Handle type="target" position={Position.Left} id="tl" />
<Handle type="target" position={Position.Right} id="tr" />
```

This allows variables to connect to multiple factors from any direction.

---

## 8. Message Flow Visualization on Edges

### Files Created
- `src/propflow_web/src/components/MessageEdge.tsx`

### Files Modified
- `src/propflow_web/src/components/GraphEditor.tsx`

### Purpose
Display Q and R message values directly on the edges during simulation playback, with different colors to distinguish direction.

### Implementation

**Custom Edge Component:**
```tsx
// MessageEdge.tsx
export const MessageEdge = ({ source, target, ... }: EdgeProps) => {
  const snapshot = snapshots[currentStep];

  // Determine Q key (var → factor) and R key (factor → var)
  if (sourceIsVar && !targetIsVar) {
    qKey = `${sourceName}->${targetName}`;
    rKey = `${targetName}->${sourceName}`;
  }

  // Get message values
  const qMessage = snapshot?.Q?.[qKey];
  const rMessage = snapshot?.R?.[rKey];

  // Display labels offset perpendicular to edge
  return (
    <>
      <BaseEdge path={edgePath} />
      <EdgeLabelRenderer>
        {qDisplay && <div className="bg-blue-100 text-blue-700">Q{qDisplay}</div>}
        {rDisplay && <div className="bg-orange-100 text-orange-700">R{rDisplay}</div>}
      </EdgeLabelRenderer>
    </>
  );
};
```

**Edge Type Registration:**
```tsx
// GraphEditor.tsx
const edgeTypes = {
  message: MessageEdge,
};

const edgesWithType = edges.map((e) => ({ ...e, type: 'message' }));

<ReactFlow edges={edgesWithType} edgeTypes={edgeTypes} />
```

### Visual Design
- **Q messages** (variable → factor): Blue background (`bg-blue-100`), blue text
- **R messages** (factor → variable): Orange background (`bg-orange-100`), orange text
- Labels positioned perpendicular to edge center
- Format: `Q(0.0,1.5)` or `R(0.0,0.0)` showing values for each domain element
- Only visible during simulation playback when snapshot data exists

### Direction Handling
The component automatically determines which node is the variable and which is the factor:
- If source is variable → Q goes source→target, R goes target→source
- If source is factor → Q goes target→source, R goes source→target

---

## Architecture Overview

```
Frontend (React + TypeScript + Vite)
├── store.ts           - Zustand state management
├── App.tsx            - Main app with controls
├── components/
│   ├── GraphEditor.tsx    - React Flow canvas
│   ├── FactorNode.tsx     - Factor node with MiniCostTable
│   ├── VariableNode.tsx   - Variable node (multi-handle)
│   ├── MessageEdge.tsx    - Custom edge with Q/R labels
│   ├── MiniCostTable.tsx  - Compact cost table grid
│   ├── Inspector.tsx      - Property editor panel
│   └── CostTableEditor.tsx- Full cost table editor
└── utils/
    └── bp_math.ts         - BP math utilities

Backend (FastAPI)
├── main.py            - API routes
├── models.py          - Pydantic models
└── simulation.py      - Simulation logic
    ├── build_graph()      - Creates FactorGraph with unary factors
    ├── serialize_snapshot() - Converts to JSON
    └── run_simulation()   - Runs BP with optional damping
```

---

## Data Flow

1. **User builds graph** in React Flow canvas
2. **User configures** damping, max iterations, unary costs
3. **Click "Run Simulation"** triggers API call with graph spec
4. **Backend builds FactorGraph** including unary factors
5. **Backend runs BPEngine/DampingEngine** for max_iters steps
6. **Snapshots captured** at each step (Q/R messages, assignments, costs)
7. **Snapshots returned** to frontend
8. **Frontend plays back** snapshots with slider
9. **Factor nodes update** highlighting based on current snapshot

---

## Key Types

### Frontend
```typescript
interface VariableData {
  name: string;
  domainSize: number;
  unaryCost?: number[];
}

interface FactorData {
  name: string;
  costTable: number[][] | number[];
  costLabels?: string[];
}

interface Snapshot {
  step: number;
  Q: Record<string, number[]>;
  R: Record<string, number[]>;
  assignments: Record<string, number>;
  global_cost?: number;
  cost_tables?: Record<string, number[][]>;
  cost_labels?: Record<string, string[]>;
}
```

### Backend
```python
class VariableSpec(BaseModel):
    name: str
    domain_size: int = 2
    unary_cost: Optional[List[float]] = None

class FactorSpec(BaseModel):
    name: str
    neighbors: List[str]
    cost_table: Union[List[float], List[List[float]]]

class EngineConfig(BaseModel):
    max_iters: int = 10
    damping: float = 0.0
    engine_type: str = "min_sum"
```
