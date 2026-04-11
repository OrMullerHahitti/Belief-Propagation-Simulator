import { useMemo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { useStore } from '../store';
import { MiniCostTable } from './MiniCostTable';
import { computeRMessageAndHighlight } from '../utils/bp_math';

interface FactorNodeData {
  name: string;
  costTable?: number[][] | number[];
  costLabels?: string[];
  neighbors?: string[];
}

export const FactorNode = ({ id, data }: { id: string; data: FactorNodeData }) => {
  const currentStep = useStore((s) => s.currentStep);
  const snapshots = useStore((s) => s.snapshots);
  const edges = useStore((s) => s.edges);
  const factorHighlightModes = useStore((s) => s.factorHighlightModes);
  const setFactorHighlightMode = useStore((s) => s.setFactorHighlightMode);

  // get highlight mode from store (-1 = both, 0+ = index of target var)
  const highlightMode = factorHighlightModes[id] ?? -1;

  // get current snapshot if available
  const snapshot = snapshots[currentStep];

  // get connected variables (sorted for consistent ordering)
  const connectedVars = useMemo(() => {
    return edges
      .filter((e) => e.source === id || e.target === id)
      .map((e) => (e.source === id ? e.target : e.source))
      .sort();
  }, [edges, id]);

  // compute highlighted cells from R message computation
  const highlightedCells = useMemo(() => {
    if (!snapshot || !data.costTable || !Array.isArray(data.costTable[0])) {
      return [];
    }

    const costTable = data.costTable as number[][];

    // get cost_labels from snapshot if available, otherwise from node data
    const costLabels =
      snapshot.cost_labels?.[data.name] || data.costLabels || [];

    if (costLabels.length < 2) {
      return [];
    }

    const highlights: [number, number][] = [];

    // determine which target variables to compute highlights for based on mode
    // -1 = all (both), 0+ = specific index
    const targetVars =
      highlightMode === -1
        ? connectedVars
        : connectedVars[highlightMode]
          ? [connectedVars[highlightMode]]
          : [];

    // for each target variable, compute which cells achieve the argmin
    // when computing the R message to that variable
    for (const targetVar of targetVars) {
      // find Q messages coming INTO this factor from other variables
      const otherVars = connectedVars.filter((v) => v !== targetVar);

      for (const sourceVar of otherVars) {
        const qKey = `${sourceVar}->${data.name}`;
        const incomingQ = snapshot.Q?.[qKey];

        if (incomingQ && Array.isArray(incomingQ)) {
          try {
            const results = computeRMessageAndHighlight(
              costTable,
              costLabels,
              targetVar,
              incomingQ,
              'min'
            );
            // collect highlighted cells from results
            for (const r of results) {
              highlights.push([r.cellRow, r.cellCol]);
            }
          } catch {
            // ignore computation errors
          }
        }
      }
    }

    // deduplicate
    const seen = new Set<string>();
    return highlights.filter(([r, c]) => {
      const key = `${r},${c}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [snapshot, data.costTable, data.costLabels, data.name, connectedVars, highlightMode]);

  const hasCostTable = data.costTable && data.costTable.length > 0;

  // cycle through highlight modes: -1 (both), 0, 1, 2, ... (each connected var)
  const cycleHighlightMode = (e: React.MouseEvent) => {
    e.stopPropagation();
    const numVars = connectedVars.length;
    if (numVars < 2) return;
    // cycle: -1 -> 0 -> 1 -> ... -> (numVars-1) -> -1
    const nextMode = highlightMode === -1 ? 0 : (highlightMode + 1) % numVars === 0 ? -1 : highlightMode + 1;
    setFactorHighlightMode(id, nextMode);
  };

  // label for current mode
  const getModeLabel = () => {
    if (connectedVars.length < 2) return '↔';
    if (highlightMode === -1) return '↔';
    return `→${connectedVars[highlightMode] || '?'}`;
  };

  return (
    <div
      className={`rounded-md flex flex-col items-center justify-center border-2 border-black bg-gray-900 text-white shadow-lg p-1 ${
        hasCostTable ? 'min-w-16 min-h-16' : 'w-14 h-14'
      }`}
    >
      <div className="font-bold text-[10px] mb-0.5">{data.name}</div>
      {hasCostTable && data.costTable && (
        <>
          <MiniCostTable
            costTable={data.costTable}
            highlightedCells={highlightedCells}
            maxCellSize={12}
          />
          {/* toggle button to cycle highlight direction */}
          {connectedVars.length >= 2 && (
            <button
              onClick={cycleHighlightMode}
              className="mt-0.5 px-1 py-0 text-[7px] bg-gray-700 hover:bg-gray-600 rounded text-gray-300 leading-tight"
              title="Click to cycle R message direction"
            >
              {getModeLabel()}
            </button>
          )}
        </>
      )}
      <Handle
        type="source"
        position={Position.Bottom}
        id="a"
        className="w-3 h-3 bg-gray-900"
        style={{ bottom: -6 }}
      />
      <Handle
        type="target"
        position={Position.Top}
        id="b"
        className="w-3 h-3 bg-gray-900"
        style={{ top: -6 }}
      />
      <Handle
        type="source"
        position={Position.Left}
        id="c"
        className="w-3 h-3 bg-gray-900"
        style={{ left: -6 }}
      />
      <Handle
        type="target"
        position={Position.Right}
        id="d"
        className="w-3 h-3 bg-gray-900"
        style={{ right: -6 }}
      />
    </div>
  );
};
