import { useState } from 'react';
import { useStore } from '../store';
import type { VariableData } from '../store';
import { CostTableEditor } from './CostTableEditor';

// Unary cost editor for variable nodes
interface UnaryCostEditorProps {
  domainSize: number;
  unaryCost?: number[];
  onChange: (costs: number[]) => void;
}

const UnaryCostEditor = ({ domainSize, unaryCost, onChange }: UnaryCostEditorProps) => {
  const [enabled, setEnabled] = useState(!!unaryCost);

  const handleToggle = () => {
    if (enabled) {
      onChange(undefined as any); // remove unary cost
      setEnabled(false);
    } else {
      onChange(Array(domainSize).fill(0));
      setEnabled(true);
    }
  };

  const handleCellChange = (idx: number, value: string) => {
    const costs = unaryCost ? [...unaryCost] : Array(domainSize).fill(0);
    costs[idx] = parseFloat(value) || 0;
    onChange(costs);
  };

  const generate = (type: 'zero' | 'random' | 'prefer0') => {
    let costs: number[];
    if (type === 'zero') {
      costs = Array(domainSize).fill(0);
    } else if (type === 'random') {
      costs = Array(domainSize).fill(0).map(() => Number((Math.random() * 5).toFixed(1)));
    } else {
      // prefer0: lower cost for value 0
      costs = Array(domainSize).fill(0).map((_, i) => i === 0 ? 0 : 1);
    }
    onChange(costs);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="block text-xs font-semibold uppercase text-gray-500">Unary Cost</label>
        <button
          onClick={handleToggle}
          className={`text-[10px] px-2 py-0.5 rounded ${enabled ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-600'}`}
        >
          {enabled ? 'On' : 'Off'}
        </button>
      </div>

      {enabled && unaryCost && (
        <>
          <div className="flex gap-1 mb-1 justify-end">
            <button onClick={() => generate('zero')} className="text-[10px] px-1 bg-slate-200 rounded">0</button>
            <button onClick={() => generate('random')} className="text-[10px] px-1 bg-slate-200 rounded">Rnd</button>
            <button onClick={() => generate('prefer0')} className="text-[10px] px-1 bg-slate-200 rounded">Prefer 0</button>
          </div>
          <div className="flex gap-1 flex-wrap">
            {unaryCost.map((cost, idx) => (
              <div key={idx} className="flex flex-col items-center">
                <span className="text-[9px] text-gray-400">{idx}</span>
                <input
                  type="number"
                  step="0.1"
                  value={cost}
                  onChange={(e) => handleCellChange(idx, e.target.value)}
                  className="w-12 px-1 py-0.5 border rounded text-xs text-center"
                />
              </div>
            ))}
          </div>
          <div className="text-[10px] text-gray-400 mt-1">
            Cost for each domain value (lower = preferred)
          </div>
        </>
      )}
    </div>
  );
};

export const Inspector = () => {
  const { nodes, updateNodeData } = useStore();
  const selectedId = useStore((s) => s.selectedNodeId);

  const selectedNode = nodes.find((n) => n.id === selectedId);

  if (!selectedNode) {
    return (
      <div className="p-4 text-gray-500 text-sm text-center mt-10">
        Select a node to edit its properties.
      </div>
    );
  }

  const isFactor = selectedNode.type === 'factor';
  const data = selectedNode.data as unknown as VariableData;

  const handleChange = (field: string, value: unknown) => {
    updateNodeData(selectedNode.id, { [field]: value });
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b bg-slate-50">
        <h3 className="font-bold text-lg mb-1">{isFactor ? 'Factor' : 'Variable'}</h3>
        <div className="text-xs text-mono text-gray-500">{selectedNode.id}</div>
      </div>

      <div className="p-4 space-y-4 flex-1 overflow-auto">
        {/* Common Properties */}
        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Name</label>
          <input
            type="text"
            value={data.name}
            onChange={(e) => handleChange('name', e.target.value)}
            className="w-full px-2 py-1 border rounded text-sm"
          />
        </div>

        {/* Variable Specific */}
        {!isFactor && (
          <>
            <div>
              <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Domain Size</label>
              <input
                type="number"
                min={2}
                max={10}
                value={data.domainSize}
                onChange={(e) => {
                  const newSize = Number(e.target.value);
                  handleChange('domainSize', newSize);
                  // resize unary cost array if it exists
                  if (data.unaryCost) {
                    const newUnary = Array(newSize).fill(0).map((_, i) => data.unaryCost?.[i] ?? 0);
                    handleChange('unaryCost', newUnary);
                  }
                }}
                className="w-full px-2 py-1 border rounded text-sm"
              />
            </div>

            {/* Unary Cost Editor */}
            <UnaryCostEditor
              domainSize={data.domainSize}
              unaryCost={data.unaryCost}
              onChange={(costs) => handleChange('unaryCost', costs)}
            />
          </>
        )}

        {/* Factor Specific */}
        {isFactor && (
          <div className="flex flex-col gap-4">
             {/* Cost Table Editor */}
             <CostTableEditor
                nodeId={selectedNode.id}
                data={data as unknown as { costTable: number[][] | number[] }}
                updateData={updateNodeData}
             />
          </div>
        )}
      </div>
    </div>
  );
};
