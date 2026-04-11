import { useEffect, useState } from 'react';

interface CostTableEditorProps {
  nodeId: string;
  data: { costTable: number[][] | number[] };
  updateData: (id: string, data: Record<string, unknown>) => void;
}

export const CostTableEditor = ({ nodeId, data, updateData }: CostTableEditorProps) => {
  // normalize to 2D for consistent handling
  const normalize = (table: number[][] | number[]): number[][] => {
    if (!table || table.length === 0) return [[0]];
    if (typeof table[0] === 'number') {
      // 1D array - convert to column vector
      return (table as number[]).map(v => [v]);
    }
    return table as number[][];
  };

  const [localTable, setLocalTable] = useState<number[][]>(() => normalize(data.costTable));

  useEffect(() => {
    setLocalTable(normalize(data.costTable));
  }, [nodeId, data.costTable]);

  const rows = localTable.length;
  const cols = localTable[0]?.length || 1;
  const isUnary = cols === 1;

  const updateCell = (r: number, c: number, val: string) => {
    const num = parseFloat(val) || 0;
    const newTable = localTable.map((row, ri) =>
      ri === r ? row.map((cell, ci) => (ci === c ? num : cell)) : [...row]
    );
    setLocalTable(newTable);
    // save as 1D if unary, 2D otherwise
    const saveTable = isUnary ? newTable.map(row => row[0]) : newTable;
    updateData(nodeId, { costTable: saveTable });
  };

  const generate = (type: 'random' | 'potts' | 'zero') => {
    const newTable = Array(rows)
      .fill(0)
      .map((_, r) =>
        Array(cols)
          .fill(0)
          .map((_, c) => {
            if (type === 'random') return Number((Math.random() * 10).toFixed(1));
            if (type === 'potts') return r === c ? 0 : 1;
            return 0;
          })
      );
    setLocalTable(newTable);
    const saveTable = isUnary ? newTable.map(row => row[0]) : newTable;
    updateData(nodeId, { costTable: saveTable });
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <label className="block text-xs font-semibold uppercase text-gray-500">Cost Table</label>
        <div className="flex gap-1">
          <button onClick={() => generate('potts')} className="text-[10px] px-1 bg-slate-200 rounded">Potts</button>
          <button onClick={() => generate('random')} className="text-[10px] px-1 bg-slate-200 rounded">Rnd</button>
          <button onClick={() => generate('zero')} className="text-[10px] px-1 bg-slate-200 rounded">0</button>
        </div>
      </div>

      <div className="overflow-auto max-h-60 border rounded bg-white">
        <table className="w-full text-center text-xs">
          <tbody>
            {localTable.map((row, r) => (
              <tr key={r}>
                {row.map((cell, c) => (
                  <td key={c} className="p-0 border">
                    <input
                      className="w-full h-full p-1 text-center outline-none focus:bg-blue-50"
                      value={cell}
                      onChange={(e) => updateCell(r, c, e.target.value)}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-[10px] text-gray-400 mt-1 text-right">
        {isUnary ? `${rows}x1` : `${rows}x${cols}`}
      </div>
    </div>
  );
};
