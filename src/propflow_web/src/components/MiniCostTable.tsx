import React from 'react';
import { clsx } from 'clsx';

interface MiniCostTableProps {
  costTable: number[][] | number[];
  highlightedCells?: [number, number][]; // [row, col] pairs to highlight
  maxCellSize?: number; // max px per cell
}

/**
 * Compact cost table visualization for factor nodes.
 * Renders a small grid showing cost values with optional cell highlighting
 * to indicate which cells contributed to R message computation.
 */
export const MiniCostTable: React.FC<MiniCostTableProps> = ({
  costTable,
  highlightedCells = [],
  maxCellSize = 14,
}) => {
  if (!costTable || costTable.length === 0) {
    return <div className="text-[8px] text-gray-400">no table</div>;
  }

  const is2D = Array.isArray(costTable[0]);
  const rows = costTable.length;
  const cols = is2D ? (costTable[0] as number[]).length : 1;

  // build a set of highlighted cells for quick lookup
  const highlightSet = new Set(
    highlightedCells.map(([r, c]) => `${r},${c}`)
  );

  // format value to fit in small cell
  const formatVal = (v: number) => {
    if (Number.isInteger(v) && Math.abs(v) < 100) return String(v);
    if (Math.abs(v) < 10) return v.toFixed(1);
    return v.toFixed(0);
  };

  return (
    <div
      className="grid gap-px bg-gray-600 rounded overflow-hidden"
      style={{
        gridTemplateColumns: `repeat(${cols}, minmax(0, ${maxCellSize}px))`,
        gridTemplateRows: `repeat(${rows}, minmax(0, ${maxCellSize}px))`,
      }}
    >
      {Array(rows)
        .fill(0)
        .map((_, r) =>
          Array(cols)
            .fill(0)
            .map((_, c) => {
              const val = is2D
                ? (costTable[r] as number[])[c]
                : (costTable[r] as number);
              const isHighlighted = highlightSet.has(`${r},${c}`);

              return (
                <div
                  key={`${r}-${c}`}
                  className={clsx(
                    'flex items-center justify-center text-[7px] font-mono leading-none',
                    isHighlighted
                      ? 'bg-yellow-300 text-gray-900 font-bold'
                      : 'bg-gray-800 text-gray-300'
                  )}
                  title={`[${r},${c}] = ${val}`}
                >
                  {formatVal(val)}
                </div>
              );
            })
        )}
    </div>
  );
};
