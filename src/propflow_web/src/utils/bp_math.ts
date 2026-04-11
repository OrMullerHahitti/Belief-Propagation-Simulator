// Utility types and functions for BP math

export interface CostTable {
  // Can be 1D (unary) or 2D (binary)
  data: number[] | number[][];
  labels: string[]; // [row_var, col_var] for binary
}

// Check if graph is valid bipartite
export const isBipartite = (_edges: any[]) => {
  // This logic is mostly enforced by the UI builder, preventing var-var or factor-factor edges
  // But useful helper for validation
  return true; 
};

/**
 * Computes the R message highlighting info.
 * R_f->a[a] = min_b (C(a,b) + Q_{b->f}[b])
 * 
 * Returns: {
 *   highlightCells: Array of [row_idx, col_idx] to highlight in cost table
 *   tooltipData: Breakdown of computation
 * }
 */
export const computeRMessageAndHighlight = (
  costTable: number[][],
  costLabels: string[], // [row_var, col_var]
  targetVar: string, // 'a'
  incomingQ: number[], // Q_{b->f}
  op: 'min' | 'max' = 'min'
) => {
  // If costLabels[0] == targetVar, then row is 'a', col is 'b'
  // We iterate over 'a' (target value), and for each 'a', we find the best 'b'
  
  // Wait, R message is a vector over 'a'. 
  // For a specific value of 'a' (say a=0), we want to find b* that minimizes sum.
  // The UI usually highlights the *entire* R message generation or a specific element?
  // User req: "Highlight the cost-table cell(s) (a, b*) that achieve the argmin."
  
  // Actually, usually we visualize R message values.
  // If the user hovers over the R message edge, maybe we show for EACH 'a' which 'b' won?
  // Or just one?
  
  // Let's assume we return a map: for each index 'i' of targetVar, which cell (i, j) or (j, i) was the winner.
  
  const isRowTarget = costLabels[0] === targetVar;
  const numRows = costTable.length;
  const numCols = costTable[0].length;
  
  const results = [];
  
  // If row is target 'a', then for each row i, we iterate cols j (variable 'b')
  // cost(i, j) + Q[j]
  
  if (isRowTarget) {
     for (let i = 0; i < numRows; i++) {
       let bestVal = op === 'min' ? Infinity : -Infinity;
       let bestIdx = -1;
       
       for (let j = 0; j < numCols; j++) {
         const val = costTable[i][j] + (incomingQ[j] || 0);
         if (op === 'min') {
           if (val < bestVal) { bestVal = val; bestIdx = j; }
         } else {
           if (val > bestVal) { bestVal = val; bestIdx = j; }
         }
       }
       results.push({
         targetIdx: i,
         sourceIdx: bestIdx,
         cost: costTable[i][bestIdx],
         qVal: incomingQ[bestIdx],
         total: bestVal,
         cellRow: i,
         cellCol: bestIdx
       });
     }
  } else {
    // Col is target 'a', row is 'b'
    // for each col j, iterate rows i
    // cost(i, j) + Q[i]
    for (let j = 0; j < numCols; j++) {
       let bestVal = op === 'min' ? Infinity : -Infinity;
       let bestIdx = -1;
       
       for (let i = 0; i < numRows; i++) {
         const val = costTable[i][j] + (incomingQ[i] || 0);
         if (op === 'min') {
           if (val < bestVal) { bestVal = val; bestIdx = i; }
         } else {
            if (val > bestVal) { bestVal = val; bestIdx = i; }
         }
       }
       results.push({
         targetIdx: j,
         sourceIdx: bestIdx,
         cost: costTable[bestIdx][j],
         qVal: incomingQ[bestIdx],
         total: bestVal,
         cellRow: bestIdx, // row is source
         cellCol: j
       });
    }
  }
  
  return results;
};
