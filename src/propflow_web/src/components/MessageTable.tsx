import { useMemo, useState } from 'react';
import { useStore } from '../store';
import { ChevronDown, ChevronUp } from 'lucide-react';
import type { MessageType } from '../store';

interface MessageRow {
  iteration: number;
  type: 'Q' | 'R';
  from: string;
  to: string;
  values: number[];
}

/**
 * Table showing message history across iterations.
 * Displays Q (variable → factor) and R (factor → variable) messages.
 */
export const MessageTable = () => {
  const snapshots = useStore((s) => s.snapshots);
  const currentStep = useStore((s) => s.currentStep);
  const setCurrentStep = useStore((s) => s.setCurrentStep);
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const selectedMessage = useStore((s) => s.selectedMessage);
  const setSelectedMessage = useStore((s) => s.setSelectedMessage);
  const [expanded, setExpanded] = useState(true);
  const [showAllIterations, setShowAllIterations] = useState(false);
  const [traceOnlyExisting, setTraceOnlyExisting] = useState(true);

  const adjacency = useMemo(() => {
    const idToNode = new Map(nodes.map((n) => [n.id, n]));
    const varToFactors = new Map<string, Set<string>>();
    const factorToVars = new Map<string, Set<string>>();

    const nodeName = (id: string): string => {
      const node = idToNode.get(id);
      const name = (node?.data as any)?.name;
      return typeof name === 'string' && name.length > 0 ? name : id;
    };

    const nodeType = (id: string): string | undefined => idToNode.get(id)?.type;

    for (const e of edges) {
      const aType = nodeType(e.source);
      const bType = nodeType(e.target);
      if (!aType || !bType) continue;

      const aName = nodeName(e.source);
      const bName = nodeName(e.target);

      if (aType === 'variable' && bType === 'factor') {
        if (!varToFactors.has(aName)) varToFactors.set(aName, new Set());
        if (!factorToVars.has(bName)) factorToVars.set(bName, new Set());
        varToFactors.get(aName)!.add(bName);
        factorToVars.get(bName)!.add(aName);
      } else if (aType === 'factor' && bType === 'variable') {
        if (!varToFactors.has(bName)) varToFactors.set(bName, new Set());
        if (!factorToVars.has(aName)) factorToVars.set(aName, new Set());
        varToFactors.get(bName)!.add(aName);
        factorToVars.get(aName)!.add(bName);
      }
    }

    return {
      varsForFactor: (factor: string) => Array.from(factorToVars.get(factor) ?? []),
      factorsForVar: (varName: string) => Array.from(varToFactors.get(varName) ?? []),
    };
  }, [nodes, edges]);

  // parse messages from snapshots into table rows
  const allMessages = useMemo(() => {
    const rows: MessageRow[] = [];

    snapshots.forEach((snap, iteration) => {
      // parse Q messages (var → factor)
      if (snap.Q) {
        Object.entries(snap.Q).forEach(([key, values]) => {
          const [from, to] = key.split('->');
          if (from && to && Array.isArray(values)) {
            rows.push({ iteration, type: 'Q', from, to, values });
          }
        });
      }

      // parse R messages (factor → var)
      if (snap.R) {
        Object.entries(snap.R).forEach(([key, values]) => {
          const [from, to] = key.split('->');
          if (from && to && Array.isArray(values)) {
            rows.push({ iteration, type: 'R', from, to, values });
          }
        });
      }
    });

    return rows;
  }, [snapshots]);

  // filter to current iteration or show all
  const displayedMessages = useMemo(() => {
    if (showAllIterations) {
      return allMessages;
    }
    return allMessages.filter((row) => row.iteration === currentStep);
  }, [allMessages, currentStep, showAllIterations]);

  // format values for display
  const formatValues = (values: number[]): string => {
    return `(${values.map((v) => v.toFixed(1)).join(', ')})`;
  };

  const traceRows = useMemo(() => {
    if (!selectedMessage) return [];
    const { type, key } = selectedMessage;
    const storeKey = type === 'Q' ? 'Q' : 'R';

    const rows = snapshots.map((snap, iteration) => {
      const values = (snap as any)?.[storeKey]?.[key] as number[] | undefined;
      return { iteration, values };
    });

    return traceOnlyExisting ? rows.filter((r) => Array.isArray(r.values)) : rows;
  }, [selectedMessage, snapshots, traceOnlyExisting]);

  const graphTrace = useMemo(() => {
    if (!selectedMessage) return { upstream: [], downstream: [] } as const;

    const { type, key } = selectedMessage;
    const iteration = currentStep;
    const [src, dst] = key.split('->');
    if (!src || !dst) return { upstream: [], downstream: [] } as const;

    const upstream: { type: MessageType; key: string; iteration: number }[] = [];
    const downstream: { type: MessageType; key: string; iteration: number }[] = [];

    if (type === 'Q') {
      // Q_i(x -> f) depends on R_{i-1}(g -> x) for all g != f
      const prevIter = iteration - 1;
      if (prevIter >= 0) {
        for (const fac of adjacency.factorsForVar(src).filter((f) => f !== dst)) {
          const upKey = `${fac}->${src}`;
          const values = snapshots[prevIter]?.R?.[upKey];
          if (!traceOnlyExisting || Array.isArray(values)) {
            upstream.push({ type: 'R', key: upKey, iteration: prevIter });
          }
        }
      }

      // Q_i(x -> f) influences R_i(f -> y) for all y != x
      for (const v of adjacency.varsForFactor(dst).filter((v) => v !== src)) {
        const downKey = `${dst}->${v}`;
        const values = snapshots[iteration]?.R?.[downKey];
        if (!traceOnlyExisting || Array.isArray(values)) {
          downstream.push({ type: 'R', key: downKey, iteration });
        }
      }
    } else {
      // R_i(f -> x) depends on Q_i(y -> f) for all y != x
      for (const v of adjacency.varsForFactor(src).filter((v) => v !== dst)) {
        const upKey = `${v}->${src}`;
        const values = snapshots[iteration]?.Q?.[upKey];
        if (!traceOnlyExisting || Array.isArray(values)) {
          upstream.push({ type: 'Q', key: upKey, iteration });
        }
      }

      // R_i(f -> x) influences Q_{i+1}(x -> g) for all g != f
      const nextIter = iteration + 1;
      if (nextIter < snapshots.length) {
        for (const fac of adjacency.factorsForVar(dst).filter((f) => f !== src)) {
          const downKey = `${dst}->${fac}`;
          const values = snapshots[nextIter]?.Q?.[downKey];
          if (!traceOnlyExisting || Array.isArray(values)) {
            downstream.push({ type: 'Q', key: downKey, iteration: nextIter });
          }
        }
      }
    }

    return { upstream, downstream } as const;
  }, [selectedMessage, currentStep, adjacency, snapshots, traceOnlyExisting]);

  if (snapshots.length === 0) {
    return null;
  }

  return (
    <div className="border-t bg-white">
      {/* header with toggle */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-2 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm text-slate-700">Message Log</span>
          <span className="text-xs text-gray-400">
            ({displayedMessages.length} messages)
          </span>
        </div>
        {expanded ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
      </button>

      {expanded && (
        <div className="px-4 pb-3">
          {/* filter toggle */}
          <div className="mb-2 flex items-center gap-2">
            <label className="flex items-center gap-1 text-xs text-gray-600 cursor-pointer">
              <input
                type="checkbox"
                checked={showAllIterations}
                onChange={(e) => setShowAllIterations(e.target.checked)}
                className="rounded"
              />
              Show all iterations
            </label>
          </div>

          {/* table */}
          <div className="overflow-auto max-h-48 border rounded">
            <table className="w-full text-xs">
              <thead className="bg-gray-100 sticky top-0">
                <tr>
                  <th className="px-2 py-1 text-left font-semibold text-gray-600">Iter</th>
                  <th className="px-2 py-1 text-left font-semibold text-gray-600">Type</th>
                  <th className="px-2 py-1 text-left font-semibold text-gray-600">From</th>
                  <th className="px-2 py-1 text-left font-semibold text-gray-600">To</th>
                  <th className="px-2 py-1 text-left font-semibold text-gray-600">Values</th>
                </tr>
              </thead>
              <tbody>
                {displayedMessages.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-2 py-4 text-center text-gray-400">
                      No messages for this iteration
                    </td>
                  </tr>
                ) : (
                  displayedMessages.map((row, idx) => (
                    <tr
                      key={`${row.iteration}-${row.type}-${row.from}-${row.to}-${idx}`}
                      className={`border-t hover:bg-gray-50 cursor-pointer ${
                        row.iteration === currentStep ? 'bg-blue-50/50' : ''
                      } ${
                        selectedMessage?.type === row.type &&
                        selectedMessage?.key === `${row.from}->${row.to}` &&
                        row.iteration === currentStep
                          ? 'outline outline-2 outline-emerald-400 outline-offset-[-2px]'
                          : ''
                      }`}
                      onClick={() => {
                        setCurrentStep(row.iteration);
                        setSelectedMessage({
                          type: row.type as MessageType,
                          key: `${row.from}->${row.to}`,
                        });
                      }}
                    >
                      <td className="px-2 py-1 font-mono text-gray-500">{row.iteration}</td>
                      <td className="px-2 py-1">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                            row.type === 'Q'
                              ? 'bg-blue-100 text-blue-700'
                              : 'bg-orange-100 text-orange-700'
                          }`}
                        >
                          {row.type}
                        </span>
                      </td>
                      <td className="px-2 py-1 font-mono">{row.from}</td>
                      <td className="px-2 py-1 font-mono">{row.from !== row.to ? row.to : '-'}</td>
                      <td className="px-2 py-1 font-mono text-gray-700">{formatValues(row.values)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* trace panel */}
          {selectedMessage && (
            <div className="mt-2 border rounded bg-slate-50">
              <div className="px-3 py-2 flex items-center justify-between border-b">
                <div className="text-xs font-semibold text-slate-700">
                  Trace: {selectedMessage.type} {selectedMessage.key} (viewing iter {currentStep})
                </div>
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-1 text-[11px] text-gray-600 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={traceOnlyExisting}
                      onChange={(e) => setTraceOnlyExisting(e.target.checked)}
                      className="rounded"
                    />
                    Only existing
                  </label>
                  <button
                    onClick={() => setSelectedMessage(null)}
                    className="text-[11px] px-2 py-0.5 rounded bg-white border hover:bg-gray-50"
                  >
                    Clear
                  </button>
                </div>
              </div>

              <div className="px-3 py-2 border-b text-[11px] text-gray-600">
                Graph trace semantics: Q<sub>i</sub> depends on R<sub>i-1</sub>, and R<sub>i</sub> depends on Q<sub>i</sub>.
                Click an upstream/downstream chip to traverse.
              </div>

              <div className="px-3 py-2 flex flex-wrap items-center gap-2">
                <div className="text-[11px] font-semibold text-slate-700">Upstream</div>
                {graphTrace.upstream.length === 0 ? (
                  <div className="text-[11px] text-gray-500">—</div>
                ) : (
                  graphTrace.upstream.map((m) => (
                    <button
                      key={`u-${m.type}-${m.key}-${m.iteration}`}
                      className="text-[11px] px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100"
                      onClick={() => {
                        setCurrentStep(m.iteration);
                        setSelectedMessage({ type: m.type, key: m.key });
                      }}
                      title="Jump to upstream message"
                    >
                      {m.type} {m.key} @{m.iteration}
                    </button>
                  ))
                )}
              </div>

              <div className="px-3 pb-2 flex flex-wrap items-center gap-2 border-b">
                <div className="text-[11px] font-semibold text-slate-700">Downstream</div>
                {graphTrace.downstream.length === 0 ? (
                  <div className="text-[11px] text-gray-500">—</div>
                ) : (
                  graphTrace.downstream.map((m) => (
                    <button
                      key={`d-${m.type}-${m.key}-${m.iteration}`}
                      className="text-[11px] px-2 py-0.5 rounded bg-orange-50 text-orange-700 border border-orange-200 hover:bg-orange-100"
                      onClick={() => {
                        setCurrentStep(m.iteration);
                        setSelectedMessage({ type: m.type, key: m.key });
                      }}
                      title="Jump to downstream message"
                    >
                      {m.type} {m.key} @{m.iteration}
                    </button>
                  ))
                )}
              </div>

              <div className="max-h-40 overflow-auto">
                {traceRows.length === 0 ? (
                  <div className="px-3 py-3 text-xs text-gray-500">No data for this message.</div>
                ) : (
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-slate-100">
                      <tr>
                        <th className="px-2 py-1 text-left font-semibold text-gray-600">Iter</th>
                        <th className="px-2 py-1 text-left font-semibold text-gray-600">Values</th>
                      </tr>
                    </thead>
                    <tbody>
                      {traceRows.map(({ iteration, values }) => (
                        <tr
                          key={iteration}
                          className={`border-t hover:bg-white cursor-pointer ${
                            iteration === currentStep ? 'bg-emerald-50' : ''
                          }`}
                          onClick={() => setCurrentStep(iteration)}
                          title="Jump to iteration"
                        >
                          <td className="px-2 py-1 font-mono text-gray-500">{iteration}</td>
                          <td className="px-2 py-1 font-mono text-gray-700">
                            {Array.isArray(values) ? formatValues(values) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
