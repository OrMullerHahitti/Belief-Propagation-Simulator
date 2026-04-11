import { useMemo } from 'react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from '@xyflow/react';
import { useStore } from '../store';

/**
 * Custom edge that displays Q and R message values based on factor's highlight mode.
 * - When mode is "both" (↔): show both Q and R messages
 * - When mode targets a specific variable:
 *   - On edges to the target: show only R (output)
 *   - On edges to other vars: show only Q (input)
 * Q messages (variable → factor) shown in blue
 * R messages (factor → variable) shown in orange
 */
export const MessageEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  source,
  target,
  markerEnd,
}: EdgeProps) => {
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const snapshots = useStore((s) => s.snapshots);
  const currentStep = useStore((s) => s.currentStep);
  const factorHighlightModes = useStore((s) => s.factorHighlightModes);
  const selectedMessage = useStore((s) => s.selectedMessage);
  const setSelectedMessage = useStore((s) => s.setSelectedMessage);

  const snapshot = snapshots[currentStep];

  // get node types and names
  const sourceNode = nodes.find((n) => n.id === source);
  const targetNode = nodes.find((n) => n.id === target);

  const sourceName = (sourceNode?.data as { name?: string })?.name || source;
  const targetName = (targetNode?.data as { name?: string })?.name || target;

  const sourceIsVar = sourceNode?.type === 'variable';
  const targetIsVar = targetNode?.type === 'variable';

  // identify which node is the factor and which is the variable
  const factorNode = sourceIsVar ? targetNode : sourceNode;
  const factorId = factorNode?.id;
  const factorName = sourceIsVar ? targetName : sourceName;
  const varName = sourceIsVar ? sourceName : targetName;

  // get connected variables for this factor (sorted, same as FactorNode)
  const connectedVars = useMemo(() => {
    if (!factorId) return [];
    return edges
      .filter((e) => e.source === factorId || e.target === factorId)
      .map((e) => {
        const otherId = e.source === factorId ? e.target : e.source;
        const otherNode = nodes.find((n) => n.id === otherId);
        return (otherNode?.data as { name?: string })?.name || otherId;
      })
      .sort();
  }, [edges, factorId, nodes]);

  // get factor's highlight mode
  const highlightMode = factorId ? (factorHighlightModes[factorId] ?? -1) : -1;

  // determine target variable name based on mode
  const targetVarName = highlightMode === -1 ? null : connectedVars[highlightMode] || null;

  // determine which direction is Q (var→factor) and R (factor→var)
  let qKey: string | null = null;
  let rKey: string | null = null;

  if (sourceIsVar && !targetIsVar) {
    // source is variable, target is factor
    qKey = `${sourceName}->${targetName}`; // Q: var → factor
    rKey = `${targetName}->${sourceName}`; // R: factor → var
  } else if (!sourceIsVar && targetIsVar) {
    // source is factor, target is variable
    qKey = `${targetName}->${sourceName}`; // Q: var → factor
    rKey = `${sourceName}->${targetName}`; // R: factor → var
  }

  // filter which messages to show based on highlight mode
  // - mode -1 (both): show all Q and R
  // - mode targets specific var: show R only on that edge, Q on others
  let showQ = true;
  let showR = true;

  if (targetVarName !== null) {
    // factor has a specific target variable selected
    if (varName === targetVarName) {
      // this edge goes to the target variable - show only R (output)
      showQ = false;
      showR = true;
    } else {
      // this edge goes to a non-target variable - show only Q (input)
      showQ = true;
      showR = false;
    }
  }

  const traceActive = !!selectedMessage;
  const selectedIsQ =
    traceActive && selectedMessage?.type === 'Q' && !!qKey && selectedMessage.key === qKey;
  const selectedIsR =
    traceActive && selectedMessage?.type === 'R' && !!rKey && selectedMessage.key === rKey;
  const isSelected = selectedIsQ || selectedIsR;

  // Graph trace highlights (within currentStep only):
  // - If selected is Q(x->f): highlight downstream R(f->y) for y != x (same step)
  // - If selected is R(f->x): highlight upstream Q(y->f) for y != x (same step)
  const { isDownstreamR, isUpstreamQ } = useMemo(() => {
    if (!traceActive || !selectedMessage) return { isDownstreamR: false, isUpstreamQ: false };
    const [src, dst] = selectedMessage.key.split('->');
    if (!src || !dst) return { isDownstreamR: false, isUpstreamQ: false };

    if (selectedMessage.type === 'Q') {
      // downstream R from the selected factor to other vars
      const isSameFactor = factorName === dst;
      const isOtherVar = varName !== src;
      return { isDownstreamR: isSameFactor && isOtherVar, isUpstreamQ: false };
    }

    // selected is R: upstream Q from other vars into the selected factor
    const isSameFactor = factorName === src;
    const isOtherVar = varName !== dst;
    const upstream = isSameFactor && isOtherVar;
    return { isDownstreamR: false, isUpstreamQ: upstream };
  }, [traceActive, selectedMessage, factorName, varName]);

  // tracing overrides: always show the traced message on its edge, even if the
  // current factor highlight mode would hide it.
  if (selectedIsQ) showQ = true;
  if (selectedIsR) showR = true;
  if (traceActive && isDownstreamR) showR = true;
  if (traceActive && isUpstreamQ) showQ = true;

  // get message values from snapshot (only if we should show them)
  const qMessage = showQ && qKey ? snapshot?.Q?.[qKey] : undefined;
  const rMessage = showR && rKey ? snapshot?.R?.[rKey] : undefined;

  // format message values for display
  const formatMessage = (msg: number[] | undefined): string => {
    if (!msg || !Array.isArray(msg)) return '';
    return `(${msg.map((v) => v.toFixed(1)).join(',')})`;
  };

  const qDisplay = formatMessage(qMessage);
  const rDisplay = formatMessage(rMessage);

  // calculate bezier path
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // calculate offset for labels (perpendicular to edge)
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const perpX = -dy / len;
  const perpY = dx / len;
  const offset = 12; // pixels offset from edge center

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={
          isSelected
            ? { stroke: '#10b981', strokeWidth: 2.5 } // emerald-500
            : traceActive && isDownstreamR
              ? { stroke: '#f97316', strokeWidth: 2.2 } // orange-500
              : traceActive && isUpstreamQ
                ? { stroke: '#3b82f6', strokeWidth: 2.2 } // blue-500
                : undefined
        }
      />
      <EdgeLabelRenderer>
        {/* Q message label (blue) - above/left of edge */}
        {qDisplay && (
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX + perpX * offset}px, ${labelY + perpY * offset}px)`,
              pointerEvents: 'all',
            }}
            className={`text-[9px] font-mono bg-blue-100 text-blue-700 px-1 rounded border border-blue-300 whitespace-nowrap ${
              selectedIsQ ? 'ring-2 ring-emerald-400' : traceActive && isUpstreamQ ? 'ring-2 ring-blue-400' : ''
            }`}
            title={`Q: ${qKey}`}
            onClick={(e) => {
              e.stopPropagation();
              if (qKey) setSelectedMessage({ type: 'Q', key: qKey });
            }}
          >
            Q{qDisplay}
          </div>
        )}
        {/* R message label (orange) - below/right of edge */}
        {rDisplay && (
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX - perpX * offset}px, ${labelY - perpY * offset}px)`,
              pointerEvents: 'all',
            }}
            className={`text-[9px] font-mono bg-orange-100 text-orange-700 px-1 rounded border border-orange-300 whitespace-nowrap ${
              selectedIsR
                ? 'ring-2 ring-emerald-400'
                : traceActive && isDownstreamR
                  ? 'ring-2 ring-orange-400'
                  : ''
            }`}
            title={`R: ${rKey}`}
            onClick={(e) => {
              e.stopPropagation();
              if (rKey) setSelectedMessage({ type: 'R', key: rKey });
            }}
          >
            R{rDisplay}
          </div>
        )}
      </EdgeLabelRenderer>
    </>
  );
};
