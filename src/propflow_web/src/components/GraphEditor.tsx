import { useCallback, useMemo } from 'react';
import { ReactFlow, Background, Controls, MiniMap, type OnSelectionChangeParams } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useStore } from '../store';
import { VariableNode } from './VariableNode';
import { FactorNode } from './FactorNode';
import { MessageEdge } from './MessageEdge';

const nodeTypes = {
  variable: VariableNode,
  factor: FactorNode,
};

const edgeTypes = {
  message: MessageEdge,
};

export const GraphEditor = () => {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, setSelectedNode } = useStore();

  // apply message edge type to all edges
  const edgesWithType = useMemo(
    () => edges.map((e) => ({ ...e, type: 'message' })),
    [edges]
  );

  const handleSelectionChange = useCallback(({ nodes }: OnSelectionChangeParams) => {
     if (nodes.length > 0) {
        setSelectedNode(nodes[0].id);
     } else {
        setSelectedNode(null);
     }
  }, [setSelectedNode]);

  return (
    <div className="h-full w-full bg-slate-50">
      <ReactFlow
        nodes={nodes}
        edges={edgesWithType}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={handleSelectionChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
      >
        <Background color="#ccc" gap={20} />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
};
