import { Handle, Position } from '@xyflow/react';
import clsx from 'clsx';
import { useStore } from '../store';

export const VariableNode = ({ data }: { id: string; data: { name: string; domainSize: number } }) => {
  const currentStep = useStore((s) => s.currentStep);
  const snapshots = useStore((s) => s.snapshots);

  const snapshot = snapshots[currentStep];
  const assignment = snapshot?.assignments?.[data.name];

  return (
    <div className={clsx(
      "w-16 h-16 rounded-full flex items-center justify-center border-2 bg-white shadow-md transition-all relative",
      assignment !== undefined ? "border-blue-500 bg-blue-50" : "border-gray-400"
    )}>
      <div className="text-center">
        <div className="font-bold text-sm">{data.name}</div>
        <div className="text-xs text-gray-500">D={data.domainSize}</div>
        {assignment !== undefined && (
          <div className="absolute -top-3 -right-3 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs border border-white">
            {assignment}
          </div>
        )}
      </div>
      {/* Multiple handles to allow connections to many factors */}
      <Handle type="source" position={Position.Top} id="t" className="w-2 h-2 bg-gray-400" />
      <Handle type="source" position={Position.Bottom} id="b" className="w-2 h-2 bg-gray-400" />
      <Handle type="source" position={Position.Left} id="l" className="w-2 h-2 bg-gray-400" />
      <Handle type="source" position={Position.Right} id="r" className="w-2 h-2 bg-gray-400" />
      {/* Target handles */}
      <Handle type="target" position={Position.Top} id="tt" className="w-2 h-2 bg-gray-400" style={{ top: -4 }} />
      <Handle type="target" position={Position.Bottom} id="tb" className="w-2 h-2 bg-gray-400" style={{ bottom: -4 }} />
      <Handle type="target" position={Position.Left} id="tl" className="w-2 h-2 bg-gray-400" style={{ left: -4 }} />
      <Handle type="target" position={Position.Right} id="tr" className="w-2 h-2 bg-gray-400" style={{ right: -4 }} />
    </div>
  );
};
