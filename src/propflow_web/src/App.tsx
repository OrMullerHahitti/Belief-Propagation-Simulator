import { useEffect, useState } from 'react';
import { GraphEditor } from './components/GraphEditor';
import { Inspector } from './components/Inspector';
import { MessageTable } from './components/MessageTable';
import { useStore } from './store';
import type { DampingMode, VariableData } from './store';
import { Play, Square, Plus, AlertTriangle, X } from 'lucide-react';
import axios from 'axios';

function App() {
  const {
    addVariable, addFactor,
    nodes, edges,
    setSnapshots,
    currentStep, snapshots, isPlaying, togglePlay, setCurrentStep, runId,
    dampingFactor, dampingMode, setDampingFactor, setDampingMode, maxIters, setMaxIters
  } = useStore();

  const [errors, setErrors] = useState<string[]>([]);

  // validate graph before hitting the backend: every node must be wired,
  // and the graph must form a single connected component (current solver limit)
  const validateGraph = (): string[] => {
    const problems: string[] = [];
    const variables = nodes.filter(n => n.type === 'variable');
    const factors = nodes.filter(n => n.type === 'factor');

    if (variables.length === 0) problems.push('Add at least one variable.');
    if (factors.length === 0) problems.push('Add at least one factor.');
    if (problems.length > 0) return problems;

    const adj = new Map<string, string[]>();
    nodes.forEach(n => adj.set(n.id, []));
    edges.forEach(e => {
      adj.get(e.source)?.push(e.target);
      adj.get(e.target)?.push(e.source);
    });

    const orphans = nodes.filter(n => (adj.get(n.id)?.length ?? 0) === 0);
    if (orphans.length > 0) {
      const names = orphans.map(n => (n.data as { name: string }).name).join(', ');
      const plural = orphans.length > 1;
      problems.push(
        `Unconnected ${plural ? 'nodes' : 'node'}: ${names}. Wire ${plural ? 'them' : 'it'} into the graph before running.`
      );
      return problems;
    }

    // bfs from any node; if we can't reach everyone, graph has multiple components
    const visited = new Set<string>();
    const queue: string[] = [nodes[0].id];
    while (queue.length) {
      const id = queue.shift()!;
      if (visited.has(id)) continue;
      visited.add(id);
      adj.get(id)?.forEach(nb => { if (!visited.has(nb)) queue.push(nb); });
    }
    if (visited.size !== nodes.length) {
      problems.push(
        'The graph has disconnected components. The solver currently requires a single connected component — merge them with a shared factor.'
      );
    }

    return problems;
  };

  // handler to run simulation
  const handleRun = async () => {
    const problems = validateGraph();
    if (problems.length > 0) {
      setErrors(problems);
      return;
    }
    setErrors([]);

    // build variable specs including unary costs
    const variables = nodes.filter(n => n.type === 'variable').map(n => {
      const data = n.data as unknown as VariableData;
      return {
        name: data.name,
        domain_size: data.domainSize,
        unary_cost: data.unaryCost || null
      };
    });

    // build factor specs from edges
    const factors = nodes.filter(n => n.type === 'factor').map(n => {
      const connectedEdges = edges.filter(e => e.source === n.id || e.target === n.id);
      const neighborIds = connectedEdges.map(e => e.source === n.id ? e.target : e.source);
      const neighbors = neighborIds.map(nid => nodes.find(node => node.id === nid)?.data.name);

      return {
        name: n.data.name,
        neighbors: neighbors,
        cost_table: n.data.costTable
      };
    });

    const spec = {
      variables,
      factors,
      config: {
        max_iters: maxIters,
        engine_type: 'min_sum',
        damping: (dampingMode === 'q' || dampingMode === 'both') ? dampingFactor : 0,
        r_damping: (dampingMode === 'r' || dampingMode === 'both') ? dampingFactor : 0,
      }
    };

    try {
      const res = await axios.post('/api/run', spec);
      setSnapshots(res.data.run_id, res.data.snapshots);
    } catch (e) {
      console.error(e);
      let detail = 'Simulation failed. Check the console for details.';
      if (axios.isAxiosError(e)) {
        const backendDetail = e.response?.data?.detail;
        if (typeof backendDetail === 'string' && backendDetail.length > 0) {
          detail = `Backend error: ${backendDetail}`;
        } else if (e.message) {
          detail = `Request failed: ${e.message}`;
        }
      }
      setErrors([detail]);
    }
  };
  
  // simple step timer for playback
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isPlaying && snapshots.length > 0) {
      timer = setInterval(() => {
        setCurrentStep(Math.min(currentStep + 1, snapshots.length - 1));
      }, 500);
    }
    return () => clearInterval(timer);
  }, [isPlaying, currentStep, snapshots.length, setCurrentStep]);

  return (
    <div className="flex h-screen w-screen flex-col">
      {/* Header */}
      <div className="h-14 border-b bg-white px-4 flex items-center justify-between shadow-sm z-10">
        <h1 className="font-bold text-xl text-slate-800">PropFlow Web</h1>
        
        <div className="flex gap-2">
           <button onClick={addVariable} className="flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200">
             <Plus size={16}/> Var
           </button>
           <button onClick={addFactor} className="flex items-center gap-1 px-3 py-1 bg-orange-100 text-orange-700 rounded hover:bg-orange-200">
             <Plus size={16}/> Factor
           </button>
           <div className="w-px h-6 bg-gray-300 mx-2"/>
           <button onClick={handleRun} className="px-4 py-1 bg-slate-900 text-white rounded hover:bg-slate-700 font-medium">
             Run Simulation
           </button>
        </div>
      </div>

      {/* Error banner */}
      {errors.length > 0 && (
        <div className="border-b border-amber-300 bg-amber-50 px-4 py-2 flex items-start gap-3">
          <AlertTriangle size={18} className="text-amber-600 mt-0.5 shrink-0" />
          <div className="flex-1 text-sm text-amber-900">
            <div className="font-semibold mb-0.5">Can't run simulation</div>
            <ul className={errors.length > 1 ? 'list-disc pl-5 space-y-0.5' : ''}>
              {errors.map((msg, i) => <li key={i} className={errors.length > 1 ? '' : 'list-none'}>{msg}</li>)}
            </ul>
          </div>
          <button
            onClick={() => setErrors([])}
            className="text-amber-700 hover:text-amber-900 shrink-0"
            title="Dismiss"
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Graph Canvas + Message Table */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 relative">
            <GraphEditor />

            {/* Overlay Stats */}
            {runId && (
              <div className="absolute top-4 left-4 bg-white/90 p-3 rounded-lg shadow backdrop-blur-sm border">
                <div className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Iteration {currentStep}</div>
                <div className="text-2xl font-bold text-slate-800">
                  Cost: {snapshots[currentStep]?.global_cost?.toFixed(3) ?? 'N/A'}
                </div>
              </div>
            )}
          </div>

          {/* Message Table */}
          <MessageTable />
        </div>
        
        {/* Right: Inspector / Controls */}
        <div className="w-80 border-l bg-white flex flex-col">
           <div className="p-4 border-b space-y-3">
             <h2 className="font-semibold">Simulation Controls</h2>

             {/* Playback controls */}
             <div className="flex items-center gap-2 justify-center bg-slate-50 p-2 rounded">
               <button onClick={togglePlay} className="p-2 rounded hover:bg-white shadow-sm">
                 {isPlaying ? <Square size={20}/> : <Play size={20}/>}
               </button>
               <input
                 type="range"
                 min={0}
                 max={snapshots.length - 1 || 0}
                 value={currentStep}
                 onChange={(e) => setCurrentStep(Number(e.target.value))}
                 className="flex-1"
               />
               <span className="text-sm font-mono w-8 text-center">{currentStep}</span>
             </div>

             {/* Damping control */}
             <div>
               <div className="flex items-center justify-between mb-1">
                 <label className="text-xs font-semibold uppercase text-gray-500">Damping</label>
                 <div className="flex items-center gap-2">
                   <select
                     value={dampingMode}
                     onChange={(e) => setDampingMode(e.target.value as DampingMode)}
                     className="text-[11px] px-1 py-0.5 border rounded bg-white"
                     title="Which messages to damp"
                   >
                     <option value="q">Q (Var→Factor)</option>
                     <option value="r">R (Factor→Var)</option>
                     <option value="both">Both</option>
                   </select>
                   <span className="text-xs font-mono text-gray-600">{dampingFactor.toFixed(2)}</span>
                 </div>
               </div>
               <input
                 type="range"
                 min={0}
                 max={0.99}
                 step={0.01}
                 value={dampingFactor}
                 onChange={(e) => setDampingFactor(Number(e.target.value))}
                 className="w-full"
               />
               <div className="flex justify-between text-[10px] text-gray-400">
                 <span>None</span>
                 <span>Heavy</span>
               </div>
             </div>

             {/* Max iterations */}
             <div>
               <label className="text-xs font-semibold uppercase text-gray-500 block mb-1">Max Iterations</label>
               <input
                 type="number"
                 min={1}
                 max={1000}
                 value={maxIters}
                 onChange={(e) => setMaxIters(Number(e.target.value))}
                 className="w-full px-2 py-1 border rounded text-sm"
               />
             </div>
           </div>
           
           <div className="flex-1 p-0 overflow-hidden flex flex-col">
             {/* Inspector Component */}
             <Inspector />
           </div>
        </div>
      </div>
    </div>
  );
}

export default App;
