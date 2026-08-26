import { useState } from 'react';
import modelsWeightsData from '../data/models_weights.json';
import { Database } from 'lucide-react';

export default function ParameterInspector() {
  const weights300 = (modelsWeightsData as any)["300"];
  const paramKeys = Object.keys(weights300 || {});

  const [selectedKey, setSelectedKey] = useState<string>(paramKeys[0] || 'token_embedding.weight');

  const paramData = weights300?.[selectedKey];

  const flatten = (arr: any): number[] => {
    if (!Array.isArray(arr)) return [Number(arr)];
    return arr.flatMap(flatten);
  };

  const flatValues = paramData ? flatten(paramData) : [];
  const count = flatValues.length;
  const mean = count > 0 ? flatValues.reduce((a, b) => a + b, 0) / count : 0;
  const min = count > 0 ? Math.min(...flatValues) : 0;
  const max = count > 0 ? Math.max(...flatValues) : 0;

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-3">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <Database className="w-5 h-5 text-violet-400" />
            Network Parameter Inspector (13,802 Parameters)
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Deep-dive weight matrix inspector across Token Embeddings, Multi-Head Attention, GELU FFNs, and FC Classifier.
          </p>
        </div>

        <div className="text-xs font-mono text-zinc-400 bg-zinc-950 px-3 py-1 rounded border border-zinc-800">
          Total Parameters: <strong className="text-violet-400">13,802</strong>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Parameter List Tree */}
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 space-y-1.5 max-h-[380px] overflow-y-auto custom-scrollbar font-mono text-xs">
          <p className="text-[10px] uppercase font-bold text-zinc-500 mb-2 px-1">
            Parameter Tensors ({paramKeys.length}):
          </p>
          {paramKeys.map(key => {
            const isSelected = selectedKey === key;
            return (
              <button
                key={key}
                onClick={() => setSelectedKey(key)}
                className={`w-full text-left px-2.5 py-1.5 rounded transition-all truncate block ${
                  isSelected
                    ? 'bg-violet-600 text-white font-bold shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
                }`}
              >
                {key}
              </button>
            );
          })}
        </div>

        {/* Selected Parameter Details & Values */}
        <div className="md:col-span-2 space-y-3 font-mono">
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between border-b border-zinc-850 pb-2">
              <span className="text-xs font-bold text-violet-300">{selectedKey}</span>
              <span className="text-[11px] text-zinc-400">Elements: {count}</span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs pt-1">
              <div>
                <p className="text-[10px] text-zinc-500">Min Weight:</p>
                <p className="font-bold text-zinc-200">{min.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[10px] text-zinc-500">Mean Weight:</p>
                <p className="font-bold text-zinc-200">{mean.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[10px] text-zinc-500">Max Weight:</p>
                <p className="font-bold text-zinc-200">{max.toFixed(4)}</p>
              </div>
            </div>
          </div>

          {/* Matrix Preview Grid */}
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 max-h-[260px] overflow-auto custom-scrollbar text-[10px]">
            <p className="text-zinc-500 mb-2">Weight Value Tensor Snippet:</p>
            <pre className="text-violet-300 whitespace-pre-wrap leading-relaxed">
              {JSON.stringify(paramData, null, 2)?.slice(0, 1500)}...
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
