import { useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw, FastForward } from 'lucide-react';

interface PlaybackControlsProps {
  currentStep: number;
  maxSteps: number;
  onStepChange: (step: number) => void;
  isPlaying: boolean;
  onPlayPauseToggle: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
  predictedTokens: number[];
  groundTruthTokens: number[];
}

export default function PlaybackControls({
  currentStep,
  maxSteps,
  onStepChange,
  isPlaying,
  onPlayPauseToggle,
  speed,
  onSpeedChange,
  predictedTokens,
  groundTruthTokens
}: PlaybackControlsProps) {
  useEffect(() => {
    if (!isPlaying) return;

    const intervalMs = Math.max(200, 1000 / speed);
    const timer = setInterval(() => {
      if (currentStep < maxSteps - 1) {
        onStepChange(currentStep + 1);
      } else {
        onPlayPauseToggle();
      }
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, currentStep, maxSteps, speed, onStepChange, onPlayPauseToggle]);

  return (
    <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 backdrop-blur-md space-y-3">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Step Info */}
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="px-2.5 py-1 bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 rounded-md font-bold">
            Step {currentStep + 1} / {maxSteps}
          </span>
          <span className="text-zinc-400">
            Active Sequence: [{predictedTokens.slice(0, currentStep + 1).join(', ')}]
          </span>
        </div>

        {/* Playback Button Group */}
        <div className="flex items-center gap-1.5 bg-zinc-950 p-1.5 rounded-lg border border-zinc-800">
          <button
            onClick={() => onStepChange(0)}
            disabled={currentStep === 0}
            title="Reset to Step 1"
            className="p-1.5 hover:bg-zinc-800 disabled:opacity-30 rounded text-zinc-300 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={() => onStepChange(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            title="Previous Step"
            className="p-1.5 hover:bg-zinc-800 disabled:opacity-30 rounded text-zinc-300 transition-colors"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          <button
            onClick={onPlayPauseToggle}
            title={isPlaying ? "Pause Rollout" : "Play Rollout"}
            className="p-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-md transition-colors shadow-[0_0_10px_rgba(99,102,241,0.4)]"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>

          <button
            onClick={() => onStepChange(Math.min(maxSteps - 1, currentStep + 1))}
            disabled={currentStep >= maxSteps - 1}
            title="Next Step"
            className="p-1.5 hover:bg-zinc-800 disabled:opacity-30 rounded text-zinc-300 transition-colors"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          <button
            onClick={() => onStepChange(maxSteps - 1)}
            disabled={currentStep >= maxSteps - 1}
            title="Jump to Final Step"
            className="p-1.5 hover:bg-zinc-800 disabled:opacity-30 rounded text-zinc-300 transition-colors"
          >
            <FastForward className="w-4 h-4" />
          </button>
        </div>

        {/* Speed Controls */}
        <div className="flex items-center gap-1 text-xs">
          <span className="text-zinc-500 mr-1 font-medium">Speed:</span>
          {[0.5, 1, 2, 5].map(s => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={`px-2 py-1 rounded text-[11px] font-mono transition-all ${
                speed === s
                  ? 'bg-indigo-500 text-white font-bold shadow-sm'
                  : 'bg-zinc-800/80 text-zinc-400 hover:text-zinc-200'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* Progress Bar Slider */}
      <div className="relative flex items-center pt-1">
        <input
          type="range"
          min={0}
          max={maxSteps - 1}
          value={currentStep}
          onChange={e => onStepChange(parseInt(e.target.value))}
          className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
        />
      </div>

      {/* Token Step Tokens Bar */}
      <div className="flex items-center gap-1 overflow-x-auto custom-scrollbar pb-1 pt-1">
        <span className="text-[10px] text-zinc-500 uppercase font-mono mr-2 shrink-0">Path Tokens:</span>
        {predictedTokens.map((tok, idx) => {
          const isActive = idx === currentStep;
          const isPassed = idx <= currentStep;
          const matchesGT = idx < groundTruthTokens.length && tok === groundTruthTokens[idx];

          return (
            <button
              key={idx}
              onClick={() => onStepChange(idx)}
              className={`px-2 py-0.5 rounded text-xs font-mono font-bold transition-all shrink-0 ${
                isActive
                  ? 'bg-indigo-500 text-white ring-2 ring-indigo-400 shadow-[0_0_8px_rgba(99,102,241,0.5)]'
                  : isPassed
                  ? matchesGT
                    ? 'bg-emerald-950/80 border border-emerald-500/50 text-emerald-300'
                    : 'bg-rose-950/80 border border-rose-500/50 text-rose-300'
                  : 'bg-zinc-900 border border-zinc-800 text-zinc-600'
              }`}
            >
              {tok}
            </button>
          );
        })}
      </div>
    </div>
  );
}
