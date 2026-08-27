import { useState, useMemo } from 'react';
import { Brain, Cpu, Sparkles, Compass, Flame, Lock, Grid } from 'lucide-react';
import {
  GRAPH_SAMPLES,
  GraphSample,
  runFullRolloutInference,
  FullInferenceTrace
} from './model/graph_transformer';
import PlaybackControls from './components/PlaybackControls';
import GraphVisualizer from './components/GraphVisualizer';
import AttentionHeatmap from './components/AttentionHeatmap';
import ModelComparer from './components/ModelComparer';
import LogitInspector from './components/LogitInspector';
import InferencePipelineInspector, { PipelineStage } from './components/InferencePipelineInspector';
import WeightsInspector from './components/WeightsInspector';
import VocabularyEmbeddingsInspector from './components/VocabularyEmbeddingsInspector';

type MainTab = 'visualizer' | 'comparison' | 'attention' | 'weights' | 'vocab';

export default function App() {
  const [selectedSample, setSelectedSample] = useState<GraphSample>(GRAPH_SAMPLES[0]);
  const [modelEpoch, setModelEpoch] = useState<"300" | "400" | "500">("500");
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1);
  const [activeTab, setActiveTab] = useState<MainTab>('visualizer');
  const [pipelineActiveStage, setPipelineActiveStage] = useState<PipelineStage>('classifier');

  // Attention heatmap states
  const [selectedLayer, setSelectedLayer] = useState<number>(1); // Layer 2 default
  const [selectedHead, setSelectedHead] = useState<number>(0);   // Head 0 default
  const [attentionType, setAttentionType] = useState<'cross' | 'decoder_self' | 'encoder_self'>('cross');

  // Run full rollout inference for models on selected sample
  const trace300: FullInferenceTrace = useMemo(() => {
    return runFullRolloutInference("300", selectedSample);
  }, [selectedSample]);

  const trace400: FullInferenceTrace = useMemo(() => {
    return runFullRolloutInference("400", selectedSample);
  }, [selectedSample]);

  const trace500: FullInferenceTrace = useMemo(() => {
    return runFullRolloutInference("500", selectedSample);
  }, [selectedSample]);

  const activeTrace = modelEpoch === "300" ? trace300 : modelEpoch === "400" ? trace400 : trace500;
  const maxSteps = activeTrace.steps.length;

  // Handle sample selection
  const handleSampleSelect = (sample: GraphSample) => {
    setSelectedSample(sample);
    setCurrentStep(0);
    setIsPlaying(false);
  };

  // Safe current step trace
  const safeStepIdx = Math.min(currentStep, maxSteps - 1);
  const activeStepTrace = activeTrace.steps[safeStepIdx] || activeTrace.steps[0];

  const showGraphSelection = activeTab === 'visualizer' || activeTab === 'comparison' || activeTab === 'attention';

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200 relative pb-28">
      {/* App Header (Unfrozen) */}
      <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-indigo-500 via-cyan-500 to-emerald-600 rounded-xl shadow-[0_0_15px_rgba(99,102,241,0.3)]">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-md font-bold tracking-tight bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
              Graph Shortest Path Transformer Visualizer
            </h1>
            <p className="text-[10px] text-zinc-500 font-medium">
              Autoregressive Graph Transformer (vocab=42, embed_dim=16, num_heads=2, hidden_dim=32, num_layers=2)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('comparison')}
            title="Click to view Checkpoints Comparison (300 / 400 / 500)"
            className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-500/10 border border-indigo-500/25 text-indigo-400 hover:bg-indigo-500/20 hover:border-indigo-500/40 transition-all flex items-center gap-1 font-mono cursor-pointer"
          >
            <Cpu className="w-3 h-3 text-indigo-400 animate-pulse" /> Seq2Seq AR Transformer (Checkpoints 300/400/500)
          </button>
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 font-mono">
            Epoch 500 Exact Match: 96.4%
          </span>
        </div>
      </header>

      {/* Navigation Tabs (Unfrozen) */}
      <div className="bg-zinc-950/60 border-b border-zinc-900 px-6 py-2.5 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setActiveTab('visualizer')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
                activeTab === 'visualizer'
                  ? 'bg-indigo-500 text-white shadow-[0_0_12px_rgba(99,102,241,0.3)] font-bold'
                  : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
              }`}
            >
              <Compass className="w-3.5 h-3.5" /> Step Solver &amp; Graph Layout
            </button>

            <button
              onClick={() => setActiveTab('attention')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
                activeTab === 'attention'
                  ? 'bg-violet-500 text-white shadow-[0_0_12px_rgba(139,92,246,0.3)] font-bold'
                  : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
              }`}
            >
              <Flame className="w-3.5 h-3.5" /> Attention Heads Visualizer
            </button>

            <button
              onClick={() => setActiveTab('weights')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
                activeTab === 'weights'
                  ? 'bg-cyan-500 text-zinc-950 shadow-[0_0_12px_rgba(6,182,212,0.3)] font-bold'
                  : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
              }`}
            >
              <Lock className="w-3.5 h-3.5" /> Frozen Model Weights
            </button>

            <button
              onClick={() => setActiveTab('vocab')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
                activeTab === 'vocab'
                  ? 'bg-indigo-500 text-white shadow-[0_0_12px_rgba(99,102,241,0.3)] font-bold'
                  : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
              }`}
            >
              <Grid className="w-3.5 h-3.5" /> Vocabulary Embeddings
            </button>
          </div>

          {/* Model Epoch Active Toggle */}
          <div className="flex items-center gap-2 text-xs font-mono bg-zinc-900 p-1 rounded-lg border border-zinc-800">
            <span className="text-zinc-500 px-1 text-[11px]">Active Checkpoint:</span>
            {(["300", "400", "500"] as const).map(ep => (
              <button
                key={ep}
                onClick={() => setModelEpoch(ep)}
                className={`px-2.5 py-0.5 rounded transition-all ${
                  modelEpoch === ep
                    ? ep === "300"
                      ? 'bg-amber-500 text-zinc-950 font-bold'
                      : ep === "400"
                      ? 'bg-emerald-500 text-zinc-950 font-bold'
                      : 'bg-indigo-500 text-white font-bold'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                Epoch {ep}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Sample Selection Bar - Only render when graph input selection is applicable */}
        {showGraphSelection && (
          <section className="bg-zinc-900/40 border border-zinc-900 rounded-xl p-4 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Select Graph Sample Preset
                </h3>
                <p className="text-[11px] text-zinc-400 mt-0.5">
                  Select test set graph samples with varying DFS trace lengths (K=15..22) and target shortest path lengths (M=4..9).
                </p>
              </div>
              <div className="text-xs font-mono text-zinc-400 bg-zinc-950 px-3 py-1 rounded border border-zinc-800">
                Active: <strong className="text-indigo-400">Sample #{selectedSample.id}</strong> (Nodes={selectedSample.nodes.length}, Edges={selectedSample.edges.length})
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-2">
              {GRAPH_SAMPLES.slice(0, 10).map(s => {
                const isSelected = selectedSample.id === s.id;
                return (
                  <button
                    key={s.id}
                    onClick={() => handleSampleSelect(s)}
                    className={`p-2.5 rounded-lg border text-left font-mono transition-all ${
                      isSelected
                        ? 'bg-indigo-950/60 border-indigo-500 text-indigo-200 shadow-[0_0_10px_rgba(99,102,241,0.2)]'
                        : 'bg-zinc-950 border-zinc-850 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs font-bold mb-1">
                      <span>Sample #{s.id}</span>
                      {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />}
                    </div>
                    <div className="text-[10px] text-zinc-500 space-y-0.5">
                      <p>Trace K={s.trace.length} | SP M={s.sp.length}</p>
                      <p>Backtracks: {s.backtracks}</p>
                    </div>
                  </button>
                );
              })}
            </div>
          </section>
        )}

        {/* Tab 1: Visualizer */}
        {activeTab === 'visualizer' && (
          <div className="space-y-6">
            <GraphVisualizer
              sample={selectedSample}
              currentStep={safeStepIdx}
              predictedSP={activeTrace.predictedSP}
              groundTruthSP={activeTrace.groundTruthSP}
              modelEpoch={modelEpoch}
            />

            <InferencePipelineInspector
              stepTrace={activeStepTrace}
              groundTruthSP={activeTrace.groundTruthSP}
              currentStep={safeStepIdx}
              activeStage={pipelineActiveStage}
              onStageChange={setPipelineActiveStage}
              selectedEpoch={modelEpoch}
              onSelectEpoch={setModelEpoch}
            />

            <LogitInspector
              stepTrace={activeStepTrace}
              groundTruthSP={activeTrace.groundTruthSP}
              currentStep={safeStepIdx}
            />
          </div>
        )}

        {/* Checkpoints Tab (Opened by clicking Seq2Seq AR Transformer badge) */}
        {activeTab === 'comparison' && (
          <ModelComparer
            sample={selectedSample}
            trace300={trace300}
            trace400={trace400}
            trace500={trace500}
            activeModelEpoch={modelEpoch}
            onSelectModelEpoch={setModelEpoch}
          />
        )}

        {/* Tab 3: Attention Heatmaps */}
        {activeTab === 'attention' && (
          <AttentionHeatmap
            stepTrace={activeStepTrace}
            sample={selectedSample}
            selectedLayer={selectedLayer}
            onSelectLayer={setSelectedLayer}
            selectedHead={selectedHead}
            onSelectHead={setSelectedHead}
            attentionType={attentionType}
            onSelectType={setAttentionType}
          />
        )}

        {/* Tab 4: Frozen Model Weights */}
        {activeTab === 'weights' && (
          <WeightsInspector
            selectedEpoch={modelEpoch}
            onSelectEpoch={setModelEpoch}
            activeStepTrace={activeStepTrace}
          />
        )}

        {/* Tab 5: Vocabulary Embeddings */}
        {activeTab === 'vocab' && (
          <VocabularyEmbeddingsInspector
            selectedEpoch={modelEpoch}
            onSelectEpoch={setModelEpoch}
          />
        )}
      </main>

      {/* Frozen Playback Controls at Bottom */}
      <div className="fixed bottom-0 left-0 right-0 z-40 bg-zinc-950/95 border-t border-zinc-800 px-6 py-2.5 backdrop-blur-md shadow-[0_-5px_25px_rgba(0,0,0,0.8)]">
        <div className="max-w-7xl mx-auto">
          <PlaybackControls
            currentStep={safeStepIdx}
            maxSteps={maxSteps}
            onStepChange={setCurrentStep}
            isPlaying={isPlaying}
            onPlayPauseToggle={() => setIsPlaying(!isPlaying)}
            speed={speed}
            onSpeedChange={setSpeed}
            predictedTokens={activeTrace.predictedSP}
            groundTruthTokens={activeTrace.groundTruthSP}
          />
        </div>
      </div>

      <footer className="border-t border-zinc-900 bg-zinc-950/60 py-6 px-6 text-center text-xs text-zinc-500 font-mono">
        <p className="max-w-2xl mx-auto leading-relaxed">
          This Graph Shortest Path Transformer Visualizer illustrates how seq2seq cross-attention mechanisms parse 1D Depth-First Search (DFS) traces to extract direct shortest paths without label memorization.
        </p>
        <p className="mt-2 text-[10px] text-zinc-600">
          Built with React + Vite + TypeScript. Deployed to Netlify.
        </p>
      </footer>
    </div>
  );
}
