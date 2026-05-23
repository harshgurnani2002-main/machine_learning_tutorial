import React, { useState, useEffect } from 'react';
import { ArrowRight, MessageSquare, RefreshCcw } from 'lucide-react';

// Sequence of words
const sequence = ['I', 'love', 'machine', 'learning'];
// Mock word embeddings (just a single scalar for visual simplicity)
const embeddings = [0.8, 0.9, 0.4, 0.7];

export const RNNSimulator: React.FC = () => {
  const [timeStep, setTimeStep] = useState(0); // 0 to 4
  const [hiddenStates, setHiddenStates] = useState<number[]>([0]); // starts with h0 = 0
  const [outputs, setOutputs] = useState<number[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);

  // RNN Weights
  const W_hx = 0.5; // Input weight
  const W_hh = 0.8; // Hidden weight (memory)
  const W_hy = 1.2; // Output weight

  const reset = () => {
    setTimeStep(0);
    setHiddenStates([0]);
    setOutputs([]);
    setIsPlaying(false);
  };

  const stepForward = () => {
    if (timeStep < sequence.length) {
      const x_t = embeddings[timeStep];
      const h_prev = hiddenStates[timeStep];
      
      // h_t = tanh(W_hh * h_{t-1} + W_hx * x_t)
      const h_t = Math.tanh(W_hh * h_prev + W_hx * x_t);
      // y_t = W_hy * h_t
      const y_t = W_hy * h_t;

      setHiddenStates(prev => [...prev, h_t]);
      setOutputs(prev => [...prev, y_t]);
      setTimeStep(prev => prev + 1);
    } else {
      setIsPlaying(false);
    }
  };

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    if (isPlaying && timeStep < sequence.length) {
      timer = setTimeout(() => {
        stepForward();
      }, 1000);
    } else if (timeStep >= sequence.length) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, timeStep]);

  // Auto-start on mount for one-click experience
  useEffect(() => {
    const timer = setTimeout(() => setIsPlaying(true), 600);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE] overflow-y-auto">
      <div className="lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-start shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <MessageSquare className="w-6 h-6 text-[#B6532B]" /> Recurrent Neural Network
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            RNNs process sequential data one step at a time. They maintain a <span className="font-semibold text-[#B6532B]">Hidden State</span> ({"$h_t$"}) which acts as the network's memory, combining the current input ({"$x_t$"}) with the previous state ({"$h_{t-1}$"}).
          </p>

          <div className="bg-[#FAF6EE] p-4 rounded-xl border border-[#E5DDD0] space-y-3 font-mono text-xs">
            <h5 className="font-bold text-[#2E251E] mb-2 uppercase tracking-wide">RNN Equations</h5>
            <div className="flex justify-between items-center bg-[#F4EFE6] px-3 py-2 rounded">
              <span className="text-[#6E6257]">Hidden State</span>
              <span className="text-[#B6532B] font-bold">h_t = tanh(W_hh·h_t-1 + W_hx·x_t)</span>
            </div>
            <div className="flex justify-between items-center bg-[#F4EFE6] px-3 py-2 rounded">
              <span className="text-[#6E6257]">Output</span>
              <span className="text-[#3B7A57] font-bold">y_t = W_hy·h_t</span>
            </div>
          </div>

          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4">
             <div className="flex justify-between items-center text-xs font-mono mb-2">
                <span className="text-[#6E6257] font-bold">Time Step (t)</span>
                <span className="text-[#2E251E] font-bold bg-[#E5DDD0] px-2 py-1 rounded w-8 text-center">{timeStep}</span>
              </div>
          </div>

          <div className="flex gap-2 mt-6">
            <button
              onClick={() => {
                if (timeStep >= sequence.length) reset();
                else setIsPlaying(!isPlaying);
              }}
              className={`flex-1 py-3 rounded-xl flex items-center justify-center gap-2 text-sm font-bold transition-all shadow-sm ${
                isPlaying
                  ? 'bg-amber-600 border-amber-600 text-white'
                  : 'bg-[#B6532B] border-[#B6532B] text-white hover:bg-[#B6532B]/90'
              }`}
            >
              {timeStep >= sequence.length ? 'Restart' : isPlaying ? 'Pause' : 'Auto Step'}
            </button>
            <button
              onClick={stepForward}
              disabled={timeStep >= sequence.length || isPlaying}
              className="p-3 px-6 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#B6532B] hover:text-[#2E251E] font-bold text-sm transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              Step <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={reset}
              className="p-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] transition-colors"
            >
              <RefreshCcw className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      <div className="lg:col-span-7 flex items-center justify-center overflow-x-auto pb-4">
        <div className="flex gap-6 min-w-max items-end">
          
          {/* Unrolled RNN Visualization */}
          {sequence.map((word, t) => {
            const isActive = t === timeStep;
            const isProcessed = t < timeStep;
            
            return (
              <div key={t} className={`flex flex-col items-center transition-all duration-500 ${isProcessed ? 'opacity-100' : isActive ? 'opacity-100 scale-105' : 'opacity-30'}`}>
                
                {/* Output */}
                <div className="flex flex-col items-center mb-6">
                  <div className={`text-[10px] font-mono font-bold mb-1 ${isProcessed ? 'text-[#3B7A57]' : 'text-transparent'}`}>
                    y_{t} = {outputs[t]?.toFixed(2) || '0.00'}
                  </div>
                  <div className={`w-12 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white transition-colors duration-500 ${
                    isProcessed ? 'bg-[#3B7A57]' : 'bg-[#E5DDD0]'
                  }`}>
                    Out
                  </div>
                  {/* Arrow up to output */}
                  <div className={`w-0.5 h-6 transition-colors duration-500 ${isProcessed ? 'bg-[#3B7A57]' : 'bg-[#E5DDD0]'}`}></div>
                </div>

                {/* RNN Cell (Hidden State) */}
                <div className="relative flex items-center">
                  {/* Arrow from prev state */}
                  {t > 0 && (
                    <div className="absolute right-full w-6 h-0.5 bg-[#B6532B] flex items-center justify-end">
                      <div className="w-2 h-2 border-t-2 border-r-2 border-[#B6532B] transform rotate-45 mr-0.5" />
                    </div>
                  )}

                  <div className={`w-16 h-16 rounded-xl flex flex-col items-center justify-center font-mono text-[10px] text-white shadow-lg border-2 transition-colors duration-500 z-10 ${
                    isActive ? 'bg-[#C18C3B] border-white shadow-xl' : isProcessed ? 'bg-[#B6532B] border-[#B6532B]' : 'bg-[#E5DDD0] border-[#E5DDD0]'
                  }`}>
                    <span>h_{t}</span>
                    <span className="font-bold text-xs mt-1">{(hiddenStates[t+1] !== undefined ? hiddenStates[t+1] : 0).toFixed(2)}</span>
                  </div>
                </div>

                {/* Arrow up from input */}
                <div className={`w-0.5 h-6 transition-colors duration-500 ${isProcessed || isActive ? 'bg-[#C18C3B]' : 'bg-[#E5DDD0]'}`}></div>

                {/* Input */}
                <div className="flex flex-col items-center">
                  <div className={`w-16 py-2 rounded flex flex-col items-center border transition-colors duration-500 ${
                    isProcessed || isActive ? 'bg-white border-[#C18C3B] text-[#2E251E]' : 'bg-[#FAF6EE] border-[#E5DDD0] text-[#6E6257]'
                  }`}>
                    <span className="text-sm font-bold">"{word}"</span>
                    <span className="text-[9px] font-mono opacity-60">x_{t} = {embeddings[t]}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
