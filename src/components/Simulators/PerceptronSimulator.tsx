import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, AlertTriangle, Cpu } from 'lucide-react';

type DatasetType = 'AND' | 'OR' | 'XOR';

const datasets: Record<DatasetType, Array<{x1: number, x2: number, label: number}>> = {
  AND: [
    { x1: 0, x2: 0, label: 0 },
    { x1: 0, x2: 1, label: 0 },
    { x1: 1, x2: 0, label: 0 },
    { x1: 1, x2: 1, label: 1 },
  ],
  OR: [
    { x1: 0, x2: 0, label: 0 },
    { x1: 0, x2: 1, label: 1 },
    { x1: 1, x2: 0, label: 1 },
    { x1: 1, x2: 1, label: 1 },
  ],
  XOR: [
    { x1: 0, x2: 0, label: 0 },
    { x1: 0, x2: 1, label: 1 },
    { x1: 1, x2: 0, label: 1 },
    { x1: 1, x2: 1, label: 0 },
  ],
};

export const PerceptronSimulator: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [datasetType, setDatasetType] = useState<DatasetType>('AND');
  const [learningRate, setLearningRate] = useState(0.1);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // Weights w1, w2 and bias b
  const [w1, setW1] = useState(Math.random() * 2 - 1);
  const [w2, setW2] = useState(Math.random() * 2 - 1);
  const [b, setB] = useState(Math.random() * 2 - 1);
  const [epoch, setEpoch] = useState(0);
  const [accuracy, setAccuracy] = useState(0);

  const initWeights = () => {
    setW1(Math.random() * 2 - 1);
    setW2(Math.random() * 2 - 1);
    setB(Math.random() * 2 - 1);
    setEpoch(0);
    setAccuracy(0);
    setIsPlaying(false);
  };

  // Change dataset resets weights
  useEffect(() => {
    initWeights();
  }, [datasetType]);

  // Auto-start training on mount for one-click experience
  useEffect(() => {
    const timer = setTimeout(() => setIsPlaying(true), 700);
    return () => clearTimeout(timer);
  }, []);

  const trainStep = () => {
    let nw1 = w1;
    let nw2 = w2;
    let nb = b;
    let correct = 0;
    
    const data = datasets[datasetType];
    
    data.forEach(pt => {
      // Step function activation
      const z = pt.x1 * nw1 + pt.x2 * nw2 + nb;
      const prediction = z >= 0 ? 1 : 0;
      const error = pt.label - prediction;
      
      if (error === 0) correct++;
      
      nw1 += learningRate * error * pt.x1;
      nw2 += learningRate * error * pt.x2;
      nb += learningRate * error;
    });

    setW1(nw1);
    setW2(nw2);
    setB(nb);
    setAccuracy((correct / data.length) * 100);
    setEpoch(prev => prev + 1);
    
    if (correct === data.length) {
      setIsPlaying(false);
    }
  };

  useEffect(() => {
    let animId: number;
    if (isPlaying) {
      const run = () => {
        trainStep();
        animId = requestAnimationFrame(run);
      };
      // Slow down the perceptron a bit so we can watch it jump around
      const timeout = setTimeout(() => {
         animId = requestAnimationFrame(run);
      }, 100);
      return () => {
         clearTimeout(timeout);
         cancelAnimationFrame(animId);
      };
    }
  }, [isPlaying, w1, w2, b, datasetType, learningRate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = canvas.width = 300;
    const hSize = canvas.height = 300;
    const padding = 40;
    const drawSize = size - padding * 2;
    
    // Background
    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, size, hSize);

    // Decision Boundary Map Background
    const imgData = ctx.createImageData(size, hSize);
    for (let y = 0; y < hSize; y++) {
      for (let x = 0; x < size; x++) {
        // Map pixel to [0, 1] range (with padding buffer)
        const valX = (x - padding) / drawSize;
        const valY = 1 - (y - padding) / drawSize; // flip Y
        
        const z = valX * w1 + valY * w2 + b;
        const prob = 1 / (1 + Math.exp(-z * 5)); // smoothed for visual
        
        const idx = (y * size + x) * 4;
        // Class 0: #C18C3B, Class 1: #3B7A57
        imgData.data[idx] = Math.floor(193 * (1 - prob) + 59 * prob);
        imgData.data[idx+1] = Math.floor(140 * (1 - prob) + 122 * prob);
        imgData.data[idx+2] = Math.floor(59 * (1 - prob) + 87 * prob);
        imgData.data[idx+3] = 80;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Grid and Axes
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding); ctx.lineTo(padding, size - padding);
    ctx.moveTo(padding, size - padding); ctx.lineTo(size - padding, size - padding);
    ctx.stroke();

    // Decision Line: w1*x + w2*y + b = 0 => y = -(w1/w2)x - b/w2
    if (Math.abs(w2) > 0.001) {
      const getPixelY = (valX: number) => {
        const valY = -(w1 * valX + b) / w2;
        return padding + (1 - valY) * drawSize;
      };
      
      ctx.beginPath();
      ctx.moveTo(padding - 20, getPixelY(-0.1));
      ctx.lineTo(size - padding + 20, getPixelY(1.1));
      ctx.strokeStyle = '#B6532B';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Points
    datasets[datasetType].forEach(pt => {
      const px = padding + pt.x1 * drawSize;
      const py = padding + (1 - pt.x2) * drawSize;
      
      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 1 ? '#3B7A57' : '#C18C3B';
      ctx.fill();
      ctx.strokeStyle = '#FAF6EE';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Label text
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(pt.label.toString(), px, py);
    });

  }, [w1, w2, b, datasetType]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE]">
      <div className="lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <Cpu className="w-6 h-6 text-[#B6532B]" /> Single Perceptron
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            The fundamental building block of Neural Networks. A single perceptron draws a <span className="font-semibold text-[#B6532B]">straight line</span> to separate classes.
            Notice how it perfectly solves AND / OR, but completely fails on XOR (which requires a multi-layer network).
          </p>
          
          <div className="space-y-4">
            <div className="flex bg-[#FAF6EE] p-1 rounded-xl border border-[#E5DDD0]">
              {(['AND', 'OR', 'XOR'] as DatasetType[]).map(type => (
                <button
                  key={type}
                  onClick={() => setDatasetType(type)}
                  className={`flex-1 py-2.5 rounded-lg text-xs font-bold transition-all ${
                    datasetType === type
                      ? 'bg-white shadow-sm text-[#B6532B]'
                      : 'text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                >
                  {type} Logic
                </button>
              ))}
            </div>

            <div className="bg-[#FAF6EE] p-4 rounded-xl border border-[#E5DDD0]">
              <div className="flex justify-between mb-1">
                <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold">Learning Rate (η)</label>
                <span className="text-[10px] font-mono font-bold text-[#B6532B]">{learningRate.toFixed(2)}</span>
              </div>
              <input 
                type="range" min="0.01" max="1" step="0.01" value={learningRate} 
                onChange={e => setLearningRate(parseFloat(e.target.value))}
                className="w-full accent-[#B6532B]"
              />
            </div>
          </div>

          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 grid grid-cols-2 text-center text-xs font-mono text-[#2E251E]">
            <div className="border-r border-[#E5DDD0]">
              <span className="text-[#6E6257] text-[9px] block uppercase font-bold tracking-wider mb-1">Epochs</span>
              <span className="font-bold text-lg">{epoch}</span>
            </div>
            <div>
              <span className="text-[#6E6257] text-[9px] block uppercase font-bold tracking-wider mb-1">Accuracy</span>
              <span className={`font-bold text-lg ${accuracy === 100 ? 'text-[#3B7A57]' : 'text-[#B6532B]'}`}>
                {accuracy.toFixed(0)}%
              </span>
            </div>
          </div>
          
          {datasetType === 'XOR' && (
            <div className="bg-red-100/50 border border-red-200 p-3 rounded-xl flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
              <p className="text-xs text-red-800 font-medium">
                XOR is not linearly separable. The perceptron will loop infinitely trying to draw a straight line through these points.
              </p>
            </div>
          )}
        </div>

        <div className="flex gap-2 mt-6">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            disabled={accuracy === 100 && !isPlaying}
            className={`flex-1 py-3 rounded-xl flex items-center justify-center gap-2 text-sm font-bold transition-all shadow-sm ${
              accuracy === 100 && !isPlaying
                ? 'bg-[#E5DDD0] text-[#6E6257] cursor-not-allowed'
                : isPlaying
                ? 'bg-amber-600 border-amber-600 text-white'
                : 'bg-[#B6532B] border-[#B6532B] text-white hover:bg-[#B6532B]/90'
            }`}
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            {isPlaying ? 'Pause' : accuracy === 100 ? 'Solved!' : 'Train Perceptron'}
          </button>
          <button
            onClick={initWeights}
            className="p-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] transition-colors"
            title="Reset Weights"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="lg:col-span-7 flex flex-col items-center justify-center">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-2 rounded-2xl shadow-xl overflow-hidden mb-4">
          <canvas ref={canvasRef} className="rounded-xl" />
        </div>
        
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-xl flex gap-6 text-xs font-mono text-[#2E251E]">
          <div className="flex flex-col items-center">
            <span className="text-[#6E6257] text-[9px] font-bold uppercase mb-1">Weight 1 (X)</span>
            <span>{w1.toFixed(3)}</span>
          </div>
          <div className="flex flex-col items-center border-l border-[#E5DDD0] pl-6">
            <span className="text-[#6E6257] text-[9px] font-bold uppercase mb-1">Weight 2 (Y)</span>
            <span>{w2.toFixed(3)}</span>
          </div>
          <div className="flex flex-col items-center border-l border-[#E5DDD0] pl-6">
            <span className="text-[#6E6257] text-[9px] font-bold uppercase mb-1">Bias</span>
            <span>{b.toFixed(3)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
