import React, { useState } from 'react';
import { LayoutGrid } from 'lucide-react';

// Common 3x3 filters
const filters = {
  edge: {
    name: 'Edge Detection',
    matrix: [
      [-1, -1, -1],
      [-1,  8, -1],
      [-1, -1, -1]
    ]
  },
  sharpen: {
    name: 'Sharpen',
    matrix: [
      [ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0]
    ]
  },
  blur: {
    name: 'Box Blur',
    matrix: [
      [0.11, 0.11, 0.11],
      [0.11, 0.11, 0.11],
      [0.11, 0.11, 0.11]
    ]
  },
  verticalEdge: {
    name: 'Vertical Edge (Sobel)',
    matrix: [
      [-1,  0,  1],
      [-2,  0,  2],
      [-1,  0,  1]
    ]
  }
};

// 5x5 Input Image (Values 0-255)
const defaultImage = [
  [ 50,  50, 200,  50,  50],
  [ 50,  50, 200,  50,  50],
  [ 50,  50, 200,  50,  50],
  [ 50,  50, 200,  50,  50],
  [ 50,  50, 200,  50,  50]
];

export const CNNSimulator: React.FC = () => {
  const [activeFilter, setActiveFilter] = useState<keyof typeof filters>('edge');
  const [hoveredOut, setHoveredOut] = useState<{x: number, y: number} | null>(null);

  const inputSize = 5;
  const filterSize = 3;
  const outputSize = inputSize - filterSize + 1; // 3

  const filterMatrix = filters[activeFilter].matrix;

  // Calculate output feature map
  const outputMatrix = Array(outputSize).fill(0).map(() => Array(outputSize).fill(0));
  
  for (let oy = 0; oy < outputSize; oy++) {
    for (let ox = 0; ox < outputSize; ox++) {
      let sum = 0;
      for (let fy = 0; fy < filterSize; fy++) {
        for (let fx = 0; fx < filterSize; fx++) {
          sum += defaultImage[oy + fy][ox + fx] * filterMatrix[fy][fx];
        }
      }
      outputMatrix[oy][ox] = sum;
    }
  }

  // Helper to map values for grayscale visualization (0-255)
  // ReLu and normalization for visual purposes
  const getOutputColor = (val: number) => {
    // Basic ReLu
    const relu = Math.max(0, val);
    // Rough normalization so it fits 0-255
    let scaled = relu;
    if (activeFilter === 'edge' || activeFilter === 'verticalEdge') {
        scaled = Math.min(255, (relu / 800) * 255);
    } else if (activeFilter === 'sharpen') {
        scaled = Math.min(255, relu);
    } else { // blur
        scaled = Math.min(255, relu);
    }
    return `rgb(${scaled}, ${scaled}, ${scaled})`;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE] overflow-y-auto">
      <div className="lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-start shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <LayoutGrid className="w-6 h-6 text-[#B6532B]" /> Convolution Operation
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            A CNN extracts features by sliding a <span className="font-semibold text-[#B6532B]">Filter Kernel</span> over an Input Image. 
            The math is a simple element-wise multiplication and sum (dot product) for each overlapping region.
          </p>
          
          <div className="space-y-4">
            <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold block mb-2">Select Filter Kernel</label>
            <div className="grid grid-cols-2 gap-2 bg-[#FAF6EE] p-1 rounded-xl border border-[#E5DDD0]">
              {(Object.keys(filters) as (keyof typeof filters)[]).map(type => (
                <button
                  key={type}
                  onClick={() => setActiveFilter(type)}
                  className={`py-2 px-2 rounded-lg text-[10px] font-bold transition-all truncate ${
                    activeFilter === type
                      ? 'bg-white shadow-sm text-[#B6532B]'
                      : 'text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                  title={filters[type].name}
                >
                  {filters[type].name}
                </button>
              ))}
            </div>
            
            <div className="bg-[#FAF6EE] p-4 rounded-xl border border-[#E5DDD0] flex flex-col items-center">
              <span className="text-[10px] text-[#6E6257] font-mono uppercase font-bold mb-3 tracking-wider">Filter Values (3x3)</span>
              <div className="grid grid-cols-3 gap-1">
                {filterMatrix.map((row, y) => row.map((val, x) => (
                  <div key={`${x}-${y}`} className="w-10 h-10 flex items-center justify-center bg-[#F4EFE6] rounded text-xs font-mono font-bold text-[#B6532B] border border-[#E5DDD0]">
                    {val.toFixed(val % 1 !== 0 ? 2 : 0)}
                  </div>
                )))}
              </div>
            </div>
          </div>
          
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4">
            <p className="text-xs text-[#6E6257] italic">
              <strong>Tip:</strong> Hover over the Output Feature Map on the right to see exactly which pixels in the input were used to calculate that specific value (the <span className="text-[#B6532B]">Receptive Field</span>).
            </p>
          </div>
        </div>
      </div>

      <div className="lg:col-span-7 flex flex-col gap-8 items-center justify-center">
        {/* Math (Numeric) Representation */}
        <div className="w-full max-w-2xl bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm">
          <h5 className="text-xs font-mono font-bold text-[#6E6257] uppercase tracking-wider mb-4 text-center border-b border-[#E5DDD0] pb-2">Numeric / Math Representation</h5>
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 md:gap-8">
            
            {/* Input Grid (Numeric) */}
            <div className="flex flex-col items-center gap-2">
              <span className="text-[10px] font-mono font-bold text-[#2E251E] bg-[#E5DDD0] px-2 py-1 rounded">5x5 Input</span>
              <div className="grid grid-cols-5 gap-1 p-1 bg-[#E5DDD0] rounded-lg">
                {defaultImage.map((row, y) => row.map((val, x) => {
                  let isHovered = false;
                  if (hoveredOut) {
                    isHovered = x >= hoveredOut.x && x < hoveredOut.x + 3 && y >= hoveredOut.y && y < hoveredOut.y + 3;
                  }
                  return (
                    <div 
                      key={`in-${x}-${y}`} 
                      className={`w-7 h-7 sm:w-8 sm:h-8 flex items-center justify-center text-[10px] font-mono bg-white rounded-sm transition-colors duration-150 ${
                        isHovered ? 'ring-2 ring-[#B6532B] font-bold text-[#B6532B] bg-orange-50' : 'text-[#6E6257]'
                      }`}
                    >
                      {val}
                    </div>
                  );
                }))}
              </div>
            </div>

            <span className="text-[#B6532B] font-bold text-xl hidden md:block">⊗</span>

            {/* Filter Grid (Numeric) */}
            <div className="flex flex-col items-center gap-2 opacity-50 hidden sm:flex">
              <span className="text-[10px] font-mono font-bold text-[#2E251E]">3x3 Filter</span>
              <div className="grid grid-cols-3 gap-1">
                {filterMatrix.map((row, y) => row.map((val, x) => (
                  <div key={`f-${x}-${y}`} className="w-6 h-6 flex items-center justify-center bg-transparent border border-[#E5DDD0] rounded-sm text-[8px] font-mono text-[#6E6257]">
                    {val}
                  </div>
                )))}
              </div>
            </div>

            <span className="text-[#B6532B] font-bold text-xl">=</span>

            {/* Output Grid (Numeric) */}
            <div className="flex flex-col items-center gap-2">
              <span className="text-[10px] font-mono font-bold text-white bg-[#B6532B] px-2 py-1 rounded">3x3 Output</span>
              <div className="grid grid-cols-3 gap-1 p-1 bg-[#E5DDD0] rounded-lg">
                {outputMatrix.map((row, y) => row.map((val, x) => (
                  <div 
                    key={`out-${x}-${y}`} 
                    className="w-9 h-9 sm:w-11 sm:h-11 flex items-center justify-center text-[10px] font-mono bg-white rounded-sm cursor-crosshair hover:bg-orange-100 hover:text-[#B6532B] hover:font-bold transition-colors"
                    onMouseEnter={() => setHoveredOut({x, y})}
                    onMouseLeave={() => setHoveredOut(null)}
                  >
                    {val.toFixed(0)}
                  </div>
                )))}
              </div>
            </div>
            
          </div>
        </div>

        {/* Visual (Pixel) Representation */}
        <div className="w-full max-w-2xl bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-xl">
          <h5 className="text-xs font-mono font-bold text-[#6E6257] uppercase tracking-wider mb-4 text-center border-b border-[#E5DDD0] pb-2">Visual / Pixel Representation</h5>
          <div className="flex items-center justify-center gap-8">
            
            {/* Input Pixels */}
            <div className="flex flex-col items-center gap-2">
               <div className="grid grid-cols-5 gap-0 border-2 border-[#E5DDD0]">
                {defaultImage.map((row, y) => row.map((val, x) => {
                  let isHovered = false;
                  if (hoveredOut) {
                    isHovered = x >= hoveredOut.x && x < hoveredOut.x + 3 && y >= hoveredOut.y && y < hoveredOut.y + 3;
                  }
                  return (
                    <div 
                      key={`p-in-${x}-${y}`} 
                      className="w-10 h-10 sm:w-12 sm:h-12 relative"
                      style={{ backgroundColor: `rgb(${val},${val},${val})` }}
                    >
                      {isHovered && <div className="absolute inset-0 bg-[#B6532B]/30 border border-[#B6532B]" />}
                    </div>
                  );
                }))}
              </div>
            </div>

            {/* Output Pixels */}
            <div className="flex flex-col items-center gap-2">
               <div className="grid grid-cols-3 gap-0 border-2 border-[#E5DDD0]">
                {outputMatrix.map((row, y) => row.map((val, x) => (
                  <div 
                    key={`p-out-${x}-${y}`} 
                    className="w-10 h-10 sm:w-12 sm:h-12 cursor-crosshair hover:ring-2 hover:ring-[#C18C3B] hover:z-10 relative"
                    style={{ backgroundColor: getOutputColor(val) }}
                    onMouseEnter={() => setHoveredOut({x, y})}
                    onMouseLeave={() => setHoveredOut(null)}
                  />
                )))}
              </div>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
};
