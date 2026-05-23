import React, { useState, useMemo } from 'react';

// ─── Seeded random ──────────────────────────────────────────────────────────
function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s ^= s >>> 16;
    return (s >>> 0) / 4294967296;
  };
}

type ModelType = 'single' | 'bagging' | 'boosting';

interface DataPoint { x: number; y: number; cls: 0 | 1; }

// ─── SVG Decision Boundary via grid coloring ───────────────────────────────
interface BoundaryCell { x: number; y: number; cls: 0 | 1; confidence: number; }

// Simulate a boundary grid based on mode
function computeBoundary(
  mode: ModelType,
  iteration: number
): BoundaryCell[] {
  const cells: BoundaryCell[] = [];
  const gridN = 20;
  const W = 460, H = 280;

  for (let gx = 0; gx < gridN; gx++) {
    for (let gy = 0; gy < gridN; gy++) {
      const px = (gx / gridN) * W;
      const py = (gy / gridN) * H;

      let score = 0;

      if (mode === 'single') {
        // Single tree: jagged, axis-aligned boundaries
        // Simulate a deep tree by using many thresholds
        const threshold1 = 230;
        const threshold2 = 140;
        const threshold3 = 310;
        if (px < threshold1) {
          score = py < threshold2 ? 0.9 : 0.15;
        } else {
          score = py > threshold3 * 0.9 ? 0.8 : 0.35;
        }
        // Add jaggedness
        score += Math.sin(px * 0.05) * 0.15 + Math.cos(py * 0.07) * 0.1;
      } else if (mode === 'bagging') {
        // Bagging: smooth, averaged boundary
        // Soft diagonal boundary with slight curve
        const normalized_x = (px - W / 2) / W;
        const normalized_y = (py - H / 2) / H;
        score = 0.5 + normalized_x * 1.2 - normalized_y * 0.3;
        score += Math.sin(px * 0.01) * 0.05; // subtle smoothness
      } else {
        // Boosting: iterative refinement — more accurate, less smooth
        const t = iteration / 8;
        const normalized_x = (px - W * 0.45) / W;
        const normalized_y = (py - H * 0.5) / H;
        // Core boundary that sharpens with iterations
        const raw = normalized_x * 1.5 - normalized_y * 0.4 + Math.sin(normalized_y * 4) * 0.3 * t;
        score = 0.5 + raw;
        // Boosting focuses on hard examples near boundary
        const distFromBoundary = Math.abs(score - 0.5);
        if (distFromBoundary < 0.15 * (1 - t * 0.6)) {
          score += (Math.random() - 0.5) * 0.3;
        }
      }

      score = Math.max(0, Math.min(1, score));
      cells.push({ x: px, y: py, cls: score > 0.5 ? 1 : 0, confidence: Math.abs(score - 0.5) * 2 });
    }
  }
  return cells;
}

// ─── Component ─────────────────────────────────────────────────────────────
export const EnsembleSimulator: React.FC = () => {
  const [mode, setMode] = useState<ModelType>('single');
  const [boostIter, setBoostIter] = useState(1);
  const [showBoundary, setShowBoundary] = useState(true);

  const W = 460, H = 280;

  // Generate dataset
  const points = useMemo<DataPoint[]>(() => {
    const rng = seededRng(77);
    const pts: DataPoint[] = [];
    // Class 0: bottom-left region
    for (let i = 0; i < 40; i++) {
      pts.push({ x: rng() * 200 + 20, y: rng() * 180 + 80, cls: 0 });
    }
    // Class 1: top-right region
    for (let i = 0; i < 40; i++) {
      pts.push({ x: rng() * 200 + 240, y: rng() * 160 + 20, cls: 1 });
    }
    // Overlapping noisy points near boundary
    for (let i = 0; i < 20; i++) {
      pts.push({ x: rng() * 120 + 170, y: rng() * 120 + 80, cls: rng() > 0.5 ? 1 : 0 });
    }
    return pts;
  }, []);

  const boundary = useMemo(
    () => computeBoundary(mode, mode === 'boosting' ? boostIter : 8),
    [points, mode, boostIter]
  );

  // Stats per mode
  const modeStats = {
    single: {
      trainErr: '1%', testErr: '28%',
      bias: 'Low', variance: 'High',
      desc: 'Deep tree memorizes training data. Jagged, complex boundary captures noise.',
      color: '#ef4444',
      icon: '🌳',
    },
    bagging: {
      trainErr: '8%', testErr: '9%',
      bias: 'Low', variance: 'Low',
      desc: 'Averaging 100 trees in parallel smooths the boundary. Variance drops dramatically.',
      color: '#3b82f6',
      icon: '🌳🌲🌳🌲',
    },
    boosting: {
      trainErr: `${Math.max(2, 18 - boostIter * 2)}%`,
      testErr: `${Math.max(7, 22 - boostIter * 2)}%`,
      bias: boostIter < 4 ? 'High' : 'Low',
      variance: boostIter > 6 ? 'Medium' : 'Low',
      desc: `Iteration ${boostIter}/8: Each tree corrects residuals of the previous ensemble.`,
      color: '#7c3aed',
      icon: '🌱→🌿→🌳',
    },
  };

  const stat = modeStats[mode];

  const gridSize = W / 20;

  return (
    <div className="flex flex-col h-full bg-[#FAF6EE] p-4 gap-3 select-none">
      {/* Header */}
      <div className="text-center">
        <h3 className="font-bold text-[#2E251E] text-base">Ensemble Mechanics Simulator</h3>
        <p className="text-xs text-[#6E6257] mt-0.5">Compare how Single Trees, Bagging, and Boosting create fundamentally different decision boundaries</p>
      </div>

      {/* Mode Tabs */}
      <div className="flex justify-center gap-2">
        {(['single', 'bagging', 'boosting'] as const).map(m => (
          <button key={m} onClick={() => { setMode(m); if (m === 'boosting') setBoostIter(1); }}
            className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all border ${
              mode === m ? 'text-white border-transparent shadow-md' : 'bg-white text-[#6E6257] border-[#E5DDD0] hover:border-[#B6532B]'
            }`}
            style={{ backgroundColor: mode === m ? stat.color : undefined }}>
            {m === 'single' ? '🌳 Single Tree' : m === 'bagging' ? '🌲 Bagging (RF)' : '⚡ Boosting (GBM)'}
          </button>
        ))}
      </div>

      {/* Boosting iteration slider */}
      {mode === 'boosting' && (
        <div className="flex items-center gap-3 bg-white border border-[#E5DDD0] rounded-xl px-4 py-2 justify-center">
          <span className="text-xs font-mono text-[#6E6257]">Boosting Iteration:</span>
          <input type="range" min={1} max={8} step={1} value={boostIter}
            onChange={e => setBoostIter(Number(e.target.value))}
            className="w-36 accent-purple-600" />
          <span className="text-sm font-bold text-purple-600 font-mono w-12">{boostIter}/8</span>
        </div>
      )}

      {/* Main: scatter + stats */}
      <div className="flex gap-3 flex-1">
        {/* Decision Boundary Canvas */}
        <div className="relative flex-shrink-0">
          <svg width={W} height={H} className="rounded-xl border border-[#E5DDD0] bg-white shadow-inner">
            {/* Background boundary cells */}
            {showBoundary && boundary.map((cell, i) => (
              <rect key={i}
                x={cell.x} y={cell.y}
                width={gridSize + 0.5} height={gridSize + 0.5}
                fill={cell.cls === 1 ? '#dbeafe' : '#fee2e2'}
                opacity={0.3 + cell.confidence * 0.5}
              />
            ))}

            {/* Boundary line overlay — find transition cells */}
            {showBoundary && boundary.map((cell, i) => {
              const nextX = boundary[i + 1];
              const nextY = boundary[i + 20];
              const isEdgeX = nextX && nextX.cls !== cell.cls;
              const isEdgeY = nextY && nextY.cls !== cell.cls;
              if (!isEdgeX && !isEdgeY) return null;
              return (
                <React.Fragment key={`edge-${i}`}>
                  {isEdgeX && <line x1={cell.x + gridSize} y1={cell.y} x2={cell.x + gridSize} y2={cell.y + gridSize}
                    stroke={stat.color} strokeWidth={1.5} opacity={0.7} />}
                  {isEdgeY && <line x1={cell.x} y1={cell.y + gridSize} x2={cell.x + gridSize} y2={cell.y + gridSize}
                    stroke={stat.color} strokeWidth={1.5} opacity={0.7} />}
                </React.Fragment>
              );
            })}

            {/* Data points */}
            {points.map((pt, i) => (
              <circle key={i}
                cx={pt.x} cy={pt.y} r={5}
                fill={pt.cls === 0 ? '#ef4444' : '#3b82f6'}
                stroke="white" strokeWidth={1.5}
                opacity={0.9} />
            ))}

            {/* Trees icon for bagging */}
            {mode === 'bagging' && [0,1,2,3,4].map(i => (
              <text key={i} x={20 + i*88} y={H - 8} fontSize={14} textAnchor="middle" opacity={0.4}>🌳</text>
            ))}

            {/* Boosting iteration indicator */}
            {mode === 'boosting' && (
              <g>
                {Array.from({ length: boostIter }, (_, i) => (
                  <rect key={i} x={8 + i*26} y={H - 22} width={20} height={10}
                    fill="#7c3aed" rx={3} opacity={0.6 - i * 0.04} />
                ))}
                <text x={8} y={H - 26} fontSize={8} fill="#7c3aed" fontFamily="monospace">
                  Active trees:
                </text>
              </g>
            )}

            {/* Axis labels */}
            <text x={W/2} y={H - 3} textAnchor="middle" fontSize={9} fill="#d1d5db" fontFamily="monospace">Feature 1</text>
            <text x={8} y={H/2} textAnchor="middle" fontSize={9} fill="#d1d5db" fontFamily="monospace"
              transform={`rotate(-90, 8, ${H/2})`}>Feature 2</text>
          </svg>

          {/* Boundary toggle */}
          <button onClick={() => setShowBoundary(!showBoundary)}
            className="absolute bottom-2 right-2 text-[9px] font-mono bg-white/90 border border-[#E5DDD0] rounded px-1.5 py-0.5 text-gray-500 hover:text-[#B6532B] transition-colors">
            {showBoundary ? 'Hide' : 'Show'} Boundary
          </button>
        </div>

        {/* Stats Panel */}
        <div className="flex flex-col gap-2 min-w-[150px]">
          {/* Mode description */}
          <div className="bg-white rounded-xl border p-3 flex-1" style={{ borderColor: stat.color + '66' }}>
            <div className="text-2xl mb-1">{stat.icon}</div>
            <div className="text-xs font-bold mb-1" style={{ color: stat.color }}>
              {mode === 'single' ? 'Single Decision Tree' : mode === 'bagging' ? 'Bagging (Random Forest)' : `Gradient Boosting (Iter ${boostIter})`}
            </div>
            <p className="text-[10px] text-gray-500 leading-relaxed">{stat.desc}</p>
          </div>

          {/* Error bars */}
          <div className="bg-white rounded-xl border border-[#E5DDD0] p-3 space-y-2">
            <div className="text-[10px] font-bold text-[#2E251E] font-mono">Error Comparison</div>
            {/* Train error */}
            <div>
              <div className="flex justify-between text-[9px] font-mono text-gray-500 mb-0.5">
                <span>Train Error</span><span className="font-bold text-green-600">{stat.trainErr}</span>
              </div>
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div className="h-full bg-green-400 rounded-full transition-all duration-500"
                  style={{ width: stat.trainErr }} />
              </div>
            </div>
            {/* Test error */}
            <div>
              <div className="flex justify-between text-[9px] font-mono text-gray-500 mb-0.5">
                <span>Test Error</span><span className="font-bold text-red-500">{stat.testErr}</span>
              </div>
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div className="h-full bg-red-400 rounded-full transition-all duration-500"
                  style={{ width: stat.testErr }} />
              </div>
            </div>
            {/* Bias/Variance badges */}
            <div className="flex gap-1 pt-1">
              <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                stat.bias === 'Low' ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'
              }`}>Bias: {stat.bias}</span>
              <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                stat.variance === 'Low' ? 'bg-green-100 text-green-700' :
                stat.variance === 'Medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
              }`}>Var: {stat.variance}</span>
            </div>
          </div>

          {/* Comparison mini-table */}
          <div className="bg-white rounded-xl border border-[#E5DDD0] p-3">
            <div className="text-[10px] font-bold text-[#2E251E] font-mono mb-1.5">All Models</div>
            {(['single', 'bagging', 'boosting'] as const).map(m => (
              <div key={m} className={`flex justify-between items-center text-[9px] font-mono py-0.5 px-1 rounded mb-0.5 cursor-pointer transition-all ${
                mode === m ? 'bg-[#F4EFE6]' : 'hover:bg-gray-50'
              }`} onClick={() => setMode(m)}>
                <span className="text-gray-600">{m === 'single' ? '🌳 Single' : m === 'bagging' ? '🌲 Bagging' : '⚡ Boosting'}</span>
                <span className="font-bold" style={{ color: modeStats[m].color }}>
                  {modeStats[m].testErr} err
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-4 text-[10px] font-mono text-[#6E6257]">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-red-400 inline-block" /> Class 0
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-blue-400 inline-block" /> Class 1
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-4 h-2 rounded bg-blue-100 border border-blue-300 inline-block" /> Decision Region
        </span>
      </div>
    </div>
  );
};
