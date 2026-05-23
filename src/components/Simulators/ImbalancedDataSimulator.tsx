import React, { useState, useMemo } from 'react';

// ─── Types ─────────────────────────────────────────────────────────────────
interface DataPoint {
  x: number;
  y: number;
  cls: 0 | 1 | 2; // 0=majority, 1=minority-real, 2=minority-synthetic
  id: number;
}

// ─── Seeded random for reproducibility ─────────────────────────────────────
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 4294967296);
  };
}

// ─── Component ─────────────────────────────────────────────────────────────
export const ImbalancedDataSimulator: React.FC = () => {
  const [smoteCount, setSmoteCount] = useState(0);
  const [viewMode, setViewMode] = useState<'raw' | 'smote' | 'undersample'>('raw');
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);

  const W = 520, H = 340;
  const PAD = 30;

  // Generate the base dataset (stable with seeded random)
  const basePoints = useMemo<DataPoint[]>(() => {
    const rng = seededRandom(42);
    const pts: DataPoint[] = [];

    // Majority class — two clusters
    for (let i = 0; i < 120; i++) {
      const angle = rng() * 2 * Math.PI;
      const r = rng() * 70;
      pts.push({
        x: PAD + 100 + r * Math.cos(angle),
        y: PAD + 60 + r * Math.sin(angle),
        cls: 0, id: i
      });
    }
    for (let i = 0; i < 40; i++) {
      pts.push({
        x: PAD + 20 + rng() * 100,
        y: PAD + 160 + rng() * 80,
        cls: 0, id: 120 + i
      });
    }

    // Real minority class — small tight cluster (top right)
    for (let i = 0; i < 8; i++) {
      pts.push({
        x: PAD + 280 + rng() * 60,
        y: PAD + 40 + rng() * 60,
        cls: 1, id: 200 + i
      });
    }
    return pts;
  }, []);

  // Minority real points for SMOTE interpolation anchors
  const minorityPoints = useMemo(() => basePoints.filter(p => p.cls === 1), [basePoints]);

  // Generate synthetic SMOTE points (interpolated between real minority)
  const syntheticPoints = useMemo<DataPoint[]>(() => {
    if (smoteCount === 0) return [];
    const rng = seededRandom(99);
    const pts: DataPoint[] = [];
    for (let i = 0; i < smoteCount; i++) {
      const a = minorityPoints[Math.floor(rng() * minorityPoints.length)];
      const b = minorityPoints[Math.floor(rng() * minorityPoints.length)];
      const λ = rng();
      pts.push({
        x: a.x + λ * (b.x - a.x) + (rng() - 0.5) * 12,
        y: a.y + λ * (b.y - a.y) + (rng() - 0.5) * 12,
        cls: 2, id: 300 + i
      });
    }
    return pts;
  }, [smoteCount, minorityPoints]);

  // Undersampled majority
  const undersampledMajority = useMemo(() => {
    const majority = basePoints.filter(p => p.cls === 0);
    const minority = basePoints.filter(p => p.cls === 1);
    return majority.slice(0, minority.length * 2); // 2:1 ratio
  }, [basePoints]);

  // Points to render
  const displayPoints = useMemo<DataPoint[]>(() => {
    if (viewMode === 'raw') return basePoints;
    if (viewMode === 'smote') return [...basePoints, ...syntheticPoints];
    if (viewMode === 'undersample') return [...undersampledMajority, ...basePoints.filter(p => p.cls === 1)];
    return basePoints;
  }, [viewMode, basePoints, syntheticPoints, undersampledMajority]);

  // Class counts
  const counts = useMemo(() => ({
    majority: displayPoints.filter(p => p.cls === 0).length,
    minority: displayPoints.filter(p => p.cls === 1).length,
    synthetic: displayPoints.filter(p => p.cls === 2).length,
  }), [displayPoints]);

  // Decision boundary X position
  const boundaryX = useMemo(() => {
    const total = counts.majority + counts.minority + counts.synthetic;
    const balance = (counts.minority + counts.synthetic) / total;
    return PAD + 200 - (balance * 100);
  }, [counts]);

  // Accuracy estimate (dummy but meaningful)
  const majorityAccuracy = Math.min(99, 85 + counts.majority * 0.05).toFixed(0);
  const minorityAccuracy = Math.min(95, 10 + (counts.minority + counts.synthetic) * 2.5).toFixed(0);

  const getColor = (cls: 0 | 1 | 2, hovered: boolean) => {
    if (cls === 0) return hovered ? '#ef4444' : '#f87171';
    if (cls === 1) return hovered ? '#2563eb' : '#3b82f6';
    return hovered ? '#7c3aed' : '#a78bfa'; // synthetic = purple
  };

  return (
    <div className="flex flex-col h-full bg-[#FAF6EE] p-4 gap-4 select-none">
      {/* Header */}
      <div className="text-center">
        <h3 className="font-bold text-[#2E251E] text-base">SMOTE & Imbalanced Data Simulator</h3>
        <p className="text-xs text-[#6E6257] mt-0.5">Visualize how SMOTE synthesizes minority samples and shifts the decision boundary</p>
      </div>

      {/* View Mode Tabs */}
      <div className="flex justify-center gap-2">
        {(['raw', 'smote', 'undersample'] as const).map(mode => (
          <button
            key={mode}
            onClick={() => {
              setViewMode(mode);
              if (mode !== 'smote') setSmoteCount(0);
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all border ${
              viewMode === mode
                ? 'bg-[#B6532B] text-white border-[#B6532B] shadow-md'
                : 'bg-white text-[#6E6257] border-[#E5DDD0] hover:border-[#B6532B]'
            }`}
          >
            {mode === 'raw' ? '📊 Raw Imbalanced' : mode === 'smote' ? '✨ Apply SMOTE' : '✂️ Undersample'}
          </button>
        ))}
      </div>

      {/* SMOTE Slider */}
      {viewMode === 'smote' && (
        <div className="flex items-center justify-center gap-3 bg-white rounded-xl border border-[#E5DDD0] px-4 py-2">
          <span className="text-xs font-mono text-[#6E6257] w-28">Synthetic Points:</span>
          <input
            type="range" min={0} max={112} value={smoteCount}
            onChange={e => setSmoteCount(Number(e.target.value))}
            className="flex-1 accent-[#B6532B]"
          />
          <span className="font-mono font-bold text-[#B6532B] text-sm w-8 text-right">{smoteCount}</span>
        </div>
      )}

      {/* SVG Canvas */}
      <div className="relative flex-1 flex justify-center">
        <svg width={W} height={H} className="rounded-xl border border-[#E5DDD0] bg-white shadow-inner">
          {/* Grid lines */}
          {[1,2,3,4].map(i => (
            <React.Fragment key={i}>
              <line x1={PAD + i*100} y1={PAD} x2={PAD + i*100} y2={H - PAD}
                stroke="#F4EFE6" strokeWidth={1} />
              <line x1={PAD} y1={PAD + i*70} x2={W - PAD} y2={PAD + i*70}
                stroke="#F4EFE6" strokeWidth={1} />
            </React.Fragment>
          ))}

          {/* Background shading for class regions */}
          <rect x={PAD} y={PAD} width={Math.max(0, boundaryX - PAD)} height={H - 2*PAD}
            fill="#fef2f2" opacity={0.5} rx={4} />
          <rect x={boundaryX} y={PAD} width={Math.max(0, W - PAD - boundaryX)} height={H - 2*PAD}
            fill="#eff6ff" opacity={0.5} rx={4} />

          {/* Decision boundary */}
          <line x1={boundaryX} y1={PAD - 10} x2={boundaryX} y2={H - PAD + 10}
            stroke="#6b7280" strokeWidth={2} strokeDasharray="6 4"
            className="transition-all duration-700" />
          <text x={boundaryX + 4} y={PAD + 12} fontSize={9} fill="#9ca3af" fontFamily="monospace">
            Decision Boundary
          </text>

          {/* Data points */}
          {displayPoints.map(pt => (
            <g key={pt.id} onMouseEnter={() => setHoveredPoint(pt.id)}
              onMouseLeave={() => setHoveredPoint(null)}>
              {pt.cls === 2 && (
                <circle cx={pt.x} cy={pt.y} r={9} fill="none"
                  stroke="#a78bfa" strokeWidth={1.5} strokeDasharray="3 2" opacity={0.6} />
              )}
              <circle
                cx={pt.x} cy={pt.y}
                r={pt.cls === 0 ? 5 : 6}
                fill={getColor(pt.cls, hoveredPoint === pt.id)}
                stroke="white" strokeWidth={1.5}
                opacity={pt.cls === 0 ? 0.8 : 1}
                style={{ transition: 'fill 0.15s, r 0.15s' }}
              />
            </g>
          ))}

          {/* Axis labels */}
          <text x={W/2} y={H - 5} textAnchor="middle" fontSize={9} fill="#9ca3af" fontFamily="monospace">Feature 1</text>
          <text x={10} y={H/2} textAnchor="middle" fontSize={9} fill="#9ca3af" fontFamily="monospace"
            transform={`rotate(-90, 10, ${H/2})`}>Feature 2</text>
        </svg>
      </div>

      {/* Stats Panel */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="bg-[#B6532B]/5 border border-[#B6532B]/20 rounded-xl p-2">
          <div className="text-lg font-bold text-[#B6532B]">{counts.majority}</div>
          <div className="text-[10px] text-[#B6532B] font-mono">Majority Class</div>
          <div className="text-[10px] text-[#6E6257] mt-0.5">Recall ≈ {majorityAccuracy}%</div>
        </div>
        <div className="bg-[#3B7A57]/5 border border-[#3B7A57]/20 rounded-xl p-2">
          <div className="text-lg font-bold text-[#3B7A57]">{counts.minority}</div>
          <div className="text-[10px] text-[#3B7A57] font-mono">Minority (Real)</div>
          <div className="text-[10px] text-[#6E6257] mt-0.5">
            Ratio 1:{(counts.majority / Math.max(1, counts.minority)).toFixed(0)}
          </div>
        </div>
        <div className="bg-[#C18C3B]/5 border border-[#C18C3B]/20 rounded-xl p-2">
          <div className="text-lg font-bold text-[#C18C3B]">{counts.synthetic}</div>
          <div className="text-[10px] text-[#C18C3B] font-mono">Synthetic (SMOTE)</div>
          <div className="text-[10px] text-[#6E6257] mt-0.5">Minority Recall ≈ {minorityAccuracy}%</div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-4 text-[10px] font-mono text-[#6E6257]">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-[#B6532B] inline-block" /> Majority
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-[#3B7A57] inline-block" /> Minority (Real)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-[#C18C3B] inline-block border border-dashed border-[#C18C3B]" /> Synthetic
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-5 border-t-2 border-dashed border-[#6E6257] inline-block" /> Decision Boundary
        </span>
      </div>
    </div>
  );
};
