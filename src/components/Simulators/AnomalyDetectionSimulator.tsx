import React, { useState, useMemo, useEffect, useRef } from 'react';

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

interface Point {
  x: number;
  y: number;
  trueAnomaly: boolean;
  score: number; // 0..1, higher = more anomalous
  id: number;
}

// ─── Component ─────────────────────────────────────────────────────────────
export const AnomalyDetectionSimulator: React.FC = () => {
  const [contamination, setContamination] = useState(0.10);
  const [algo, setAlgo] = useState<'isolation' | 'lof'>('isolation');
  const [splitStep, setSplitStep] = useState(0);
  const [animating, setAnimating] = useState(false);
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const W = 520, H = 320;
  const CX = 220, CY = 160, CLUSTER_R = 70;

  // ── Generate dataset ────────────────────────────────────────────────────
  const points = useMemo<Point[]>(() => {
    const rng = seededRng(123);
    const pts: Point[] = [];

    // Normal cluster — Gaussian around center
    for (let i = 0; i < 140; i++) {
      const angle = rng() * 2 * Math.PI;
      const r = Math.sqrt(-2 * Math.log(rng() + 1e-10)) * CLUSTER_R * 0.5;
      const px = CX + r * Math.cos(angle);
      const py = CY + r * Math.sin(angle);
      // Score proportional to distance from center (lower = more normal)
      const dist = Math.sqrt((px - CX) ** 2 + (py - CY) ** 2);
      const score = Math.min(0.48, 0.05 + dist / (CLUSTER_R * 4));
      pts.push({ x: px, y: py, trueAnomaly: false, score, id: i });
    }

    // True anomalies — scattered far from cluster
    const anomalyPositions = [
      { x: 420, y: 60 }, { x: 60, y: 280 }, { x: 460, y: 290 },
      { x: 40, y: 50 }, { x: 490, y: 160 }, { x: 100, y: 310 },
      { x: 400, y: 310 }, { x: 30, y: 160 }, { x: 310, y: 30 },
      { x: 160, y: 290 }, { x: 480, y: 80 }, { x: 55, y: 200 },
      { x: 350, y: 290 }, { x: 150, y: 20 },
    ];
    anomalyPositions.forEach((pos, i) => {
      pts.push({
        x: pos.x, y: pos.y, trueAnomaly: true,
        score: 0.62 + rng() * 0.36, id: 200 + i
      });
    });
    return pts;
  }, []);

  // ── Apply threshold to get detections ──────────────────────────────────
  const detections = useMemo(() => {
    const sorted = [...points].sort((a, b) => b.score - a.score);
    const n = Math.round(contamination * points.length);
    const detectedIds = new Set(sorted.slice(0, n).map(p => p.id));
    return detectedIds;
  }, [points, contamination]);

  // ── Metrics ────────────────────────────────────────────────────────────
  const metrics = useMemo(() => {
    let tp = 0, fp = 0, fn = 0;
    points.forEach(p => {
      const detected = detections.has(p.id);
      if (detected && p.trueAnomaly) tp++;
      else if (detected && !p.trueAnomaly) fp++;
      else if (!detected && p.trueAnomaly) fn++;
    });
    const precision = tp / Math.max(1, tp + fp);
    const recall = tp / Math.max(1, tp + fn);
    const f1 = 2 * precision * recall / Math.max(0.001, precision + recall);
    return { tp, fp, fn, precision, recall, f1 };
  }, [detections, points]);

  // ── Animate isolation splits ────────────────────────────────────────────
  const handleAnimate = () => {
    setAnimating(true);
    setSplitStep(0);
    let step = 0;
    const tick = () => {
      step++;
      setSplitStep(step);
      if (step < 8) {
        animRef.current = setTimeout(tick, 500);
      } else {
        setAnimating(false);
      }
    };
    animRef.current = setTimeout(tick, 400);
  };
  useEffect(() => {
    return () => { if (animRef.current) clearTimeout(animRef.current); };
  }, []);

  // Auto-animate on mount for one-click experience
  useEffect(() => {
    const timer = setTimeout(() => handleAnimate(), 800);
    return () => clearTimeout(timer);
  }, []);

  // ── Isolation split lines (for animation) ──────────────────────────────
  const splitLines = useMemo(() => [
    { x1: 310, y1: 10, x2: 310, y2: 330 },  // vertical through middle
    { x1: 10, y1: 200, x2: 510, y2: 200 },  // horizontal
    { x1: 400, y1: 10, x2: 400, y2: 200 },  // isolate top-right
    { x1: 10, y1: 270, x2: 310, y2: 270 },  // isolate bottom-left
    { x1: 440, y1: 200, x2: 440, y2: 330 }, // subdivide
    { x1: 10, y1: 100, x2: 310, y2: 100 },  // subdivide cluster area
    { x1: 150, y1: 100, x2: 150, y2: 200 }, // cluster subdivision
    { x1: 310, y1: 130, x2: 440, y2: 130 }, // right area
  ], []);

  // ── Score → color gradient ─────────────────────────────────────────────
  const scoreToColor = (score: number, detected: boolean, hover: boolean) => {
    if (!detected) return hover ? '#60a5fa' : '#93c5fd'; // normal = blue
    // Anomaly: red gradient based on score
    const intensity = Math.min(1, (score - 0.5) / 0.5);
    const r = Math.round(220 + 35 * intensity);
    const g = Math.round(50 - 30 * intensity);
    return hover ? `rgb(${r},${g},50)` : `rgb(${r},${g},80)`;
  };

  const histBins = useMemo(() => {
    const bins = Array(10).fill(0);
    points.forEach(p => {
      const bin = Math.min(9, Math.floor(p.score * 10));
      bins[bin]++;
    });
    return bins;
  }, [points]);
  const maxBin = Math.max(...histBins);

  return (
    <div className="flex flex-col h-full bg-[#FAF6EE] p-4 gap-3 select-none">
      {/* Header */}
      <div className="text-center">
        <h3 className="font-bold text-[#2E251E] text-base">Anomaly Detection Simulator</h3>
        <p className="text-xs text-[#6E6257] mt-0.5">Adjust contamination to tune the detection threshold. Animate isolation splits to see HOW Isolation Forest works.</p>
      </div>

      {/* Controls Row */}
      <div className="flex flex-wrap items-center gap-3 justify-center">
        {/* Algorithm toggle */}
        <div className="flex rounded-lg border border-[#E5DDD0] overflow-hidden text-xs font-mono">
          {(['isolation', 'lof'] as const).map(a => (
            <button key={a}
              onClick={() => setAlgo(a)}
              className={`px-3 py-1.5 transition-all ${algo === a ? 'bg-[#B6532B] text-white' : 'bg-white text-[#6E6257] hover:bg-[#F4EFE6]'}`}>
              {a === 'isolation' ? '🌲 Isolation Forest' : '🔵 LOF'}
            </button>
          ))}
        </div>

        {/* Contamination slider */}
        <div className="flex items-center gap-2 bg-white border border-[#E5DDD0] rounded-xl px-3 py-1.5">
          <span className="text-[10px] font-mono text-[#6E6257]">Contamination:</span>
          <input type="range" min={0.01} max={0.25} step={0.01} value={contamination}
            onChange={e => setContamination(Number(e.target.value))}
            className="w-28 accent-[#B6532B]" />
          <span className="text-xs font-bold text-[#B6532B] font-mono w-10">{(contamination * 100).toFixed(0)}%</span>
        </div>

        {/* Animate button */}
        {algo === 'isolation' && (
          <button onClick={handleAnimate} disabled={animating}
            className={`px-3 py-1.5 rounded-xl text-xs font-mono border transition-all ${animating ? 'bg-gray-100 text-gray-400 border-gray-200' : 'bg-[#2E251E] text-white border-[#2E251E] hover:bg-[#B6532B]'}`}>
            {animating ? '⏳ Splitting...' : '▶ Animate Splits'}
          </button>
        )}
      </div>

      {/* Main area: scatter + histogram */}
      <div className="flex gap-3 flex-1">
        {/* Scatter plot */}
        <svg width={W} height={H} className="rounded-xl border border-[#E5DDD0] bg-white shadow-inner flex-shrink-0">
          {/* Background regions */}
          <defs>
            <radialGradient id="normalRegion" cx="42%" cy="50%" r="35%">
              <stop offset="0%" stopColor="#dbeafe" stopOpacity={0.5} />
              <stop offset="100%" stopColor="white" stopOpacity={0} />
            </radialGradient>
          </defs>
          <rect width={W} height={H} fill="url(#normalRegion)" />

          {/* Grid */}
          {[1,2,3,4].map(i => (
            <React.Fragment key={i}>
              <line x1={i*100} y1={0} x2={i*100} y2={H} stroke="#F4EFE6" strokeWidth={1} />
              <line x1={0} y1={i*80} x2={W} y2={i*80} stroke="#F4EFE6" strokeWidth={1} />
            </React.Fragment>
          ))}

          {/* Isolation split animation */}
          {algo === 'isolation' && splitLines.slice(0, splitStep).map((line, i) => (
            <line key={i} x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2}
              stroke="#B6532B" strokeWidth={1} strokeDasharray="4 3" opacity={0.5 - i * 0.04} />
          ))}

          {/* LOF density rings */}
          {algo === 'lof' && (
            <>
              <circle cx={CX} cy={CY} r={CLUSTER_R * 0.6} fill="none" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 3" opacity={0.3} />
              <circle cx={CX} cy={CY} r={CLUSTER_R} fill="none" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 3" opacity={0.2} />
              <circle cx={CX} cy={CY} r={CLUSTER_R * 1.4} fill="none" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 3" opacity={0.1} />
            </>
          )}

          {/* Data points */}
          {points.map(pt => {
            const detected = detections.has(pt.id);
            const hovered = hoveredId === pt.id;
            const color = scoreToColor(pt.score, detected, hovered);
            return (
              <g key={pt.id}
                onMouseEnter={() => setHoveredId(pt.id)}
                onMouseLeave={() => setHoveredId(null)}>
                {detected && (
                  <circle cx={pt.x} cy={pt.y} r={12}
                    fill="none" stroke="#ef4444" strokeWidth={1.5}
                    strokeDasharray="4 2" opacity={0.7}>
                    <animate attributeName="r" values="10;14;10" dur="2s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.7;0.3;0.7" dur="2s" repeatCount="indefinite" />
                  </circle>
                )}
                <circle cx={pt.x} cy={pt.y}
                  r={hovered ? 7 : (detected ? 6 : 5)}
                  fill={color} stroke="white" strokeWidth={1.5}
                  style={{ transition: 'fill 0.3s, r 0.15s', cursor: 'pointer' }} />
                {hovered && (
                  <g>
                    <rect x={pt.x + 8} y={pt.y - 18} width={70} height={20}
                      fill="white" stroke="#E5DDD0" rx={4} strokeWidth={1} />
                    <text x={pt.x + 12} y={pt.y - 4} fontSize={9} fill="#6E6257" fontFamily="monospace">
                      Score: {pt.score.toFixed(3)} {detected ? '⚠️' : '✓'}
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* Labels */}
          <text x={CX} y={CY - CLUSTER_R - 8} textAnchor="middle" fontSize={9} fill="#60a5fa" fontFamily="monospace" opacity={0.8}>
            Normal Cluster
          </text>
        </svg>

        {/* Right panel: metrics + histogram */}
        <div className="flex flex-col gap-2 min-w-[130px]">
          {/* Metrics */}
          <div className="bg-[#FAF6EE] rounded-xl border border-[#E5DDD0] p-3 text-[10px] font-mono space-y-1.5">
            <div className="font-bold text-[#2E251E] text-xs mb-1">Detection Metrics</div>
            <div className="flex justify-between">
              <span className="text-[#6E6257]">Precision</span>
              <span className="font-bold text-[#3B7A57]">{(metrics.precision * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6E6257]">Recall</span>
              <span className="font-bold text-[#C18C3B]">{(metrics.recall * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6E6257]">F1 Score</span>
              <span className="font-bold text-[#B6532B]">{(metrics.f1 * 100).toFixed(0)}%</span>
            </div>
            <div className="border-t border-[#E5DDD0] pt-1 mt-1">
              <div className="flex justify-between"><span className="text-[#6E6257]">TP</span><span className="text-[#3B7A57] font-bold">{metrics.tp}</span></div>
              <div className="flex justify-between"><span className="text-[#6E6257]">FP</span><span className="text-[#C18C3B] font-bold">{metrics.fp}</span></div>
              <div className="flex justify-between"><span className="text-[#6E6257]">FN</span><span className="text-[#B6532B] font-bold">{metrics.fn}</span></div>
            </div>
          </div>

          {/* Score Histogram */}
          <div className="bg-white rounded-xl border border-[#E5DDD0] p-3 flex-1">
            <div className="text-[10px] font-mono font-bold text-[#2E251E] mb-2">Anomaly Score Distribution</div>
            <div className="flex items-end gap-0.5 h-20">
              {histBins.map((count, i) => {
                const binScore = (i + 0.5) / 10;
                const isAboveThreshold = binScore > (1 - contamination);
                const height = maxBin > 0 ? (count / maxBin) * 100 : 0;
                return (
                  <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
                    <div
                      className="w-full rounded-sm transition-all duration-300"
                      style={{
                        height: `${height}%`,
                        backgroundColor: isAboveThreshold ? '#ef4444' : '#93c5fd',
                        minHeight: count > 0 ? 2 : 0
                      }} />
                  </div>
                );
              })}
            </div>
            <div className="flex justify-between text-[8px] text-gray-400 font-mono mt-1">
              <span>0.0</span><span className="text-red-400">↑ threshold</span><span>1.0</span>
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-4 text-[10px] font-mono text-[#6E6257]">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-blue-300 inline-block" /> Normal
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> Detected Anomaly
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-4 border-t-2 border-dashed border-[#B6532B] inline-block" /> Isolation Split
        </span>
      </div>
    </div>
  );
};
