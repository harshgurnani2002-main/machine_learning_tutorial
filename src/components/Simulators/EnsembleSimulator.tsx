import React, { useState, useMemo, useCallback } from 'react';

// ─── Mathematical Models (Real Decision Trees) ─────────────────────────────

interface DataPoint { id: string; x: number; y: number; cls: number; }

interface TreeNode {
  feature: 'x' | 'y';
  threshold: number;
  left?: TreeNode;
  right?: TreeNode;
  value?: number;
}

// Calculate Variance (MSE) for regression splits
function calculateVariance(points: DataPoint[]): number {
  if (points.length === 0) return 0;
  const mean = points.reduce((sum, p) => sum + p.cls, 0) / points.length;
  return points.reduce((sum, p) => sum + Math.pow(p.cls - mean, 2), 0) / points.length;
}

// Build a CART Regression Tree
function buildTree(points: DataPoint[], depth: number, maxDepth: number, minSamples: number = 2): TreeNode {
  if (depth >= maxDepth || points.length < minSamples) {
    const mean = points.length > 0 ? points.reduce((sum, p) => sum + p.cls, 0) / points.length : 0;
    return { value: mean, feature: 'x', threshold: 0 };
  }

  let bestFeature: 'x' | 'y' = 'x';
  let bestThreshold = 0;
  let bestCost = Infinity;
  let bestGroups: [DataPoint[], DataPoint[]] = [[], []];

  ['x', 'y'].forEach(feat => {
    const sorted = [...points].sort((a, b) => a[feat as 'x'|'y'] - b[feat as 'x'|'y']);
    for (let i = 0; i < sorted.length - 1; i++) {
      if (sorted[i][feat as 'x'|'y'] === sorted[i + 1][feat as 'x'|'y']) continue;
      const thresh = (sorted[i][feat as 'x'|'y'] + sorted[i + 1][feat as 'x'|'y']) / 2;
      
      const left: DataPoint[] = [];
      const right: DataPoint[] = [];
      for (const p of points) {
        if (p[feat as 'x'|'y'] <= thresh) left.push(p);
        else right.push(p);
      }
      
      if (left.length === 0 || right.length === 0) continue;

      const cost = (left.length * calculateVariance(left) + right.length * calculateVariance(right)) / points.length;
      if (cost < bestCost) {
        bestCost = cost;
        bestFeature = feat as 'x'|'y';
        bestThreshold = thresh;
        bestGroups = [left, right];
      }
    }
  });

  if (bestCost === Infinity || bestGroups[0].length === 0 || bestGroups[1].length === 0) {
    const mean = points.length > 0 ? points.reduce((sum, p) => sum + p.cls, 0) / points.length : 0;
    return { value: mean, feature: 'x', threshold: 0 };
  }

  return {
    feature: bestFeature,
    threshold: bestThreshold,
    left: buildTree(bestGroups[0], depth + 1, maxDepth, minSamples),
    right: buildTree(bestGroups[1], depth + 1, maxDepth, minSamples)
  };
}

function predictTree(node: TreeNode, x: number, y: number): number {
  if (node.value !== undefined) return node.value;
  const val = node.feature === 'x' ? x : y;
  if (val <= node.threshold) return predictTree(node.left!, x, y);
  return predictTree(node.right!, x, y);
}

// Seeded random for bootstrap sampling
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = Math.sin(s) * 10000;
    return s - Math.floor(s);
  };
}

// ─── Main Component ────────────────────────────────────────────────────────

type Mode = 'single' | 'bagging' | 'boosting';
type ClassType = 0 | 1;

export const EnsembleSimulator: React.FC = () => {
  const [mode, setMode] = useState<Mode>('single');
  const [points, setPoints] = useState<DataPoint[]>([]);
  const [activeClass, setActiveClass] = useState<ClassType>(0);
  
  // Controls
  const [rfTrees, setRfTrees] = useState(10);
  const [gbmIters, setGbmIters] = useState(5);
  const [maxDepthSingle, setMaxDepthSingle] = useState(6);

  const W = 500, H = 400;
  const GRID_SIZE = 10;

  // Generate Initial Data
  const generateData = useCallback(() => {
    const newPoints: DataPoint[] = [];
    const rand = Math.random;
    for (let i = 0; i < 30; i++) {
      newPoints.push({ id: `p${rand()}`, x: 50 + rand() * 150, y: 50 + rand() * 200, cls: 0 });
      newPoints.push({ id: `p${rand()}`, x: 250 + rand() * 200, y: 150 + rand() * 200, cls: 1 });
    }
    // Add noise
    for (let i = 0; i < 15; i++) {
      newPoints.push({ id: `p${rand()}`, x: 150 + rand() * 200, y: 100 + rand() * 200, cls: rand() > 0.5 ? 1 : 0 });
    }
    setPoints(newPoints);
  }, []);

  // Initialize once
  React.useEffect(() => {
    generateData();
  }, [generateData]);

  // Train Models dynamically based on current points and mode
  const { boundary, trainError } = useMemo(() => {
    const cells: {x: number, y: number, score: number}[] = [];
    if (points.length === 0) return { boundary: [], trainError: '0.0' };

    let modelPredict: (x: number, y: number) => number;

    if (mode === 'single') {
      const tree = buildTree(points, 0, maxDepthSingle, 2);
      modelPredict = (x, y) => predictTree(tree, x, y);
    } 
    else if (mode === 'bagging') {
      const trees: TreeNode[] = [];
      const rng = seededRandom(42); // deterministic bagging
      for (let i = 0; i < rfTrees; i++) {
        const bootstrap: DataPoint[] = [];
        for (let j = 0; j < points.length; j++) {
          bootstrap.push(points[Math.floor(rng() * points.length)]);
        }
        trees.push(buildTree(bootstrap, 0, 6, 2)); // deeper trees for RF
      }
      modelPredict = (x, y) => {
        let sum = 0;
        for (const t of trees) sum += predictTree(t, x, y);
        return sum / trees.length;
      };
    } 
    else { // boosting
      const trees: TreeNode[] = [];
      const learningRate = 0.3;
      let currentF = points.map(() => 0.5); // Initial F0
      
      for (let i = 0; i < gbmIters; i++) {
        // Compute residuals
        const residuals = points.map((p, idx) => ({ ...p, cls: p.cls - currentF[idx] }));
        const tree = buildTree(residuals, 0, 2, 2); // Shallow trees (stumps/depth 2) for Boosting
        trees.push(tree);
        // Update F
        currentF = currentF.map((f, idx) => f + learningRate * predictTree(tree, points[idx].x, points[idx].y));
      }

      modelPredict = (x, y) => {
        let f = 0.5;
        for (const t of trees) {
          f += learningRate * predictTree(t, x, y);
        }
        return f;
      };
    }

    // Compute boundary grid
    for (let gx = 0; gx < W; gx += GRID_SIZE) {
      for (let gy = 0; gy < H; gy += GRID_SIZE) {
        const px = gx + GRID_SIZE / 2;
        const py = gy + GRID_SIZE / 2;
        let score = modelPredict(px, py);
        score = Math.max(0, Math.min(1, score));
        cells.push({ x: gx, y: gy, score });
      }
    }

    // Calculate training error
    let errCount = 0;
    for (const p of points) {
      const pred = modelPredict(p.x, p.y) > 0.5 ? 1 : 0;
      if (pred !== p.cls) errCount++;
    }

    return { boundary: cells, trainError: ((errCount / points.length) * 100).toFixed(1) };
  }, [points, mode, maxDepthSingle, rfTrees, gbmIters]);

  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPoints(prev => [...prev, { id: `p${Math.random()}`, x, y, cls: activeClass }]);
  };

  const clearData = () => setPoints([]);

  // Theme Constants (Premium Light Theme)
  const colors = {
    bg: '#F8FAFC',
    card: '#FFFFFF',
    text: '#1E293B',
    textMuted: '#64748B',
    border: '#E2E8F0',
    primary: '#6366F1', // Indigo
    class0: '#F43F5E', // Rose
    class1: '#0EA5E9', // Sky Blue
    accentBg: '#EEF2FF'
  };

  return (
    <div className="ens-container" style={{
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      backgroundColor: colors.bg,
      color: colors.text,
      padding: '32px',
      borderRadius: '24px',
      boxShadow: '0 10px 30px -5px rgba(0,0,0,0.05), border 1px solid rgba(0,0,0,0.05)',
      display: 'flex',
      flexDirection: 'column',
      gap: '24px',
      maxWidth: '1000px',
      margin: '0 auto',
    }}>
      <style>{`
        .ens-btn {
          padding: 10px 18px;
          border-radius: 12px;
          font-weight: 600;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s ease;
          border: 1px solid ${colors.border};
          background: ${colors.card};
          color: ${colors.textMuted};
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .ens-btn:hover {
          background: ${colors.accentBg};
          border-color: ${colors.primary};
          color: ${colors.primary};
        }
        .ens-btn.active {
          background: ${colors.primary};
          color: white;
          border-color: ${colors.primary};
          box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        
        .ens-class-btn {
          width: 32px; height: 32px;
          border-radius: 50%;
          border: 3px solid transparent;
          cursor: pointer;
          transition: transform 0.2s;
        }
        .ens-class-btn:hover { transform: scale(1.1); }
        .ens-class-btn.active { border-color: ${colors.text}; transform: scale(1.1); }

        .ens-slider {
          -webkit-appearance: none;
          width: 100%;
          height: 6px;
          background: #E2E8F0;
          border-radius: 3px;
          outline: none;
        }
        .ens-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: ${colors.primary};
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          transition: transform 0.1s;
        }
        .ens-slider::-webkit-slider-thumb:hover { transform: scale(1.2); }
      `}</style>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '8px' }}>
        <h2 style={{ margin: 0, fontSize: '32px', fontWeight: 800, color: colors.text, letterSpacing: '-0.5px' }}>
          Interactive True Ensemble Simulator
        </h2>
        <p style={{ margin: '8px 0 0', fontSize: '16px', color: colors.textMuted }}>
          Add points to the canvas and watch real math algorithms construct the decision boundaries live!
        </p>
      </div>

      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        
        {/* Left: Canvas & Data Controls */}
        <div style={{ flex: '1', minWidth: '500px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Canvas Toolbar */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: colors.card, padding: '12px 20px', borderRadius: '16px', border: `1px solid ${colors.border}`, boxShadow: '0 2px 4px rgba(0,0,0,0.02)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '14px', fontWeight: 600, color: colors.textMuted }}>Draw Tool:</span>
              <button 
                className={`ens-class-btn ${activeClass === 0 ? 'active' : ''}`}
                style={{ backgroundColor: colors.class0 }}
                onClick={() => setActiveClass(0)}
                title="Add Class 0 (Red)"
              />
              <button 
                className={`ens-class-btn ${activeClass === 1 ? 'active' : ''}`}
                style={{ backgroundColor: colors.class1 }}
                onClick={() => setActiveClass(1)}
                title="Add Class 1 (Blue)"
              />
              <span style={{ fontSize: '12px', color: colors.textMuted, marginLeft: '8px' }}>(Click canvas to add)</span>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button onClick={generateData} className="ens-btn" style={{ padding: '6px 12px', fontSize: '12px' }}>🎲 Randomize</button>
              <button onClick={clearData} className="ens-btn" style={{ padding: '6px 12px', fontSize: '12px' }}>🗑️ Clear</button>
            </div>
          </div>

          {/* SVG Canvas */}
          <div style={{ position: 'relative', borderRadius: '20px', overflow: 'hidden', border: `1px solid ${colors.border}`, background: '#FFFFFF', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.05)' }}>
            <svg 
              width={W} height={H} 
              onClick={handleCanvasClick}
              style={{ cursor: 'crosshair', display: 'block' }}
            >
              {/* Render Boundary Grid */}
              {boundary.map((cell, i) => (
                <rect key={`bg-${i}`}
                  x={cell.x} y={cell.y}
                  width={GRID_SIZE + 0.5} height={GRID_SIZE + 0.5}
                  fill={cell.score > 0.5 ? colors.class1 : colors.class0}
                  opacity={0.1 + Math.abs(cell.score - 0.5) * 0.8} // Confidence mapping
                  style={{ transition: 'opacity 0.2s, fill 0.2s' }}
                />
              ))}

              {/* Render Data Points */}
              {points.map((pt) => (
                <circle key={pt.id}
                  cx={pt.x} cy={pt.y} r={6}
                  fill={pt.cls === 0 ? colors.class0 : colors.class1}
                  stroke="#FFFFFF" strokeWidth={2}
                  style={{ transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)' }}
                />
              ))}
            </svg>
          </div>
        </div>

        {/* Right: Model Controls & Info */}
        <div style={{ flex: '0 0 340px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Mode Selector */}
          <div style={{ background: colors.card, padding: '20px', borderRadius: '20px', border: `1px solid ${colors.border}`, boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', textTransform: 'uppercase', color: colors.textMuted, letterSpacing: '1px' }}>Algorithm Selection</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <button className={`ens-btn ${mode === 'single' ? 'active' : ''}`} onClick={() => setMode('single')}>
                🌳 Single Decision Tree
              </button>
              <button className={`ens-btn ${mode === 'bagging' ? 'active' : ''}`} onClick={() => setMode('bagging')}>
                🌲 Bagging (Random Forest)
              </button>
              <button className={`ens-btn ${mode === 'boosting' ? 'active' : ''}`} onClick={() => setMode('boosting')}>
                ⚡ Boosting (Gradient Boost)
              </button>
            </div>
          </div>

          {/* Model Specific Controls */}
          <div style={{ background: colors.card, padding: '20px', borderRadius: '20px', border: `1px solid ${colors.border}`, boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)', flex: 1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '20px' }}>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: colors.primary }}>
                {mode === 'single' ? 'Single Tree Setup' : mode === 'bagging' ? 'Forest Settings' : 'Boosting Stages'}
              </h3>
            </div>

            {mode === 'single' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <p style={{ margin: 0, fontSize: '13px', color: colors.textMuted, lineHeight: 1.5 }}>
                  A single unconstrained tree perfectly memorizes data but creates jagged, unnatural boundaries. Try adding outlier points!
                </p>
                <div style={{ marginTop: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>
                    <span>Max Depth</span>
                    <span>{maxDepthSingle}</span>
                  </div>
                  <input type="range" min={1} max={10} value={maxDepthSingle} onChange={e => setMaxDepthSingle(Number(e.target.value))} className="ens-slider" />
                </div>
              </div>
            )}

            {mode === 'bagging' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <p style={{ margin: 0, fontSize: '13px', color: colors.textMuted, lineHeight: 1.5 }}>
                  Trains many deep trees on random subsets of points and averages them. Notice how the boundary becomes remarkably smooth and resistant to outliers!
                </p>
                <div style={{ marginTop: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>
                    <span>Number of Trees</span>
                    <span>{rfTrees}</span>
                  </div>
                  <input type="range" min={1} max={50} value={rfTrees} onChange={e => setRfTrees(Number(e.target.value))} className="ens-slider" />
                </div>
              </div>
            )}

            {mode === 'boosting' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <p style={{ margin: 0, fontSize: '13px', color: colors.textMuted, lineHeight: 1.5 }}>
                  Sequentially adds shallow trees (stumps). Each new tree focuses only on correcting the mistakes of the previous sequence.
                </p>
                <div style={{ marginTop: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>
                    <span>Boosting Iterations</span>
                    <span>{gbmIters}</span>
                  </div>
                  <input type="range" min={1} max={30} value={gbmIters} onChange={e => setGbmIters(Number(e.target.value))} className="ens-slider" />
                </div>
              </div>
            )}

            <div style={{ marginTop: '24px', paddingTop: '20px', borderTop: `1px solid ${colors.border}` }}>
              <div style={{ fontSize: '12px', color: colors.textMuted, textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px' }}>Live Metrics</div>
              <div style={{ background: colors.accentBg, borderRadius: '12px', padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '14px', fontWeight: 600, color: colors.primary }}>Training Error</span>
                <span style={{ fontSize: '24px', fontWeight: 800, color: trainError === '0.0' ? '#10B981' : colors.text, fontFamily: 'monospace' }}>
                  {trainError}%
                </span>
              </div>
              <div style={{ fontSize: '11px', color: colors.textMuted, marginTop: '8px', textAlign: 'center' }}>
                (Models automatically retrain when data changes)
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};
