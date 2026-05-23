import React, { useState, useEffect, useRef } from 'react';
import { Trash2, GitMerge } from 'lucide-react';

export const DecisionTreeSimulator: React.FC = () => {
  const [points, setPoints] = useState<Array<{ x: number; y: number; label: 0 | 1 }>>([
    { x: 150, y: 150, label: 0 },
    { x: 180, y: 180, label: 0 },
    { x: 200, y: 100, label: 0 },
    { x: 400, y: 100, label: 1 },
    { x: 450, y: 160, label: 1 },
    { x: 500, y: 220, label: 1 },
    { x: 300, y: 280, label: 0 },
    { x: 350, y: 350, label: 1 },
  ]);
  const [activeLabel, setActiveLabel] = useState<0 | 1>(0);
  const [maxDepth, setMaxDepth] = useState<number>(3);
  const [criterion, setCriterion] = useState<'gini' | 'entropy'>('gini');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const clearPoints = () => setPoints([]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPoints(prev => [...prev, { x, y, label: activeLabel }]);
  };

  interface TreeNode {
    feature?: 'x' | 'y';
    threshold?: number;
    left?: TreeNode;
    right?: TreeNode;
    isLeaf: boolean;
    val: number;
    depth: number;
    impurity: number;
    samples: number;
    classCounts: [number, number];
    region: { minX: number, maxX: number, minY: number, maxY: number };
  }

  const computeImpurity = (pts: typeof points): number => {
    if (pts.length === 0) return 0;
    const c0 = pts.filter(p => p.label === 0).length;
    const c1 = pts.length - c0;
    const p0 = c0 / pts.length;
    const p1 = c1 / pts.length;
    
    if (criterion === 'gini') {
      return 1 - (p0 * p0 + p1 * p1);
    } else {
      let ent = 0;
      if (p0 > 0) ent -= p0 * Math.log2(p0);
      if (p1 > 0) ent -= p1 * Math.log2(p1);
      return ent;
    }
  };

  const buildTree = (
    pts: typeof points,
    depth: number,
    region: { minX: number, maxX: number, minY: number, maxY: number }
  ): TreeNode => {
    const c0 = pts.filter(p => p.label === 0).length;
    const c1 = pts.length - c0;
    const val = c1 > c0 ? 1 : 0;
    const impurity = computeImpurity(pts);
    
    const leaf: TreeNode = {
      isLeaf: true,
      val,
      depth,
      impurity,
      samples: pts.length,
      classCounts: [c0, c1],
      region
    };

    if (depth >= maxDepth || pts.length < 2 || impurity === 0) {
      return leaf;
    }

    let bestGain = -1;
    let bestFeat: 'x' | 'y' = 'x';
    let bestThresh = 0;

    ['x', 'y'].forEach(feat => {
      const vals = Array.from(new Set(pts.map(p => feat === 'x' ? p.x : p.y))).sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const thresh = (vals[i] + vals[i + 1]) / 2;
        const left = pts.filter(p => (feat === 'x' ? p.x : p.y) <= thresh);
        const right = pts.filter(p => (feat === 'x' ? p.x : p.y) > thresh);

        if (left.length === 0 || right.length === 0) continue;

        const pLeft = left.length / pts.length;
        const pRight = right.length / pts.length;
        const gain = impurity - (pLeft * computeImpurity(left) + pRight * computeImpurity(right));

        if (gain > bestGain) {
          bestGain = gain;
          bestFeat = feat as 'x' | 'y';
          bestThresh = thresh;
        }
      }
    });

    if (bestGain <= 0) return leaf;

    const leftPts = pts.filter(p => (bestFeat === 'x' ? p.x : p.y) <= bestThresh);
    const rightPts = pts.filter(p => (bestFeat === 'x' ? p.x : p.y) > bestThresh);

    const leftRegion = { ...region, [bestFeat === 'x' ? 'maxX' : 'maxY']: bestThresh };
    const rightRegion = { ...region, [bestFeat === 'x' ? 'minX' : 'minY']: bestThresh };

    return {
      isLeaf: false,
      feature: bestFeat,
      threshold: bestThresh,
      depth,
      impurity,
      samples: pts.length,
      classCounts: [c0, c1],
      region,
      val,
      left: buildTree(leftPts, depth + 1, leftRegion),
      right: buildTree(rightPts, depth + 1, rightRegion)
    };
  };

  const tree = buildTree(points, 0, { minX: 0, maxX: 600, minY: 0, maxY: 400 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 600;
    const h = canvas.height = 400;

    ctx.fillStyle = '#0f172a'; // slate-900
    ctx.fillRect(0, 0, w, h);

    const drawRegions = (node: TreeNode) => {
      if (node.isLeaf) {
        // Red for 0, Blue for 1
        ctx.fillStyle = node.val === 0 ? 'rgba(244, 63, 94, 0.15)' : 'rgba(56, 189, 248, 0.15)';
        ctx.fillRect(node.region.minX, node.region.minY, node.region.maxX - node.region.minX, node.region.maxY - node.region.minY);
      } else {
        if (node.left) drawRegions(node.left);
        if (node.right) drawRegions(node.right);
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        if (node.feature === 'x') {
          ctx.moveTo(node.threshold!, node.region.minY);
          ctx.lineTo(node.threshold!, node.region.maxY);
        } else {
          ctx.moveTo(node.region.minX, node.threshold!);
          ctx.lineTo(node.region.maxX, node.threshold!);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
    };

    drawRegions(tree);

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 50) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Draw points
    points.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? '#f43f5e' : '#38bdf8';
      ctx.shadowBlur = 10;
      ctx.shadowColor = pt.label === 0 ? '#f43f5e' : '#38bdf8';
      ctx.fill();
      ctx.shadowBlur = 0;
      
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

  }, [points, maxDepth, criterion, tree]);

  let totalNodes = 0;
  let totalLeaves = 0;
  const traverseStats = (node: TreeNode) => {
    totalNodes++;
    if (node.isLeaf) totalLeaves++;
    else {
      if (node.left) traverseStats(node.left);
      if (node.right) traverseStats(node.right);
    }
  };
  traverseStats(tree);

  let correct = 0;
  points.forEach(p => {
    let curr = tree;
    while (!curr.isLeaf) {
      if ((curr.feature === 'x' ? p.x : p.y) <= curr.threshold!) curr = curr.left!;
      else curr = curr.right!;
    }
    if (curr.val === p.label) correct++;
  });
  const accuracy = points.length > 0 ? (correct / points.length) * 100 : 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-8 p-6">
      <div className="md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-slate-900 font-bold text-xl tracking-tight flex items-center gap-3">
            <GitMerge className="w-6 h-6 text-emerald-500" /> Decision Tree
          </h4>
          <p className="text-slate-500 text-sm leading-relaxed">
            Trees partition space into pure rectangular regions using recursive orthogonal splits. Watch how <span className="font-semibold text-emerald-500">Max Depth</span> prevents overfitting.
          </p>

          <div className="space-y-3">
            <label className="text-xs text-slate-400 font-bold uppercase tracking-wider block">Class to Plot</label>
            <div className="flex bg-slate-100 p-1 rounded-xl gap-1">
              <button
                onClick={() => setActiveLabel(0)}
                className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${
                  activeLabel === 0 ? 'bg-rose-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'
                }`}
              >
                <span className={`w-3 h-3 rounded-full ${activeLabel === 0 ? 'bg-white' : 'bg-rose-500'}`} /> Class 0
              </button>
              <button
                onClick={() => setActiveLabel(1)}
                className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${
                  activeLabel === 1 ? 'bg-sky-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'
                }`}
              >
                <span className={`w-3 h-3 rounded-full ${activeLabel === 1 ? 'bg-white' : 'bg-sky-500'}`} /> Class 1
              </button>
            </div>
          </div>

          <div className="space-y-3">
            <label className="text-xs text-slate-400 font-bold uppercase tracking-wider block">Impurity Metric</label>
            <div className="flex bg-slate-100 p-1 rounded-xl gap-1">
              <button
                onClick={() => setCriterion('gini')}
                className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${
                  criterion === 'gini' ? 'bg-emerald-600 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'
                }`}
              >
                Gini Impurity
              </button>
              <button
                onClick={() => setCriterion('entropy')}
                className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${
                  criterion === 'entropy' ? 'bg-emerald-600 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'
                }`}
              >
                Entropy
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center text-sm font-bold">
              <span className="text-slate-500">Max Depth Limit:</span>
              <span className="text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full">{maxDepth}</span>
            </div>
            <input
              type="range"
              min="1"
              max="8"
              step="1"
              value={maxDepth}
              onChange={(e) => setMaxDepth(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
          </div>

          <div className="p-5 bg-slate-50 border border-slate-200 rounded-2xl space-y-3 shadow-inner">
            <div className="flex justify-between items-center">
              <span className="text-slate-500 text-sm font-medium">Accuracy</span>
              <span className="text-emerald-600 font-mono font-bold text-lg">{accuracy.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500 text-sm font-medium">Tree Size</span>
              <span className="text-slate-900 font-mono font-bold">{totalNodes} nodes</span>
            </div>
          </div>
        </div>

        <button
          onClick={clearPoints}
          className="w-full py-3 mt-6 rounded-xl bg-red-50 text-red-600 hover:bg-red-100 text-sm font-bold transition-colors flex justify-center items-center gap-2"
        >
          <Trash2 className="w-5 h-5" /> Clear Data
        </button>
      </div>

      <div className="md:col-span-8 flex flex-col items-center justify-center">
        <div className="bg-[#2E251E] border border-[#4A3D31] p-2 rounded-3xl w-full flex justify-center shadow-2xl relative overflow-hidden group">
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            className="rounded-2xl cursor-crosshair w-full aspect-[3/2]"
          />
          <div className="absolute top-6 left-6 bg-white/10 backdrop-blur-md px-4 py-2 rounded-xl border border-white/20 text-xs font-mono text-white shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
            Click to plot • Observe recursive splits
          </div>
        </div>
      </div>
    </div>
  );
};
