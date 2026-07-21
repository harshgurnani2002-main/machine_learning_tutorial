import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Shuffle, Eye, EyeOff, UserPlus, Star } from 'lucide-react';

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s ^= s >>> 16;
    return (s >>> 0) / 4294967296;
  };
}

interface FactorResult {
  P: number[][];
  Q: number[][];
  R_pred: number[][];
  globalMean: number;
}

function generateMatrix(rng: () => number, numUsers: number, numItems: number, sparsity: number): number[][] {
  const trueK = 3;
  const P_true = Array.from({ length: numUsers }, () =>
    Array.from({ length: trueK }, () => (rng() - 0.5) * 2.5));
  const Q_true = Array.from({ length: numItems }, () =>
    Array.from({ length: trueK }, () => (rng() - 0.5) * 2.5));
  const R: number[][] = Array.from({ length: numUsers }, () => Array(numItems).fill(0));
  for (let u = 0; u < numUsers; u++) {
    for (let i = 0; i < numItems; i++) {
      let pred = 0;
      for (let f = 0; f < trueK; f++) pred += P_true[u][f] * Q_true[i][f];
      pred += (rng() - 0.5) * 1.5;
      const rating = Math.max(1, Math.min(5, Math.round(pred)));
      R[u][i] = rng() > sparsity ? rating : 0;
    }
    const knownCount = R[u].filter(r => r > 0).length;
    if (knownCount < 2) {
      for (let i = 0; i < numItems && R[u].filter(r => r > 0).length < 2; i++) {
        if (R[u][i] === 0) R[u][i] = Math.floor(rng() * 5) + 1;
      }
    }
  }
  return R;
}

function factorizeMatrix(R: number[][], k: number, rng: () => number): FactorResult {
  const m = R.length;
  const n = R[0].length;
  const userKnown = R.map(row => row.filter(r => r > 0).length);
  let sum = 0;
  let count = 0;
  for (let u = 0; u < m; u++)
    for (let i = 0; i < n; i++)
      if (R[u][i] > 0) { sum += R[u][i]; count++; }
  const globalMean = count > 0 ? sum / count : 3;
  const P: number[][] = Array.from({ length: m }, (_, u) =>
    userKnown[u] === 0 ? Array(k).fill(0) : Array.from({ length: k }, () => (rng() - 0.5) * 0.3));
  const Q: number[][] = Array.from({ length: n }, () =>
    Array.from({ length: k }, () => (rng() - 0.5) * 0.3));
  const lr = 0.03;
  const reg = 0.02;
  for (let step = 0; step < 300; step++) {
    let totalError = 0;
    for (let u = 0; u < m; u++) {
      if (userKnown[u] === 0) continue;
      for (let i = 0; i < n; i++) {
        if (R[u][i] === 0) continue;
        let pred = 0;
        for (let f = 0; f < k; f++) pred += P[u][f] * Q[i][f];
        const err = R[u][i] - pred;
        totalError += err * err;
        for (let f = 0; f < k; f++) {
          const puf = P[u][f];
          P[u][f] += lr * (err * Q[i][f] - reg * P[u][f]);
          Q[i][f] += lr * (err * puf - reg * Q[i][f]);
        }
      }
    }
    if (totalError < 0.01) break;
  }
  const R_pred: number[][] = Array.from({ length: m }, () => Array(n).fill(0));
  for (let u = 0; u < m; u++) {
    for (let i = 0; i < n; i++) {
      if (userKnown[u] === 0) {
        R_pred[u][i] = globalMean;
      } else {
        let pred = 0;
        for (let f = 0; f < k; f++) pred += P[u][f] * Q[i][f];
        R_pred[u][i] = Math.round(Math.max(1, Math.min(5, pred)) * 10) / 10;
      }
    }
  }
  return { P, Q, R_pred, globalMean };
}

function ratingColor(r: number): string {
  if (r === 0) return 'transparent';
  if (r >= 4.5) return '#3B7A57';
  if (r >= 3.5) return '#5B9A77';
  if (r >= 2.5) return '#C18C3B';
  if (r >= 1.5) return '#D4735A';
  return '#B6532B';
}

export const RecommenderSimulator: React.FC = () => {
  const [k, setK] = useState(3);
  const [showLatents, setShowLatents] = useState(false);
  const [selectedUser, setSelectedUser] = useState<number | null>(null);
  const [coldStart, setColdStart] = useState(false);
  const [predictCell, setPredictCell] = useState<{ row: number; col: number; value: number } | null>(null);
  const [seed, setSeed] = useState(42);
  const matrixRef = useRef<HTMLCanvasElement>(null);
  const latentRef = useRef<HTMLCanvasElement>(null);

  const ITEMS = 8;
  const USERS = coldStart ? 7 : 6;
  const CELL_W = 54;
  const CELL_H = 36;
  const ROW_LABEL = 34;
  const COL_LABEL = 22;
  const PAD = 8;

  const rng = useMemo(() => seededRng(seed), [seed]);

  const ratingMatrix = useMemo<number[][]>(() => generateMatrix(rng, USERS, ITEMS, 0.45), [seed, USERS, coldStart]);

  const factors = useMemo<FactorResult>(() => factorizeMatrix(ratingMatrix, k, rng), [ratingMatrix, k]);

  const topRecs = useMemo(() => {
    if (selectedUser === null || selectedUser >= USERS) return [];
    const preds = factors.R_pred[selectedUser].map((v, i) => ({ item: i, rating: v }));
    preds.sort((a, b) => b.rating - a.rating);
    return preds.slice(0, 3);
  }, [factors, selectedUser, USERS]);

  const mw = PAD + ROW_LABEL + ITEMS * CELL_W + PAD;
  const mh = PAD + COL_LABEL + USERS * CELL_H + PAD;

  const getCell = useCallback((x: number, y: number) => {
    const col = Math.floor((x - PAD - ROW_LABEL) / CELL_W);
    const row = Math.floor((y - PAD - COL_LABEL) / CELL_H);
    if (row < 0 || row >= USERS || col < 0 || col >= ITEMS) return null;
    if (x < PAD + ROW_LABEL || y < PAD + COL_LABEL) return 'label';
    return { row, col };
  }, [USERS]);

  const handleMatrixClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = matrixRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const cell = getCell(x, y);
    if (!cell || cell === 'label') return;
    if (cell.row >= USERS) return;
    if (x < PAD + ROW_LABEL) {
      setSelectedUser(prev => prev === cell.row ? null : cell.row);
      return;
    }
    if (ratingMatrix[cell.row][cell.col] === 0) {
      setPredictCell({ row: cell.row, col: cell.col, value: factors.R_pred[cell.row][cell.col] });
    }
  }, [getCell, ratingMatrix, factors, USERS]);

  useEffect(() => {
    const canvas = matrixRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = mw;
    canvas.height = mh;
    ctx.clearRect(0, 0, mw, mh);

    // Background
    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, mw, mh);

    // Column headers
    ctx.fillStyle = '#2E251E';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < ITEMS; i++) {
      const cx = PAD + ROW_LABEL + i * CELL_W + CELL_W / 2;
      const cy = PAD + COL_LABEL / 2;
      ctx.fillStyle = '#6E6257';
      ctx.fillText(`I${i + 1}`, cx, cy);
    }

    // Row headers + cells
    for (let r = 0; r < USERS; r++) {
      const rowY = PAD + COL_LABEL + r * CELL_H;
      // Row label
      const lx = PAD + ROW_LABEL / 2;
      const ly = rowY + CELL_H / 2;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = 'bold 11px monospace';
      const isSelected = selectedUser === r;
      const isCold = coldStart && r === USERS - 1;
      ctx.fillStyle = isSelected ? '#B6532B' : isCold ? '#D4735A' : '#2E251E';
      ctx.fillText(isCold ? 'New' : `U${r + 1}`, lx, ly);

      for (let c = 0; c < ITEMS; c++) {
        const cx = PAD + ROW_LABEL + c * CELL_W;
        const cy = rowY;
        const val = ratingMatrix[r][c];
        const pred = factors.R_pred[r][c];
        const isEmpty = val === 0;
        const isPredict = predictCell && predictCell.row === r && predictCell.col === c;

        // Cell background
        if (isSelected) {
          ctx.fillStyle = '#E5DDD0';
          ctx.fillRect(cx + 1, cy + 1, CELL_W - 2, CELL_H - 2);
        }
        if (isPredict) {
          ctx.fillStyle = '#C18C3B30';
          ctx.fillRect(cx + 1, cy + 1, CELL_W - 2, CELL_H - 2);
        }

        // Cell border
        ctx.strokeStyle = '#E5DDD0';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(cx, cy, CELL_W, CELL_H);

        // Cell content
        const px = cx + CELL_W / 2;
        const py = cy + CELL_H / 2;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        if (isPredict) {
          ctx.fillStyle = '#2E251E';
          ctx.font = 'bold 13px monospace';
          ctx.fillText(pred.toFixed(1), px, py - 4);
          ctx.fillStyle = '#B6532B';
          ctx.font = '7px monospace';
          ctx.fillText('pred', px, py + 8);
        } else if (isEmpty) {
          ctx.fillStyle = '#CFC5B4';
          ctx.font = '14px sans-serif';
          ctx.fillText('?', px, py);
        } else {
          ctx.fillStyle = ratingColor(val);
          ctx.font = 'bold 14px sans-serif';
          ctx.fillText(val.toString(), px, py - 2);
          ctx.font = '10px sans-serif';
          ctx.fillStyle = '#6E6257';
          ctx.fillText('★'.repeat(val), px, py + 10);
        }
      }
    }

    // Legend
    ctx.fillStyle = '#CFC5B4';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Click empty cell (?) to predict  •  Click user label for recs', PAD, mh - 2);

    setPredictCell(prev => {
      if (!prev) return null;
      if (prev.row >= USERS) return null;
      return prev;
    });
  }, [ratingMatrix, factors, selectedUser, predictCell, coldStart, USERS, mw, mh]);

  useEffect(() => {
    const canvas = latentRef.current;
    if (!canvas || !showLatents) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const size = 280;
    canvas.width = size;
    canvas.height = size;
    ctx.clearRect(0, 0, size, size);

    const margin = 35;
    const plotSize = size - 2 * margin;
    const { P, Q } = factors;
    const allPoints = [
      ...P.map((v, i) => ({ x: v[0], y: v[1], type: 'user' as const, idx: i })),
      ...Q.map((v, i) => ({ x: v[0], y: v[1], type: 'item' as const, idx: i })),
    ];
    const xs = allPoints.map(p => p.x);
    const ys = allPoints.map(p => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const pad = 1.0;
    const xRange = maxX - minX || 1;
    const yRange = maxY - minY || 1;
    const scaleX = plotSize / (xRange + 2 * pad);
    const scaleY = plotSize / (yRange + 2 * pad);
    const toPlotX = (v: number) => margin + (v - minX + pad) * scaleX;
    const toPlotY = (v: number) => margin + (maxY + pad - v) * scaleY;

    // Background
    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, size, size);

    // Grid
    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const gx = margin + (plotSize / 4) * i;
      ctx.beginPath();
      ctx.moveTo(gx, margin);
      ctx.lineTo(gx, margin + plotSize);
      ctx.stroke();
      const gy = margin + (plotSize / 4) * i;
      ctx.beginPath();
      ctx.moveTo(margin, gy);
      ctx.lineTo(margin + plotSize, gy);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#6E6257';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, margin + plotSize);
    ctx.lineTo(margin + plotSize, margin + plotSize);
    ctx.stroke();

    // Points
    const userColors = ['#B6532B', '#C18C3B', '#3B7A57', '#6E6257', '#2E251E', '#D4735A', '#8B5CF6'];
    for (const p of allPoints) {
      const px = toPlotX(p.x);
      const py = toPlotY(p.y);
      ctx.beginPath();
      if (p.type === 'user') {
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = userColors[p.idx % userColors.length];
        ctx.fill();
        ctx.strokeStyle = '#FAF6EE';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else {
        const s = 5;
        ctx.moveTo(px - s, py - s);
        ctx.lineTo(px + s, py + s);
        ctx.moveTo(px + s, py - s);
        ctx.lineTo(px - s, py + s);
        ctx.strokeStyle = userColors[p.idx % userColors.length];
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Legend
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#6E6257';
    ctx.fillText('● Users  ✕ Items', margin, 4);
  }, [showLatents, factors]);

  const randomize = useCallback(() => {
    setSeed(prev => prev + 1);
    setPredictCell(null);
    setSelectedUser(null);
  }, []);

  const toggleColdStart = useCallback(() => {
    setColdStart(prev => !prev);
    setSelectedUser(null);
    setPredictCell(null);
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-5 p-4">
      {/* Left: Controls */}
      <div className="md:col-span-4 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col gap-4">
        <div>
          <h4 className="text-[#2E251E] font-bold text-base flex items-center gap-2 mb-1">
            <Star className="w-5 h-5 text-[#B6532B]" /> Recommender System
          </h4>
          <p className="text-[#6E6257] text-xs leading-relaxed">
            Matrix factorization via SGD decomposes the rating matrix into user and item latent factors, predicting missing ratings.
          </p>
        </div>

        <button
          onClick={randomize}
          className="w-full py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#2E251E] text-xs font-semibold flex items-center justify-center gap-2 transition-colors cursor-pointer"
        >
          <Shuffle className="w-4 h-4 text-[#B6532B]" /> Randomize Matrix
        </button>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowLatents(!showLatents)}
            className={`flex-1 py-2.5 rounded-xl border text-xs font-semibold flex items-center justify-center gap-2 transition-colors cursor-pointer ${
              showLatents
                ? 'bg-[#B6532B] border-[#B6532B] text-white'
                : 'bg-[#FAF6EE] border-[#E5DDD0] text-[#2E251E] hover:bg-[#FAF6EE]/80'
            }`}
          >
            {showLatents ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            {showLatents ? 'Hide Factors' : 'Show Factors'}
          </button>

          <button
            onClick={toggleColdStart}
            className={`flex-1 py-2.5 rounded-xl border text-xs font-semibold flex items-center justify-center gap-2 transition-colors cursor-pointer ${
              coldStart
                ? 'bg-[#D4735A] border-[#D4735A] text-white'
                : 'bg-[#FAF6EE] border-[#E5DDD0] text-[#2E251E] hover:bg-[#FAF6EE]/80'
            }`}
          >
            <UserPlus className="w-4 h-4" />
            {coldStart ? 'Remove Cold' : 'Cold-Start'}
          </button>
        </div>

        <div>
          <div className="flex justify-between text-xs font-mono mb-1">
            <span className="text-[#6E6257]">Latent Factors (k)</span>
            <span className="text-[#B6532B] font-bold">{k}</span>
          </div>
          <input
            type="range"
            min={2}
            max={6}
            step={1}
            value={k}
            onChange={e => setK(parseInt(e.target.value))}
            className="w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]"
          />
        </div>

        {predictCell && (
          <div className="p-3 bg-[#FAF6EE] border border-[#C18C3B] rounded-xl text-xs">
            <span className="text-[#6E6257]">Predicted </span>
            <span className="font-bold text-[#B6532B]">
              U{predictCell.row + 1} × I{predictCell.col + 1}
            </span>
            <span className="text-[#6E6257]">: </span>
            <span className="font-bold text-[#2E251E] text-lg">{predictCell.value.toFixed(1)}</span>
          </div>
        )}

        {coldStart && (
          <div className="p-3 bg-[#FAF6EE] border border-[#D4735A] rounded-xl text-xs text-[#6E6257] leading-relaxed">
            <span className="font-bold text-[#D4735A]">Cold-start user</span> has no ratings.
            Predictions default to global mean rating: <span className="font-bold text-[#2E251E]">{factors.globalMean.toFixed(2)}</span>
          </div>
        )}

        {selectedUser !== null && selectedUser < USERS && (
          <div className="p-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl">
            <h5 className="text-xs font-bold text-[#2E251E] mb-2 uppercase tracking-wider">
              Top-3 for U{selectedUser + 1}
            </h5>
            <div className="space-y-1.5">
              {topRecs.map((rec, idx) => (
                <div key={idx} className="flex justify-between items-center text-xs font-mono">
                  <span className="text-[#6E6257]">
                    <span className="text-[#B6532B] font-bold">#{idx + 1}</span> Item {rec.item + 1}
                  </span>
                  <span className="font-bold text-[#2E251E]">{rec.rating.toFixed(1)} ★</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="text-[10px] text-[#6E6257] mt-auto border-t border-[#E5DDD0] pt-3 leading-relaxed">
          <strong>How it works:</strong> SGD factorizes the sparse rating matrix R ≈ P·Qᵀ.
          Each user/item gets a k-dimensional latent vector. Click <strong>?</strong> cells to predict,
          click user labels for personalized recommendations.
        </div>
      </div>

      {/* Right: Visualizations */}
      <div className="md:col-span-8 flex flex-col gap-4">
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-3 rounded-2xl flex justify-center shadow-inner">
          <canvas
            ref={matrixRef}
            onClick={handleMatrixClick}
            className="rounded-xl cursor-pointer w-full max-w-[500px]"
            style={{ aspectRatio: `${mw} / ${mh}` }}
          />
        </div>

        {showLatents && (
          <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-3 rounded-2xl flex justify-center shadow-inner">
            <canvas
              ref={latentRef}
              className="rounded-xl w-[280px] h-[280px]"
            />
          </div>
        )}
      </div>
    </div>
  );
};
