import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Database, RotateCcw } from 'lucide-react';

interface Point {
  x: number;
  y: number;
  cluster: number;
  isCore: boolean;
}

const CLUSTER_COLORS = ['#B6532B', '#C18C3B', '#3B7A57', '#6E6257', '#2E251E', '#8B4513', '#4682B4'];

function generateData(): Point[] {
  const w = 400;
  const h = 320;
  const centers = [
    { x: w * 0.28, y: h * 0.3 },
    { x: w * 0.72, y: h * 0.28 },
    { x: w * 0.48, y: h * 0.72 },
  ];
  const points: Point[] = [];
  for (let c = 0; c < 3; c++) {
    const count = 18 + Math.floor(Math.random() * 10);
    const center = centers[c];
    for (let j = 0; j < count; j++) {
      points.push({
        x: Math.max(5, Math.min(395, center.x + (Math.random() - 0.5) * 75)),
        y: Math.max(5, Math.min(315, center.y + (Math.random() - 0.5) * 75)),
        cluster: -1,
        isCore: false,
      });
    }
  }
  const noiseCount = 10 + Math.floor(Math.random() * 8);
  for (let i = 0; i < noiseCount; i++) {
    points.push({
      x: 20 + Math.random() * 360,
      y: 20 + Math.random() * 280,
      cluster: -1,
      isCore: false,
    });
  }
  return points;
}

function dbscan(points: Point[], epsilon: number, minPts: number): Point[] {
  const eps = epsilon * 20;
  const n = points.length;
  const result: Point[] = points.map(p => ({ ...p }));
  const dist: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = result[i].x - result[j].x;
      const dy = result[i].y - result[j].y;
      dist[i][j] = dx * dx + dy * dy;
      dist[j][i] = dist[i][j];
    }
  }
  const eps2 = eps * eps;
  const getNeighbors = (idx: number): number[] => {
    const nbrs: number[] = [];
    for (let j = 0; j < n; j++) {
      if (idx !== j && dist[idx][j] <= eps2) nbrs.push(j);
    }
    return nbrs;
  };
  const visited = new Array(n).fill(false);
  let clusterId = 0;
  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = true;
    const neighbors = getNeighbors(i);
    if (neighbors.length < minPts) continue;
    result[i].isCore = true;
    result[i].cluster = clusterId;
    const queue = [...neighbors];
    let qIdx = 0;
    while (qIdx < queue.length) {
      const q = queue[qIdx++];
      if (result[q].cluster !== -1) continue;
      result[q].cluster = clusterId;
      if (!visited[q]) {
        visited[q] = true;
        const qNeighbors = getNeighbors(q);
        if (qNeighbors.length >= minPts) {
          result[q].isCore = true;
          for (const nbr of qNeighbors) {
            if (result[nbr].cluster === -1) {
              queue.push(nbr);
            }
          }
        }
      }
    }
    clusterId++;
  }
  return result;
}

export const DbscanSandbox: React.FC = () => {
  const [data, setData] = useState<Point[]>([]);
  const [epsilon, setEpsilon] = useState(0.8);
  const [minPts, setMinPts] = useState(4);
  const [showConnections, setShowConnections] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [clusteredData, setClusteredData] = useState<Point[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const initializeSimulation = useCallback(() => {
    setData(generateData());
  }, []);

  useEffect(() => {
    initializeSimulation();
  }, [initializeSimulation]);

  useEffect(() => {
    if (data.length === 0) return;
    setClusteredData(dbscan(data, epsilon, minPts));
  }, [data, epsilon, minPts]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 400;
    const h = canvas.height = 320;
    const eps = epsilon * 20;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let x = 0; x <= w; x += 40) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y <= h; y += 40) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    if (clusteredData.length === 0) return;

    const eps2 = eps * eps;

    if (showConnections) {
      ctx.strokeStyle = '#CFC5B4';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < clusteredData.length; i++) {
        for (let j = i + 1; j < clusteredData.length; j++) {
          if (clusteredData[i].cluster === -1 || clusteredData[j].cluster === -1) continue;
          const dx = clusteredData[i].x - clusteredData[j].x;
          const dy = clusteredData[i].y - clusteredData[j].y;
          if (dx * dx + dy * dy <= eps2) {
            ctx.beginPath();
            ctx.moveTo(clusteredData[i].x, clusteredData[i].y);
            ctx.lineTo(clusteredData[j].x, clusteredData[j].y);
            ctx.stroke();
          }
        }
      }
    }

    if (hoveredIndex !== null && hoveredIndex < clusteredData.length) {
      const p = clusteredData[hoveredIndex];
      ctx.beginPath();
      ctx.arc(p.x, p.y, eps, 0, Math.PI * 2);
      ctx.strokeStyle = '#B6532B';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    for (let i = 0; i < clusteredData.length; i++) {
      const pt = clusteredData[i];
      const isHovered = hoveredIndex === i;
      const isBorder = !pt.isCore && pt.cluster !== -1;

      let fillColor: string;
      if (pt.cluster === -1) {
        fillColor = '#9E9485';
      } else {
        fillColor = CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length];
      }

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isHovered ? 6 : 4.5, 0, Math.PI * 2);

      if (isBorder) ctx.globalAlpha = 0.55;
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.globalAlpha = 1;

      ctx.strokeStyle = '#FAF6EE';
      ctx.lineWidth = isHovered ? 2 : 1;
      ctx.stroke();
    }
  }, [clusteredData, epsilon, hoveredIndex, showConnections]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    let found = -1;
    for (let i = 0; i < clusteredData.length; i++) {
      const dx = clusteredData[i].x - mx;
      const dy = clusteredData[i].y - my;
      if (dx * dx + dy * dy <= 8 * 8) {
        found = i;
        break;
      }
    }
    setHoveredIndex(found === -1 ? null : found);
  };

  const handleMouseLeave = () => setHoveredIndex(null);

  const nClusters = new Set(clusteredData.filter(p => p.cluster !== -1).map(p => p.cluster)).size;
  const nNoise = clusteredData.filter(p => p.cluster === -1).length;

  let hoverNeighborCount = 0;
  if (hoveredIndex !== null && clusteredData.length > 0) {
    const eps2 = (epsilon * 20) ** 2;
    const p = clusteredData[hoveredIndex];
    for (const q of clusteredData) {
      const dx = p.x - q.x;
      const dy = p.y - q.y;
      if (dx * dx + dy * dy <= eps2) hoverNeighborCount++;
    }
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 p-4">
      <div className="md:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between">
        <div>
          <h4 className="text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2">
            <Database className="w-5 h-5 text-[#B6532B]" /> DBSCAN Sandbox
          </h4>
          <p className="text-[#6E6257] text-xs mb-5">
            Density-based clustering — discover arbitrarily shaped clusters.
          </p>

          <div className="space-y-6">
            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-[#2E251E]">Epsilon (ε)</span>
                <span className="text-[#B6532B] font-semibold">{epsilon.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="3.0"
                step="0.1"
                value={epsilon}
                onChange={e => setEpsilon(parseFloat(e.target.value))}
                className="w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-[#2E251E]">Min Points (MinPts)</span>
                <span className="text-[#B6532B] font-semibold">{minPts}</span>
              </div>
              <input
                type="range"
                min="2"
                max="10"
                step="1"
                value={minPts}
                onChange={e => setMinPts(parseInt(e.target.value))}
                className="w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]"
              />
            </div>

            <label className="flex items-center gap-2 text-xs font-mono text-[#2E251E] cursor-pointer">
              <input
                type="checkbox"
                checked={showConnections}
                onChange={e => setShowConnections(e.target.checked)}
                className="accent-[#B6532B]"
              />
              Show Connections
            </label>

            <div className="p-3.5 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl space-y-2 text-xs font-mono text-[#2E251E]">
              <div className="flex justify-between">
                <span className="text-[#6E6257]">Clusters Found:</span>
                <span className="text-[#B6532B] font-bold">{nClusters}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#6E6257]">Noise Points:</span>
                <span className="text-[#6E6257] font-bold">{nNoise}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#6E6257]">Total Points:</span>
                <span className="text-[#2E251E] font-bold">{clusteredData.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#6E6257]">Core / Border / Noise:</span>
                <span className="text-[#2E251E] font-bold">
                  {clusteredData.filter(p => p.isCore).length}
                  <span className="text-[#6E6257]"> / </span>
                  {clusteredData.filter(p => !p.isCore && p.cluster !== -1).length}
                  <span className="text-[#6E6257]"> / </span>
                  {nNoise}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="pt-5 border-t border-[#E5DDD0]">
          <button
            onClick={initializeSimulation}
            className="w-full py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#FAF6EE]/85 text-[#6E6257] hover:text-[#2E251E] transition-colors flex items-center justify-center gap-2 text-xs font-semibold cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" /> Regenerate Data
          </button>
        </div>
      </div>

      <div className="md:col-span-7 flex flex-col items-center">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl w-full flex justify-center shadow-inner relative">
          <canvas
            ref={canvasRef}
            className="rounded-xl border border-[#E5DDD0] w-full max-w-[400px] h-[320px] cursor-crosshair"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
          <div className="absolute top-7 left-7 bg-[#2E251E]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE] uppercase tracking-wide pointer-events-none">
            ε = {epsilon.toFixed(1)}  ·  ε-neighbors: {hoveredIndex !== null ? hoverNeighborCount : '-'}
          </div>
        </div>
      </div>
    </div>
  );
};
