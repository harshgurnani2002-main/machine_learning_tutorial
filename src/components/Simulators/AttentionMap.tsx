import React, { useState, useEffect, useRef } from 'react';

import { HelpCircle, Sparkles, BookOpen } from 'lucide-react';

export const AttentionMap: React.FC = () => {


  const sentences = [
    "The animal didn't cross the street because it was too tired.",
    "Transformers handle sequential context with self-attention loops."
  ];

  const [activeSentence, setActiveSentence] = useState<string>(sentences[0]);
  const [selectedWord, setSelectedWord] = useState<string>('it');
  const [activeHead, setActiveHead] = useState<1 | 2>(1);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const tokens = activeSentence.replace(/[.]/g, '').split(' ');

  // Simulated Multi-Head Self-Attention Weights
  const getAttentionWeight = (src: string, tgt: string, head: 1 | 2): number => {
    const s = src.toLowerCase();
    const t = tgt.toLowerCase();

    if (head === 1) {
      // Head 1 focuses on pronoun-noun links
      if (s === 'it') {
        if (t === 'animal') return 0.72;
        if (t === 'street') return 0.08;
        if (t === 'it') return 0.05;
        return 0.02;
      }
      if (s === 'tired') {
        if (t === 'animal') return 0.45;
        if (t === 'tired') return 0.25;
        return 0.03;
      }
    } else {
      // Head 2 focuses on verb-noun actions
      if (s === 'cross') {
        if (t === 'street') return 0.65;
        if (t === 'animal') return 0.20;
        return 0.02;
      }
      if (s === 'it') {
        if (t === 'street') return 0.52; 
        if (t === 'animal') return 0.15;
        return 0.05;
      }
    }

    // Default decay weights based on distance
    const dist = Math.abs(tokens.indexOf(src) - tokens.indexOf(tgt));
    const baseWeight = Math.max(0.01, 0.35 / (dist + 1));
    return parseFloat(baseWeight.toFixed(2));
  };

  useEffect(() => {
    if (activeSentence === sentences[0]) {
      setSelectedWord('it');
    } else {
      setSelectedWord('Transformers');
    }
  }, [activeSentence]);

  // Draw attention link arcs
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 460;
    const h = canvas.height = 180;

    // Clear background
    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, w, h);

    // Draw Grid lines
    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }

    // Render nodes positions
    const tokenCount = tokens.length;
    const paddingX = 35;
    const stepX = (w - paddingX * 2) / (tokenCount - 1);
    const nodeY = h / 2 + 20;

    const positions = tokens.map((_, idx) => ({
      x: paddingX + stepX * idx,
      y: nodeY
    }));

    // Draw attention links from the hovered/selected word
    const srcIdx = tokens.indexOf(selectedWord);
    if (srcIdx !== -1) {
      const srcPos = positions[srcIdx];

      tokens.forEach((tgt, tgtIdx) => {
        if (srcIdx === tgtIdx) return;
        const tgtPos = positions[tgtIdx];
        const weight = getAttentionWeight(selectedWord, tgt, activeHead);

        // Draw bezier curved links
        ctx.beginPath();
        ctx.moveTo(srcPos.x, srcPos.y - 12);
        
        // Control point for arc heights
        const cpX = (srcPos.x + tgtPos.x) / 2;
        const arcHeight = Math.min(65, Math.abs(srcPos.x - tgtPos.x) * 0.45);
        const cpY = srcPos.y - 12 - arcHeight;

        ctx.quadraticCurveTo(cpX, cpY, tgtPos.x, tgtPos.y - 12);

        // Draw connections: Terracotta (Head 1) vs Golden Ochre (Head 2)
        ctx.strokeStyle = activeHead === 1 
          ? `rgba(182, 83, 43, ${weight * 0.9})` 
          : `rgba(193, 140, 59, ${weight * 0.9})`;
        ctx.lineWidth = Math.max(0.75, weight * 7);
        ctx.stroke();

        // Draw connection weights text labels
        if (weight > 0.05) {
          ctx.fillStyle = '#2E251E';
          ctx.font = 'bold 8px monospace';
          ctx.fillText(weight.toFixed(2), cpX - 8, cpY + 8);
        }
      });
    }

    // Draw word labels nodes
    positions.forEach((pos, idx) => {
      const isSelected = tokens[idx] === selectedWord;
      
      // Node marker dot
      ctx.beginPath();
      ctx.arc(pos.x, pos.y - 10, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? '#B6532B' : '#CFC5B4';
      ctx.strokeStyle = '#FAF6EE';
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
    });

  }, [selectedWord, activeSentence, activeHead]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-4">
      {/* Parameters Controls */}
      <div className="lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between">
        <div>
          <h4 className="text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#B6532B]" /> Attention Explorer
          </h4>
          <p className="text-[#6E6257] text-xs mb-5">
            Query vectors matching Key indexes to dynamically compute Context weights.
          </p>

          <div className="space-y-4">
            {/* Sentence preseter */}
            <div>
              <label className="text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold">Select Sentence</label>
              <div className="space-y-2">
                {sentences.map((sent, idx) => (
                  <button
                    key={idx}
                    onClick={() => setActiveSentence(sent)}
                    className={`w-full text-left p-2.5 rounded-xl border text-xs font-mono transition-all leading-normal cursor-pointer ${
                      activeSentence === sent
                        ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#2E251E] font-bold'
                        : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] hover:bg-[#F4EFE6]'
                    }`}
                  >
                    {sent}
                  </button>
                ))}
              </div>
            </div>

            {/* Toggle attention heads */}
            <div>
              <label className="text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold">Attention Head</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setActiveHead(1)}
                  className={`py-2 rounded-xl text-xs font-mono border transition-all cursor-pointer ${
                    activeHead === 1
                      ? 'border-[#B6532B] bg-[#B6532B]/15 text-[#B6532B] font-semibold'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'
                  }`}
                >
                  Head 1: Semantic Link
                </button>
                <button
                  onClick={() => setActiveHead(2)}
                  className={`py-2 rounded-xl text-xs font-mono border transition-all cursor-pointer ${
                    activeHead === 2
                      ? 'border-[#C18C3B] bg-[#C18C3B]/15 text-[#C18C3B] font-semibold'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'
                  }`}
                >
                  Head 2: Syntactic Link
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Explain helper */}
        <div className="p-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl flex gap-2.5 items-start text-xs font-mono leading-normal pt-4 mt-4 text-[#2E251E]">
          <HelpCircle className="w-4 h-4 text-[#B6532B] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block mb-0.5">Winograd Schema Resolving:</span>
            <span className="text-[#6E6257] text-[11px]">
              In Head 1, selecting <span className="text-[#B6532B] font-bold">"it"</span> shows that the model links it directly to <span className="text-[#3B7A57] font-bold">"animal"</span> (0.72) rather than <span className="text-[#C18C3B] font-bold font-mono">"street"</span> (0.08) because it was too <span className="text-[#B6532B] font-semibold">"tired"</span>!
            </span>
          </div>
        </div>
      </div>

      {/* Visual attention Map canvas */}
      <div className="lg:col-span-7 space-y-6 flex flex-col justify-center">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl flex flex-col items-center shadow-inner">
          <div className="border border-[#E5DDD0] rounded-xl bg-white w-full overflow-hidden flex flex-col items-center py-4">
            <canvas ref={canvasRef} />
            
            {/* Clickable tokens list */}
            <div className="flex flex-wrap gap-2 justify-center px-4 mt-2">
              {tokens.map((token, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedWord(token)}
                  className={`px-2.5 py-1 text-[11px] font-mono rounded-lg border transition-all cursor-pointer ${
                    selectedWord === token
                      ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#B6532B] font-bold'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                >
                  {token}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Attention Weight matrix grid */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl">
          <h5 className="text-[#2E251E] text-xs font-semibold font-mono mb-3 flex items-center gap-1.5">
            <BookOpen className="w-4 h-4 text-[#B6532B]" /> Attention Alignment Matrix [Softmax(QK^T / {"\\sqrt{d}"})]
          </h5>
          <div className="overflow-x-auto">
            <div className="flex flex-col min-w-[320px]">
              {/* Columns headers */}
              <div className="flex text-[8px] font-mono text-[#6E6257] border-b border-[#E5DDD0] pb-1">
                <div className="w-12 shrink-0"></div>
                {tokens.map((t, idx) => (
                  <div key={idx} className="flex-1 text-center truncate px-0.5" title={t}>
                    {t}
                  </div>
                ))}
              </div>

              {/* Rows */}
              {tokens.map((rowToken, rIdx) => (
                <div key={rIdx} className="flex items-center text-[9px] font-mono py-1 border-b border-[#E5DDD0]">
                  <div className="w-12 text-[#6E6257] truncate pr-1 font-bold" title={rowToken}>
                    {rowToken}
                  </div>
                  {tokens.map((colToken, cIdx) => {
                    const weight = getAttentionWeight(rowToken, colToken, activeHead);
                    return (
                      <div
                        key={cIdx}
                        style={{ backgroundColor: activeHead === 1 ? `rgba(182, 83, 43, ${weight * 0.4})` : `rgba(193, 140, 59, ${weight * 0.4})` }}
                        className="flex-1 text-center py-1 rounded-sm text-[8px] text-[#2E251E] font-bold border border-[#E5DDD0]"
                        title={`Attn(${rowToken} -> ${colToken}): ${weight}`}
                      >
                        {weight > 0.05 ? weight.toFixed(2) : '-'}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
