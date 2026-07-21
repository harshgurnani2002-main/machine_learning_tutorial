import React, { useState, useEffect, useRef, useMemo } from 'react';
import { MessageSquare, Send, RefreshCw, Brain, Sigma, Hash, BarChart3 } from 'lucide-react';

const PRESETS = [
  'This movie was amazing!',
  'The food was terrible and cold.',
  'It was okay, nothing special.',
  'Absolutely loved every single minute of it!',
  'Worst experience of my entire life.',
];

const SENTIMENT_LEXICON: Record<string, number> = {
  amazing: 0.95, fantastic: 0.92, wonderful: 0.88, loved: 0.9, love: 0.85,
  excellent: 0.93, brilliant: 0.9, awesome: 0.85, great: 0.75, good: 0.6,
  beautiful: 0.85, incredible: 0.88, perfect: 0.9, best: 0.85, outstanding: 0.92,
  terrible: -0.95, awful: -0.9, horrible: -0.92, worst: -0.9, bad: -0.65,
  boring: -0.8, dull: -0.7, hate: -0.85, hated: -0.88, disgusting: -0.9,
  disappointing: -0.75, poor: -0.7, cold: -0.3, waste: -0.8, sucks: -0.85,
  okay: 0.15, fine: 0.2, decent: 0.3, average: 0.1, nothing: 0.0, special: 0.5,
  movie: 0.1, film: 0.1, was: 0.0, is: 0.0, the: 0.0, a: 0.0, an: 0.0,
  it: 0.0, this: 0.0, of: 0.0, and: 0.0, not: -0.2, no: -0.3, but: 0.0,
  very: 0.2, really: 0.2, so: 0.1, such: 0.1, every: 0.1, absolutely: 0.3,
  totally: 0.2, completely: 0.2, minute: 0.0, single: 0.0, entire: 0.0,
  experience: 0.1, life: 0.1, whole: 0.1, more: 0.0, never: -0.4, ever: 0.0,
  again: 0.0, would: 0.0, recommend: 0.5, watching: 0.1, food: 0.0,
};

const COLORS = {
  bg: '#FAF6EE',
  border: '#E5DDD0',
  text: '#2E251E',
  muted: '#6E6257',
  accent: '#B6532B',
  panelBg: '#F4EFE6',
  gold: '#C18C3B',
  positive: '#3B7A57',
  negative: '#B6532B',
  neutral: '#6E6257',
};

const tokenize = (text: string): string[] =>
  text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(Boolean);

const computeSentimentWordScores = (tokens: string[]): number[] =>
  tokens.map(t => SENTIMENT_LEXICON[t] ?? 0);

const computeAttention = (
  tokens: string[],
  wordScores: number[],
  isTransformer: boolean
): number[] => {
  const n = tokens.length;
  if (n === 0) return [];

  if (isTransformer) {
    const raw: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sim = 1 - Math.abs(wordScores[i] - wordScores[j]);
        if (tokens[j] === 'not' || tokens[j] === 'no' || tokens[j] === 'never') {
          sim *= 1.5;
        }
        raw[i][j] = Math.max(0, sim);
      }
    }

    const negIndices = tokens
      .map((t, i) => (t === 'not' || t === 'no' || t === 'never' ? i : -1))
      .filter(i => i >= 0);

    if (negIndices.length > 0) {
      for (const ni of negIndices) {
        for (let j = 0; j < n; j++) {
          if (ni !== j) {
            for (let k = 0; k < n; k++) {
              raw[k][j] *= 0.6;
            }
          }
        }
      }
    }

    const out: number[] = [];
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += raw[i][j];
      out.push(sum / n);
    }
    const mx = Math.max(...out, 0.001);
    return out.map(v => v / mx);
  }

  const absScores = wordScores.map(Math.abs);
  const total = absScores.reduce((a, b) => a + b, 0) || 1;
  return absScores.map(s => s / total);
};

const computeTFIDF = (tokens: string[]): { word: string; weight: number }[] => {
  const freq: Record<string, number> = {};
  tokens.forEach(t => { freq[t] = (freq[t] || 0) + 1; });
  const n = tokens.length;
  return tokens
    .filter((t, i, a) => a.indexOf(t) === i)
    .map(word => {
      const tf = freq[word] / n;
      const idf = Math.log(100 / (1 + freq[word]));
      const lexicalBoost = Math.abs(SENTIMENT_LEXICON[word] ?? 0) * 0.5;
      return { word, weight: Math.round((tf * idf + lexicalBoost) * 1000) / 1000 };
    })
    .sort((a, b) => b.weight - a.weight);
};

const heatColor = (val: number, maxVal: number): string => {
  if (maxVal === 0) return '#FFFFFF';
  const r = val / maxVal;
  const rv = Math.round(255 - (255 - 182) * r);
  const gv = Math.round(255 - (255 - 83) * r);
  const bv = Math.round(255 - (255 - 43) * r);
  return `rgb(${rv}, ${gv}, ${bv})`;
};

const isPositiveWord = (s: number) => s > 0.2;
const isNegativeWord = (s: number) => s < -0.2;

export const NLPSentimentSimulator: React.FC = () => {
  const [inputText, setInputText] = useState(PRESETS[0]);
  const [modelType, setModelType] = useState<'bow' | 'transformer'>('transformer');
  const [analyzed, setAnalyzed] = useState(false);
  const [animProgress, setAnimProgress] = useState(0);
  const heatmapRef = useRef<HTMLCanvasElement>(null);
  const cloudRef = useRef<HTMLCanvasElement>(null);

  const tokens = useMemo(() => tokenize(inputText), [inputText]);

  const wordScores = useMemo(() => computeSentimentWordScores(tokens), [tokens]);

  const attentionWeights = useMemo(
    () => computeAttention(tokens, wordScores, modelType === 'transformer'),
    [tokens, wordScores, modelType]
  );

  const tfidfKeywords = useMemo(() => computeTFIDF(tokens).slice(0, 6), [tokens]);

  const rawSentiment = useMemo(() => {
    if (tokens.length === 0) return 0;
    const sum = wordScores.reduce((a, b) => a + b, 0);
    return sum / tokens.length;
  }, [wordScores, tokens]);

  const sentimentLabel = rawSentiment > 0.15 ? 'Positive' : rawSentiment < -0.15 ? 'Negative' : 'Neutral';
  const sentimentConfidence = Math.min(
    Math.abs(rawSentiment) * 2.5 + 0.3,
    0.98
  );

  const analyze = () => {
    setAnalyzed(false);
    setAnimProgress(0);
    const duration = 600;
    const start = performance.now();
    const tick = (now: number) => {
      const elapsed = now - start;
      const pct = Math.min(elapsed / duration, 1);
      setAnimProgress(pct);
      if (pct < 1) requestAnimationFrame(tick);
      else setAnalyzed(true);
    };
    requestAnimationFrame(tick);
  };

  const drawHeatmap = () => {
    const c = heatmapRef.current;
    if (!c || tokens.length === 0) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = 500;
    const h = 300;
    c.width = w * dpr;
    c.height = h * dpr;
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
    ctx.scale(dpr, dpr);

    const pad = { top: 30, bottom: 40, left: 80, right: 20 };
    const n = tokens.length;
    const cellW = Math.min((w - pad.left - pad.right) / n, 60);
    const cellH = (h - pad.top - pad.bottom) / n;

    const displayTokens = analyzed ? tokens : [];
    const displayWeights = analyzed ? attentionWeights : [];

    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = COLORS.text;
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Token Attention Heatmap', w / 2, 16);

    const maxW = Math.max(...displayWeights, 0.001);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const val = displayWeights[j] ?? 0;
        const x = pad.left + j * cellW;
        const y = pad.top + i * cellH;
        ctx.fillStyle = heatColor(val, maxW);
        ctx.fillRect(x, y, cellW, cellH);
        ctx.strokeStyle = COLORS.border;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, cellW, cellH);
      }
    }

    ctx.fillStyle = COLORS.text;
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i < n; i++) {
      ctx.fillText(displayTokens[i] ?? '', pad.left - 6, pad.top + i * cellH + cellH / 2 + 3);
    }

    ctx.textAlign = 'center';
    for (let j = 0; j < n; j++) {
      ctx.save();
      ctx.translate(pad.left + j * cellW + cellW / 2, pad.top + n * cellH + 14);
      ctx.rotate(-0.5);
      ctx.fillStyle = COLORS.text;
      ctx.font = '9px monospace';
      ctx.fillText(displayTokens[j] ?? '', 0, 0);
      ctx.restore();
    }

    if (!analyzed && tokens.length > 0) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Click "Analyze" to see attention', w / 2, h / 2 + 40);
    }
  };

  const drawCloud = () => {
    const c = cloudRef.current;
    if (!c || tokens.length === 0) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = 300;
    const h = 180;
    c.width = w * dpr;
    c.height = h * dpr;
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    if (!analyzed) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Click "Analyze" to see word cloud', w / 2, h / 2);
      return;
    }

    const wordSizes = tokens.map((token, i) => {
      const abs = Math.abs(wordScores[i]);
      const attn = attentionWeights[i] ?? 0;
      return { token, size: Math.max(12, Math.min(40, 12 + abs * 28 + attn * 15)) };
    });

    const positions: { x: number; y: number; token: string; size: number; color: string }[] = [];
    const cx = w / 2;
    const cy = h / 2;
    const placed: { x: number; y: number; r: number }[] = [];

    const sortedWords = [...wordSizes].sort((a, b) => b.size - a.size);

    for (const { token, size } of sortedWords) {
      const score = SENTIMENT_LEXICON[token] ?? 0;
      let color = COLORS.neutral;
      if (isPositiveWord(score)) color = COLORS.positive;
      else if (isNegativeWord(score)) color = COLORS.negative;

      for (let angle = 0; angle < 360; angle += 15) {
        const rad = (angle * Math.PI) / 180;
        const dist = 10 + (angle / 360) * 80;
        const x = cx + Math.cos(rad) * dist;
        const y = cy + Math.sin(rad) * dist;

        const r = size / 2;
        let overlap = false;
        for (const p of placed) {
          const dx = x - p.x;
          const dy = y - p.y;
          if (Math.sqrt(dx * dx + dy * dy) < r + p.r + 4) {
            overlap = true;
            break;
          }
        }
        if (!overlap) {
          positions.push({ x, y, token, size, color });
          placed.push({ x, y, r });
          break;
        }
      }
    }

    for (const p of positions) {
      ctx.fillStyle = p.color;
      ctx.font = `bold ${p.size}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(p.token, p.x, p.y);
    }
  };

  useEffect(() => { drawHeatmap(); }, [tokens, attentionWeights, analyzed, animProgress]);
  useEffect(() => { drawCloud(); }, [tokens, wordScores, attentionWeights, analyzed]);

  // gauge rotation computed directly in render below

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE] overflow-y-auto">
      <div className="lg:col-span-5 space-y-6">
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl shadow-xl">
          <h4 className="text-[#2E251E] font-bold text-lg tracking-tight flex items-center gap-2 mb-4">
            <MessageSquare className="w-5 h-5 text-[#B6532B]" /> NLP Sentiment Analyzer
          </h4>

          <div className="mb-3">
            <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold block mb-1.5">
              <Hash className="w-3 h-3 inline mr-1" /> Preset Sentences
            </label>
            <div className="grid grid-cols-1 gap-1.5">
              {PRESETS.map((p, i) => (
                <button
                  key={i}
                  onClick={() => { setInputText(p); setAnalyzed(false); setAnimProgress(0); }}
                  className={`text-left py-1.5 px-3 rounded-lg text-[10px] font-medium transition-all truncate border ${
                    inputText === p
                      ? 'bg-white border-[#B6532B] text-[#B6532B] shadow-sm'
                      : 'bg-[#FAF6EE] border-[#E5DDD0] text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                >
                  "{p}"
                </button>
              ))}
            </div>
          </div>

          <div className="mb-3">
            <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold block mb-1">
              <MessageSquare className="w-3 h-3 inline mr-1" /> Custom Text
            </label>
            <input
              type="text"
              value={inputText}
              onChange={e => { setInputText(e.target.value); setAnalyzed(false); setAnimProgress(0); }}
              maxLength={120}
              className="w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-lg px-3 py-2 text-xs font-mono text-[#2E251E] focus:outline-none focus:ring-1 focus:ring-[#B6532B]"
              placeholder="Type a sentence (up to 20 words)..."
            />
          </div>

          <div className="mb-4">
            <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold block mb-1.5">
              <Brain className="w-3 h-3 inline mr-1" /> Model
            </label>
            <div className="grid grid-cols-2 gap-1.5 bg-[#FAF6EE] p-1 rounded-xl border border-[#E5DDD0]">
              <button
                onClick={() => setModelType('bow')}
                className={`py-1.5 px-2 rounded-lg text-[10px] font-bold transition-all ${
                  modelType === 'bow'
                    ? 'bg-white shadow-sm text-[#B6532B]'
                    : 'text-[#6E6257] hover:text-[#2E251E]'
                }`}
              >
                <Sigma className="w-3 h-3 inline mr-1" /> Bag-of-Words
              </button>
              <button
                onClick={() => setModelType('transformer')}
                className={`py-1.5 px-2 rounded-lg text-[10px] font-bold transition-all ${
                  modelType === 'transformer'
                    ? 'bg-white shadow-sm text-[#B6532B]'
                    : 'text-[#6E6257] hover:text-[#2E251E]'
                }`}
              >
                <Brain className="w-3 h-3 inline mr-1" /> Transformer
              </button>
            </div>
          </div>

          <button
            onClick={analyze}
            disabled={tokens.length === 0 || animProgress > 0 && animProgress < 1}
            className="w-full flex items-center justify-center gap-2 bg-[#B6532B] text-white text-xs font-bold py-2.5 rounded-xl hover:bg-[#9E4625] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {animProgress > 0 && animProgress < 1 ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Analyzing...
              </>
            ) : (
              <>
                <Send className="w-3.5 h-3.5" /> Analyze Sentiment
              </>
            )}
          </button>
        </div>

        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl shadow-xl">
          <h5 className="text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5">
            <BarChart3 className="w-4 h-4 text-[#B6532B]" /> TF-IDF Keywords
          </h5>
          {analyzed ? (
            <div className="space-y-1.5">
              {tfidfKeywords.length > 0 ? tfidfKeywords.map((kw, i) => {
                const barW = Math.max(10, kw.weight * 200);
                return (
                  <div key={i} className="flex items-center gap-2">
                    <span className="w-16 text-[10px] font-mono text-[#2E251E] truncate text-right">
                      {kw.word}
                    </span>
                    <div className="flex-1 h-4 bg-[#FAF6EE] rounded-full overflow-hidden border border-[#E5DDD0]">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.min(barW, 100)}%`,
                          background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.gold})`,
                        }}
                      />
                    </div>
                    <span className="w-10 text-[9px] font-mono text-[#6E6257]">{kw.weight.toFixed(2)}</span>
                  </div>
                );
              }) : (
                <p className="text-[10px] text-[#6E6257]">No significant keywords found.</p>
              )}
            </div>
          ) : (
            <p className="text-[10px] text-[#6E6257]">Analyze a sentence to see TF-IDF keyword weights.</p>
          )}
        </div>
      </div>

      <div className="lg:col-span-7 space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
            <h5 className="text-[10px] font-mono font-bold text-[#6E6257] uppercase tracking-wider mb-3">
              Sentiment
            </h5>
            <div className="relative w-20 h-20 mb-2">
              <div
                className="w-20 h-20 rounded-full border-4 transition-all duration-700 flex items-center justify-center"
                style={{
                  borderColor: analyzed
                    ? sentimentLabel === 'Positive'
                      ? COLORS.positive
                      : sentimentLabel === 'Negative'
                        ? COLORS.negative
                        : COLORS.neutral
                    : COLORS.border,
                }}
              >
                <div
                  className="w-14 h-14 rounded-full transition-all duration-700 flex items-center justify-center"
                  style={{
                    background: analyzed
                      ? sentimentLabel === 'Positive'
                        ? '#3B7A5715'
                        : sentimentLabel === 'Negative'
                          ? '#B6532B15'
                          : '#6E625715'
                      : '#FAF6EE',
                  }}
                >
                  <span
                    className="text-lg font-bold transition-all duration-700"
                    style={{
                      color: analyzed
                        ? sentimentLabel === 'Positive'
                          ? COLORS.positive
                          : sentimentLabel === 'Negative'
                            ? COLORS.negative
                            : COLORS.neutral
                        : COLORS.border,
                    }}
                  >
                    {analyzed
                      ? sentimentLabel === 'Positive'
                        ? '+'
                        : sentimentLabel === 'Negative'
                          ? '−'
                          : '∼'
                      : '?'}
                  </span>
                </div>
              </div>
            </div>
            <span
              className="text-xs font-bold font-mono transition-all duration-500"
              style={{
                color: analyzed
                  ? sentimentLabel === 'Positive'
                    ? COLORS.positive
                    : sentimentLabel === 'Negative'
                      ? COLORS.negative
                      : COLORS.neutral
                  : COLORS.muted,
              }}
            >
              {analyzed ? sentimentLabel : '—'}
            </span>
          </div>

          <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
            <h5 className="text-[10px] font-mono font-bold text-[#6E6257] uppercase tracking-wider mb-3">
              Confidence
            </h5>
            <div className="relative w-20 h-20 mb-2">
              <svg viewBox="0 0 100 100" className="w-20 h-20 -rotate-90">
                <circle cx="50" cy="50" r="42" fill="none" stroke={COLORS.border} strokeWidth="8" />
                <circle
                  cx="50" cy="50" r="42"
                  fill="none"
                  stroke={analyzed ? COLORS.accent : COLORS.border}
                  strokeWidth="8"
                  strokeDasharray={`${animProgress * 264} 264`}
                  strokeLinecap="round"
                  style={{ transition: 'stroke-dasharray 0.1s ease-out' }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-bold font-mono" style={{ color: COLORS.text }}>
                  {analyzed ? `${Math.round(sentimentConfidence * 100)}%` : '0%'}
                </span>
              </div>
            </div>
            <span className="text-[10px] font-mono text-[#6E6257]">
              {analyzed
                ? sentimentConfidence > 0.7
                  ? 'High'
                  : sentimentConfidence > 0.4
                    ? 'Medium'
                    : 'Low'
                : '—'}
            </span>
          </div>

          <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
            <h5 className="text-[10px] font-mono font-bold text-[#6E6257] uppercase tracking-wider mb-3">
              Model
            </h5>
            <div className="w-20 h-20 mb-2 flex items-center justify-center">
              {modelType === 'transformer' ? (
                <Brain className="w-10 h-10 text-[#B6532B]" />
              ) : (
                <Sigma className="w-10 h-10 text-[#C18C3B]" />
              )}
            </div>
            <span className="text-[10px] font-mono font-bold text-[#2E251E]">
              {modelType === 'transformer' ? 'Deep (Transformer)' : 'Simple (BoW)'}
            </span>
          </div>
        </div>

        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
          <h5 className="text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5">
            <Hash className="w-4 h-4 text-[#B6532B]" /> Attention Heatmap (500×300)
          </h5>
          <canvas
            ref={heatmapRef}
            className="border-2 border-[#E5DDD0] rounded-lg bg-white max-w-full"
            style={{ width: 500, height: 300 }}
          />
        </div>

        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
          <h5 className="text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5">
            <MessageSquare className="w-4 h-4 text-[#C18C3B]" /> Sentiment Word Cloud
          </h5>
          <canvas
            ref={cloudRef}
            className="border-2 border-[#E5DDD0] rounded-lg bg-white max-w-full"
            style={{ width: 300, height: 180 }}
          />
          <div className="flex gap-3 mt-2 text-[9px] font-mono">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: COLORS.positive }} />
              Positive
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: COLORS.negative }} />
              Negative
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: COLORS.neutral }} />
              Neutral
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
