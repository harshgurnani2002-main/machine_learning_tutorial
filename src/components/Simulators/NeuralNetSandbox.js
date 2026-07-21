import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Layers, Plus, Minus, Play, Pause, RotateCcw } from 'lucide-react';
export const NeuralNetSandbox = () => {
    // Network architecture
    const [hiddenLayers, setHiddenLayers] = useState([4, 3]);
    const [activation, setActivation] = useState('sigmoid');
    const [learningRate, setLearningRate] = useState(0.1);
    const [isPlaying, setIsPlaying] = useState(false);
    const [epoch, setEpoch] = useState(0);
    const [loss, setLoss] = useState(0.5);
    const [accuracy, setAccuracy] = useState(0);
    const canvasRef = useRef(null);
    const plotRef = useRef(null);
    // Simulation parameters
    const [weights, setWeights] = useState([]);
    const [biases, setBiases] = useState([]);
    const [activations, setActivations] = useState([]);
    const [pulsePhase, setPulsePhase] = useState(0);
    // Generate classification toy dataset (Circles dataset)
    const [dataset] = useState(() => {
        const pts = [];
        for (let i = 0; i < 80; i++) {
            const angle = Math.random() * Math.PI * 2;
            const r = Math.random() > 0.5 ? Math.random() * 0.5 : 0.8 + Math.random() * 0.4;
            const label = r < 0.6 ? 1 : 0;
            pts.push({
                x1: Math.cos(angle) * r,
                x2: Math.sin(angle) * r,
                label
            });
        }
        return pts;
    });
    const fullArchitecture = [2, ...hiddenLayers, 1];
    // Initialize weights & biases
    const initializeNetwork = () => {
        const newWeights = [];
        const newBiases = [];
        const newActivations = [];
        for (let l = 0; l < fullArchitecture.length; l++) {
            newActivations.push(new Array(fullArchitecture[l]).fill(0));
            if (l === 0)
                continue;
            const layerWeights = [];
            const layerBiases = [];
            for (let j = 0; j < fullArchitecture[l]; j++) {
                const nodeWeights = [];
                for (let i = 0; i < fullArchitecture[l - 1]; i++) {
                    nodeWeights.push((Math.random() - 0.5) * 1.5);
                }
                layerWeights.push(nodeWeights);
                layerBiases.push((Math.random() - 0.5) * 0.5);
            }
            newWeights.push(layerWeights);
            newBiases.push(layerBiases);
        }
        setWeights(newWeights);
        setBiases(newBiases);
        setActivations(newActivations);
        setEpoch(0);
        setLoss(0.5);
        setIsPlaying(false);
    };
    useEffect(() => {
        initializeNetwork();
    }, [hiddenLayers]);
    // Auto-start training on mount for one-click experience
    useEffect(() => {
        const timer = setTimeout(() => setIsPlaying(true), 900);
        return () => clearTimeout(timer);
    }, []);
    // Hidden layers manipulations
    const addHiddenLayer = () => {
        if (hiddenLayers.length >= 3)
            return;
        setHiddenLayers(prev => [...prev, 3]);
    };
    const removeHiddenLayer = () => {
        if (hiddenLayers.length <= 1)
            return;
        setHiddenLayers(prev => prev.slice(0, -1));
    };
    const changeNeurons = (idx, amount) => {
        setHiddenLayers(prev => {
            const next = [...prev];
            const val = next[idx] + amount;
            if (val >= 1 && val <= 6) {
                next[idx] = val;
            }
            return next;
        });
    };
    // Math helper activation functions
    const activate = (z) => {
        if (activation === 'sigmoid')
            return 1 / (1 + Math.exp(-z));
        if (activation === 'tanh')
            return Math.tanh(z);
        return Math.max(0, z); // ReLU
    };
    const activateDerivative = (a) => {
        if (activation === 'sigmoid')
            return a * (1 - a);
        if (activation === 'tanh')
            return 1 - a * a;
        return a > 0 ? 1 : 0; // ReLU derivative
    };
    // Neural network forward propagation
    const forward = (x1, x2, customW = weights, customB = biases) => {
        const act = [[x1, x2]];
        for (let l = 1; l < fullArchitecture.length; l++) {
            const layerAct = [];
            const layerW = customW[l - 1];
            const layerB = customB[l - 1];
            for (let j = 0; j < fullArchitecture[l]; j++) {
                let z = layerB[j];
                for (let i = 0; i < fullArchitecture[l - 1]; i++) {
                    z += act[l - 1][i] * layerW[j][i];
                }
                if (l === fullArchitecture.length - 1) {
                    layerAct.push(1 / (1 + Math.exp(-z))); // output layer always sigmoid
                }
                else {
                    layerAct.push(activate(z));
                }
            }
            act.push(layerAct);
        }
        return act;
    };
    // Run backpropagation training step
    const trainStep = () => {
        if (weights.length === 0)
            return;
        // Accumulate gradients
        const gradW = weights.map(layer => layer.map(node => node.map(() => 0)));
        const gradB = biases.map(layer => layer.map(() => 0));
        let totalLoss = 0;
        let correct = 0;
        // Backprop for each point in dataset
        dataset.forEach(pt => {
            const sampleAct = forward(pt.x1, pt.x2);
            const prediction = sampleAct[sampleAct.length - 1][0];
            // Calculate MSE Loss
            const diff = prediction - pt.label;
            totalLoss += diff * diff;
            if (Math.abs(diff) < 0.5)
                correct++;
            // Deltas list
            const deltas = [];
            for (let l = 0; l < fullArchitecture.length; l++) {
                deltas.push([]);
            }
            // Delta for output layer (MSE + Sigmoid)
            deltas[fullArchitecture.length - 1][0] = diff * prediction * (1 - prediction);
            // Backpropagate deltas
            for (let l = fullArchitecture.length - 2; l > 0; l--) {
                for (let i = 0; i < fullArchitecture[l]; i++) {
                    let error = 0;
                    for (let j = 0; j < fullArchitecture[l + 1]; j++) {
                        error += deltas[l + 1][j] * weights[l][j][i];
                    }
                    deltas[l][i] = error * activateDerivative(sampleAct[l][i]);
                }
            }
            // Calculate weight gradients
            for (let l = 1; l < fullArchitecture.length; l++) {
                for (let j = 0; j < fullArchitecture[l]; j++) {
                    gradB[l - 1][j] += deltas[l][j];
                    for (let i = 0; i < fullArchitecture[l - 1]; i++) {
                        gradW[l - 1][j][i] += deltas[l][j] * sampleAct[l - 1][i];
                    }
                }
            }
        });
        // Update weights and biases with learning rate
        const m = dataset.length;
        const updatedW = weights.map((layer, l) => layer.map((nodeWeights, j) => nodeWeights.map((w, i) => w - (learningRate / m) * gradW[l][j][i])));
        const updatedB = biases.map((layer, l) => layer.map((b, j) => b - (learningRate / m) * gradB[l][j]));
        setWeights(updatedW);
        setBiases(updatedB);
        setLoss(totalLoss / m);
        setAccuracy((correct / m) * 100);
        setEpoch(prev => prev + 1);
        // Forward pass on a default coordinate to animate node activations live
        const demoAct = forward(0.5, 0.5, updatedW, updatedB);
        setActivations(demoAct);
    };
    // Play animation loop
    useEffect(() => {
        let animId;
        if (isPlaying) {
            const run = () => {
                trainStep();
                setPulsePhase(prev => (prev + 0.15) % 1);
                animId = requestAnimationFrame(run);
            };
            animId = requestAnimationFrame(run);
        }
        else {
            setPulsePhase(0);
        }
        return () => cancelAnimationFrame(animId);
    }, [isPlaying, weights, biases, learningRate, activation, epoch]);
    // Render Network Connections graph
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 450;
        const h = canvas.height = 360;
        // Clear with light background cream
        ctx.fillStyle = '#FAF6EE';
        ctx.fillRect(0, 0, w, h);
        // Grid overlays
        ctx.strokeStyle = '#E5DDD0';
        ctx.lineWidth = 0.5;
        for (let x = 30; x < w; x += 30) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 30; y < h; y += 30) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        // Layer spacing
        const layerCount = fullArchitecture.length;
        const layerGap = w / (layerCount + 1);
        const nodePositions = [];
        // Calculate node coordinates
        for (let l = 0; l < layerCount; l++) {
            const nodes = fullArchitecture[l];
            const x = layerGap * (l + 1);
            const verticalGap = h / (nodes + 1);
            const layerPos = [];
            for (let n = 0; n < nodes; n++) {
                layerPos.push({ x, y: verticalGap * (n + 1) });
            }
            nodePositions.push(layerPos);
        }
        // Draw lines (synapses)
        for (let l = 1; l < layerCount; l++) {
            const prevNodes = nodePositions[l - 1];
            const currNodes = nodePositions[l];
            const layerW = weights[l - 1];
            if (!layerW)
                continue;
            for (let j = 0; j < currNodes.length; j++) {
                for (let i = 0; i < prevNodes.length; i++) {
                    const wVal = layerW[j]?.[i] || 0;
                    const px = prevNodes[i];
                    const py = currNodes[j];
                    ctx.beginPath();
                    ctx.moveTo(px.x, px.y);
                    ctx.lineTo(py.x, py.y);
                    // Weight colors: Terracotta positive, Golden Ochre negative
                    ctx.strokeStyle = wVal > 0
                        ? `rgba(182, 83, 43, ${Math.min(0.8, Math.abs(wVal) * 0.7)})`
                        : `rgba(193, 140, 59, ${Math.min(0.8, Math.abs(wVal) * 0.7)})`;
                    ctx.lineWidth = Math.min(5, Math.max(0.5, Math.abs(wVal) * 2.5));
                    ctx.stroke();
                    // Draw floating pulse lights during training
                    if (isPlaying && pulsePhase > 0) {
                        const pulseX = px.x + (py.x - px.x) * pulsePhase;
                        const pulseY = px.y + (py.y - px.y) * pulsePhase;
                        ctx.beginPath();
                        ctx.arc(pulseX, pulseY, 2.5, 0, Math.PI * 2);
                        ctx.fillStyle = '#2E251E'; // Charcoal moving pulse
                        ctx.fill();
                    }
                }
            }
        }
        // Draw nodes (neurons)
        for (let l = 0; l < layerCount; l++) {
            const nodes = nodePositions[l];
            const layerAct = activations[l] || [];
            nodes.forEach((pos, idx) => {
                const val = layerAct[idx] || 0;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 14, 0, Math.PI * 2);
                // Fill based on activation value (glowing cream/sand/terracotta)
                ctx.fillStyle = `rgba(182, 83, 43, ${0.05 + Math.abs(val) * 0.5})`;
                ctx.strokeStyle = '#2E251E'; // Charcoal border
                ctx.lineWidth = 1.5;
                ctx.fill();
                ctx.stroke();
                // Layer labels
                ctx.fillStyle = '#2E251E';
                ctx.font = 'bold 8px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(1), pos.x, pos.y);
            });
        }
    }, [weights, activations, isPlaying, pulsePhase]);
    // Render classification decision boundary background map
    useEffect(() => {
        const canvas = plotRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const size = canvas.width = 180;
        const hSize = canvas.height = 180;
        // Skip drawing map pixels if weights are not ready
        if (weights.length > 0) {
            const imgData = ctx.createImageData(size, hSize);
            for (let y = 0; y < hSize; y++) {
                for (let x = 0; x < size; x++) {
                    // Normalize to [-1.2, 1.2]
                    const nx1 = ((x / size) - 0.5) * 2.4;
                    const nx2 = ((y / hSize) - 0.5) * 2.4;
                    const outAct = forward(nx1, nx2);
                    const prob = outAct[outAct.length - 1][0];
                    // Map probability to terracotta/ochre background colors
                    const idx = (y * size + x) * 4;
                    // Class 0 (Ochre) vs Class 1 (Terracotta) background
                    const r = Math.floor(193 * (1 - prob) + 182 * prob);
                    const g = Math.floor(140 * (1 - prob) + 83 * prob);
                    const b = Math.floor(59 * (1 - prob) + 43 * prob);
                    imgData.data[idx] = r;
                    imgData.data[idx + 1] = g;
                    imgData.data[idx + 2] = b;
                    imgData.data[idx + 3] = 40; // opacity
                }
            }
            ctx.putImageData(imgData, 0, 0);
        }
        else {
            ctx.fillStyle = '#FAF6EE';
            ctx.fillRect(0, 0, size, hSize);
        }
        // Grid borders
        ctx.strokeStyle = 'rgba(46, 37, 30, 0.05)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(size / 2, 0);
        ctx.lineTo(size / 2, hSize);
        ctx.moveTo(0, hSize / 2);
        ctx.lineTo(size, hSize / 2);
        ctx.stroke();
        // Draw Scatter points
        dataset.forEach(pt => {
            // Map to pixels
            const px = ((pt.x1 / 2.4) + 0.5) * size;
            const py = ((pt.x2 / 2.4) + 0.5) * hSize;
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, Math.PI * 2);
            ctx.fillStyle = pt.label === 1 ? '#B6532B' : '#C18C3B'; // Terracotta vs Ochre
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 1;
            ctx.fill();
            ctx.stroke();
        });
    }, [weights, dataset]);
    return (_jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-12 gap-6 p-4", children: [_jsxs("div", { className: "lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between space-y-6", children: [_jsxs("div", { children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2", children: [_jsx(Layers, { className: "w-5 h-5 text-[#B6532B]" }), " Network Builder"] }), _jsx("p", { className: "text-[#6E6257] text-xs mb-5", children: "Assemble layers, add nodes, and watch activations adapt forward & backward." }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "flex justify-between items-center bg-[#FAF6EE] p-3 rounded-xl border border-[#E5DDD0]", children: [_jsxs("span", { className: "text-xs text-[#2E251E] font-mono font-semibold", children: ["Hidden Layers (", hiddenLayers.length, ")"] }), _jsxs("div", { className: "flex gap-2", children: [_jsx("button", { onClick: removeHiddenLayer, className: "p-1 rounded-lg bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#6E6257] border border-[#E5DDD0] transition-all cursor-pointer", children: _jsx(Minus, { className: "w-3.5 h-3.5" }) }), _jsx("button", { onClick: addHiddenLayer, className: "p-1 rounded-lg bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#6E6257] border border-[#E5DDD0] transition-all cursor-pointer", children: _jsx(Plus, { className: "w-3.5 h-3.5" }) })] })] }), _jsxs("div", { className: "space-y-2", children: [_jsx("span", { className: "text-[10px] text-[#6E6257] font-mono uppercase tracking-wider block font-bold", children: "Layer Nodes Configuration" }), hiddenLayers.map((nodes, idx) => (_jsxs("div", { className: "flex justify-between items-center bg-[#FAF6EE]/50 px-3 py-2 rounded-xl border border-[#E5DDD0] text-xs", children: [_jsxs("span", { className: "text-[#6E6257] font-mono", children: ["Hidden Layer ", idx + 1] }), _jsxs("div", { className: "flex items-center gap-3", children: [_jsx("button", { onClick: () => changeNeurons(idx, -1), className: "p-1 rounded-md bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#6E6257] border border-[#E5DDD0] transition-all cursor-pointer", children: _jsx(Minus, { className: "w-3 h-3" }) }), _jsx("span", { className: "font-mono text-[#B6532B] font-bold w-4 text-center", children: nodes }), _jsx("button", { onClick: () => changeNeurons(idx, 1), className: "p-1 rounded-md bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#6E6257] border border-[#E5DDD0] transition-all cursor-pointer", children: _jsx(Plus, { className: "w-3 h-3" }) })] })] }, idx)))] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold", children: "Activation" }), _jsxs("select", { value: activation, onChange: (e) => setActivation(e.target.value), className: "w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-3 py-2 text-xs text-[#2E251E] focus:outline-none focus:border-[#B6532B] font-mono", children: [_jsx("option", { value: "sigmoid", children: "Sigmoid" }), _jsx("option", { value: "tanh", children: "Tanh" }), _jsx("option", { value: "relu", children: "ReLU" })] })] }), _jsxs("div", { children: [_jsx("label", { className: "text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold", children: "Learning Rate (\u03B7)" }), _jsxs("select", { value: learningRate, onChange: (e) => setLearningRate(parseFloat(e.target.value)), className: "w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-3 py-2 text-xs text-[#2E251E] focus:outline-none focus:border-[#B6532B] font-mono", children: [_jsx("option", { value: "0.05", children: "0.05" }), _jsx("option", { value: "0.1", children: "0.10" }), _jsx("option", { value: "0.25", children: "0.25" }), _jsx("option", { value: "0.5", children: "0.50" })] })] })] })] })] }), _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 grid grid-cols-3 text-center text-xs font-mono text-[#2E251E]", children: [_jsxs("div", { children: [_jsx("span", { className: "text-[#6E6257] text-[9px] block", children: "EPOCH" }), _jsx("span", { className: "font-bold", children: epoch })] }), _jsxs("div", { children: [_jsx("span", { className: "text-[#6E6257] text-[9px] block", children: "TRAIN LOSS" }), _jsx("span", { className: "text-[#B6532B] font-bold", children: loss.toFixed(4) })] }), _jsxs("div", { children: [_jsx("span", { className: "text-[#6E6257] text-[9px] block", children: "ACCURACY" }), _jsxs("span", { className: "text-[#3B7A57] font-bold", children: [accuracy.toFixed(1), "%"] })] })] }), _jsxs("div", { className: "flex gap-2", children: [_jsxs("button", { onClick: () => setIsPlaying(!isPlaying), className: `flex-1 py-3 rounded-xl flex items-center justify-center gap-2 text-xs font-semibold shadow-sm transition-all cursor-pointer ${isPlaying
                                    ? 'bg-amber-600 border-amber-600 text-white'
                                    : 'bg-[#B6532B] border-[#B6532B] text-white hover:bg-[#B6532B]/95'}`, children: [isPlaying ? _jsx(Pause, { className: "w-4 h-4" }) : _jsx(Play, { className: "w-4 h-4" }), isPlaying ? 'Pause Training' : 'Run Backpropagation'] }), _jsx("button", { onClick: initializeNetwork, className: "p-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] transition-colors cursor-pointer", title: "Reset Net Weights", children: _jsx(RotateCcw, { className: "w-4 h-4" }) })] })] }), _jsxs("div", { className: "lg:col-span-7 flex flex-col md:flex-row gap-4 items-center justify-center", children: [_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-3 rounded-2xl shadow-inner shrink-0", children: [_jsx("canvas", { ref: canvasRef, className: "w-[300px] h-[240px] sm:w-[380px] sm:h-[300px] rounded-xl border border-[#E5DDD0]" }), _jsx("span", { className: "block text-center text-[9px] font-mono text-[#6E6257] mt-1 font-semibold", children: "Synaptic Weight Streams (w_i) & Firing Halos" })] }), _jsxs("div", { className: "flex flex-col gap-3 items-center", children: [_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-3 rounded-2xl flex flex-col items-center shadow-inner", children: [_jsx("canvas", { ref: plotRef, className: "w-[140px] h-[140px] sm:w-[160px] sm:h-[160px] rounded-lg border border-[#E5DDD0]" }), _jsx("span", { className: "block text-[8px] font-mono text-[#6E6257] mt-1.5 uppercase tracking-wide font-bold", children: "Decision Spaces" })] }), _jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-2.5 rounded-xl text-[9px] font-mono text-[#6E6257] w-full max-w-[160px] space-y-1", children: [_jsxs("div", { className: "flex justify-between", children: [_jsx("span", { children: "Classifier:" }), _jsx("span", { className: "text-[#B6532B] font-bold", children: "MLP Net" })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { children: "Dataset:" }), _jsx("span", { className: "text-[#C18C3B] font-bold", children: "Concentric Rings" })] })] })] })] })] }));
};
