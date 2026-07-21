import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Grid, User, Gem, Flame } from 'lucide-react';
const GRID_SIZE = 5;
const ACTIONS = [
    { dx: 0, dy: -1, name: 'up' },
    { dx: 0, dy: 1, name: 'down' },
    { dx: -1, dy: 0, name: 'left' },
    { dx: 1, dy: 0, name: 'right' }
];
export const QLearningSimulator = () => {
    const [learningRate, setLearningRate] = useState(0.1);
    const [discountFactor, setDiscountFactor] = useState(0.9);
    const [explorationRate, setExplorationRate] = useState(0.2);
    const [isPlaying, setIsPlaying] = useState(false);
    const [epoch, setEpoch] = useState(0);
    // Q-table: [x][y][actionIndex]
    const [qTable, setQTable] = useState([]);
    const [agentPos, setAgentPos] = useState({ x: 0, y: 0 });
    const startPos = { x: 0, y: 0 };
    const goalPos = { x: 4, y: 4 };
    const obstaclePos = { x: 2, y: 2 };
    const initQTable = () => {
        const table = [];
        for (let x = 0; x < GRID_SIZE; x++) {
            table[x] = [];
            for (let y = 0; y < GRID_SIZE; y++) {
                table[x][y] = [0, 0, 0, 0];
            }
        }
        setQTable(table);
        setAgentPos({ ...startPos });
        setEpoch(0);
        setIsPlaying(false);
    };
    useEffect(() => {
        initQTable();
    }, []);
    // Auto-start training on mount for one-click experience
    useEffect(() => {
        const timer = setTimeout(() => setIsPlaying(true), 600);
        return () => clearTimeout(timer);
    }, []);
    const step = () => {
        setQTable(prevQ => {
            const newQ = [...prevQ.map(row => [...row.map(cell => [...cell])])];
            let { x, y } = agentPos;
            let newX = x;
            let newY = y;
            // Choose action (epsilon-greedy)
            let actionIdx = 0;
            if (Math.random() < explorationRate) {
                actionIdx = Math.floor(Math.random() * 4);
            }
            else {
                // Exploit
                let maxQ = -Infinity;
                const qValues = newQ[x][y];
                for (let i = 0; i < 4; i++) {
                    if (qValues[i] > maxQ) {
                        maxQ = qValues[i];
                        actionIdx = i;
                    }
                }
                // Handle ties randomly to prevent getting stuck early on
                const bestActions = [];
                for (let i = 0; i < 4; i++) {
                    if (qValues[i] === maxQ)
                        bestActions.push(i);
                }
                actionIdx = bestActions[Math.floor(Math.random() * bestActions.length)];
            }
            const action = ACTIONS[actionIdx];
            newX = Math.max(0, Math.min(GRID_SIZE - 1, x + action.dx));
            newY = Math.max(0, Math.min(GRID_SIZE - 1, y + action.dy));
            let reward = -0.1; // small penalty for each step
            let done = false;
            if (newX === obstaclePos.x && newY === obstaclePos.y) {
                reward = -10; // hitting obstacle
                newX = x; // don't move
            }
            else if (newX === goalPos.x && newY === goalPos.y) {
                reward = 10;
                done = true;
            }
            // Q-learning update
            const currentQ = newQ[x][y][actionIdx];
            const maxNextQ = Math.max(...newQ[newX][newY]);
            newQ[x][y][actionIdx] = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
            if (done) {
                setAgentPos({ ...startPos });
                setEpoch(prev => prev + 1);
            }
            else {
                setAgentPos({ x: newX, y: newY });
            }
            return newQ;
        });
    };
    useEffect(() => {
        let animId;
        if (isPlaying && qTable.length > 0) {
            const run = () => {
                // run multiple steps per frame to speed up training visualization
                for (let i = 0; i < 5; i++) {
                    step();
                }
                animId = requestAnimationFrame(run);
            };
            animId = requestAnimationFrame(run);
        }
        return () => cancelAnimationFrame(animId);
    }, [isPlaying, agentPos, qTable, learningRate, discountFactor, explorationRate]);
    return (_jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE]", children: [_jsxs("div", { className: "lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-between shadow-xl", children: [_jsxs("div", { className: "space-y-6", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3", children: [_jsx(Grid, { className: "w-6 h-6 text-[#B6532B]" }), " Q-Learning"] }), _jsxs("p", { className: "text-[#6E6257] text-sm leading-relaxed", children: ["Watch the agent learn to navigate the grid. It updates its ", _jsx("span", { className: "font-semibold text-[#B6532B]", children: "Q-Table" }), " using the Bellman Equation to find the shortest path to the goal while avoiding obstacles."] }), _jsxs("div", { className: "space-y-4 bg-[#FAF6EE] p-4 rounded-xl border border-[#E5DDD0]", children: [_jsxs("div", { children: [_jsxs("div", { className: "flex justify-between mb-1", children: [_jsx("label", { className: "text-[10px] text-[#6E6257] font-mono uppercase font-bold", children: "Learning Rate (\u03B1)" }), _jsx("span", { className: "text-[10px] font-mono font-bold text-[#B6532B]", children: learningRate.toFixed(2) })] }), _jsx("input", { type: "range", min: "0.01", max: "1", step: "0.01", value: learningRate, onChange: e => setLearningRate(parseFloat(e.target.value)), className: "w-full accent-[#B6532B]" })] }), _jsxs("div", { children: [_jsxs("div", { className: "flex justify-between mb-1", children: [_jsx("label", { className: "text-[10px] text-[#6E6257] font-mono uppercase font-bold", children: "Discount Factor (\u03B3)" }), _jsx("span", { className: "text-[10px] font-mono font-bold text-[#B6532B]", children: discountFactor.toFixed(2) })] }), _jsx("input", { type: "range", min: "0", max: "0.99", step: "0.01", value: discountFactor, onChange: e => setDiscountFactor(parseFloat(e.target.value)), className: "w-full accent-[#B6532B]" })] }), _jsxs("div", { children: [_jsxs("div", { className: "flex justify-between mb-1", children: [_jsx("label", { className: "text-[10px] text-[#6E6257] font-mono uppercase font-bold", children: "Exploration (\u03B5)" }), _jsx("span", { className: "text-[10px] font-mono font-bold text-[#B6532B]", children: explorationRate.toFixed(2) })] }), _jsx("input", { type: "range", min: "0", max: "1", step: "0.01", value: explorationRate, onChange: e => setExplorationRate(parseFloat(e.target.value)), className: "w-full accent-[#B6532B]" })] })] })] }), _jsxs("div", { className: "mt-6 space-y-4", children: [_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 flex justify-between items-center text-sm font-mono text-[#2E251E]", children: [_jsx("span", { className: "text-[#6E6257] font-bold", children: "EPISODES" }), _jsx("span", { className: "text-[#B6532B] font-bold text-lg", children: epoch })] }), _jsxs("div", { className: "flex gap-2", children: [_jsxs("button", { onClick: () => setIsPlaying(!isPlaying), className: `flex-1 py-3 rounded-xl flex items-center justify-center gap-2 text-sm font-bold transition-all shadow-sm ${isPlaying
                                            ? 'bg-amber-600 border-amber-600 text-white'
                                            : 'bg-[#B6532B] border-[#B6532B] text-white hover:bg-[#B6532B]/90'}`, children: [isPlaying ? _jsx(Pause, { className: "w-5 h-5" }) : _jsx(Play, { className: "w-5 h-5" }), isPlaying ? 'Pause' : 'Train Agent'] }), _jsx("button", { onClick: initQTable, className: "p-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] transition-colors", title: "Reset Environment", children: _jsx(RotateCcw, { className: "w-5 h-5" }) })] })] })] }), _jsx("div", { className: "lg:col-span-7 flex items-center justify-center", children: _jsxs("div", { className: "bg-[#F4EFE6] p-6 rounded-3xl border border-[#E5DDD0] shadow-inner inline-block", children: [_jsx("div", { className: "grid gap-1 relative", style: {
                                gridTemplateColumns: `repeat(${GRID_SIZE}, minmax(0, 1fr))`,
                                width: 'min(100%, 400px)',
                                aspectRatio: '1/1'
                            }, children: qTable.length > 0 && Array.from({ length: GRID_SIZE }).map((_, y) => (Array.from({ length: GRID_SIZE }).map((_, x) => {
                                const isAgent = agentPos.x === x && agentPos.y === y;
                                const isGoal = goalPos.x === x && goalPos.y === y;
                                const isObstacle = obstaclePos.x === x && obstaclePos.y === y;
                                const maxQ = Math.max(...qTable[x][y]);
                                // Color intensity based on max Q value (normalized roughly to expected max reward)
                                const intensity = Math.max(0, Math.min(1, (maxQ + 1) / 11));
                                let bgColor = '#FAF6EE';
                                if (!isObstacle && !isGoal && maxQ > 0) {
                                    bgColor = `rgba(182, 83, 43, ${intensity * 0.8})`;
                                }
                                return (_jsxs("div", { className: `
                      w-full h-full relative rounded flex items-center justify-center border
                      ${isGoal ? 'bg-[#3B7A57] border-[#2E5B42]' : ''}
                      ${isObstacle ? 'bg-[#FAF6EE] border-[#E5DDD0]' : ''}
                      ${!isGoal && !isObstacle ? 'border-[#E5DDD0]' : ''}
                    `, style: { backgroundColor: (!isGoal && !isObstacle) ? bgColor : undefined }, children: [!isGoal && !isObstacle && maxQ > 0.1 && (_jsx("span", { className: "text-[8px] font-mono font-bold text-white opacity-80 mix-blend-difference", children: maxQ.toFixed(1) })), isAgent && (_jsx("div", { className: "absolute inset-0 m-auto w-4/5 h-4/5 bg-[#C18C3B] rounded-full shadow-md z-10 border-2 border-white flex items-center justify-center", children: _jsx(User, { className: "w-3/5 h-3/5 text-white" }) })), isGoal && !isAgent && (_jsx(Gem, { className: "w-3/5 h-3/5 text-white" })), isObstacle && !isAgent && (_jsx(Flame, { className: "w-3/5 h-3/5 text-[#B6532B]" }))] }, `${x}-${y}`));
                            }))) }), _jsxs("div", { className: "mt-4 flex justify-between text-xs font-mono text-[#6E6257] font-bold", children: [_jsxs("span", { className: "flex items-center gap-1", children: [_jsx("div", { className: "w-4 h-4 bg-[#C18C3B] rounded-full flex items-center justify-center", children: _jsx(User, { className: "w-3 h-3 text-white" }) }), " Agent"] }), _jsxs("span", { className: "flex items-center gap-1", children: [_jsx(Flame, { className: "w-4 h-4 text-[#B6532B]" }), " Fire"] }), _jsxs("span", { className: "flex items-center gap-1", children: [_jsx("div", { className: "w-4 h-4 bg-[#3B7A57] rounded flex items-center justify-center", children: _jsx(Gem, { className: "w-3 h-3 text-white" }) }), " Treasure"] }), _jsxs("span", { className: "flex items-center gap-1", children: [_jsx("div", { className: "w-3 h-3 bg-[#B6532B] opacity-50 rounded" }), " High Q-Value"] })] })] }) })] }));
};
