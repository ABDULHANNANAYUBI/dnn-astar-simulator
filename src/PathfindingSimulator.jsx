import { useState, useEffect, useCallback, useRef, useMemo } from "react";

// ─── Priority Queue ───
class MinHeap {
  constructor() { this.heap = []; }
  push(item) {
    this.heap.push(item);
    let i = this.heap.length - 1;
    while (i > 0) {
      const p = Math.floor((i - 1) / 2);
      if (this.heap[p][0] <= this.heap[i][0]) break;
      [this.heap[p], this.heap[i]] = [this.heap[i], this.heap[p]];
      i = p;
    }
  }
  pop() {
    if (this.heap.length === 0) return null;
    const top = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = last;
      let i = 0;
      while (true) {
        let smallest = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < this.heap.length && this.heap[l][0] < this.heap[smallest][0]) smallest = l;
        if (r < this.heap.length && this.heap[r][0] < this.heap[smallest][0]) smallest = r;
        if (smallest === i) break;
        [this.heap[smallest], this.heap[i]] = [this.heap[i], this.heap[smallest]];
        i = smallest;
      }
    }
    return top;
  }
  get size() { return this.heap.length; }
}

// ─── Graph Generation ───
function generateGraph(nodeCount, edgeDensity, seed = 42) {
  const rng = mulberry32(seed);
  const nodes = {};
  const edges = [];
  const adj = {};

  // Place nodes in clusters for realistic road network feel
  const clusterCount = Math.max(3, Math.floor(nodeCount / 8));
  const clusterCenters = [];
  for (let i = 0; i < clusterCount; i++) {
    clusterCenters.push({ x: rng() * 900 + 50, y: rng() * 550 + 25 });
  }

  for (let i = 0; i < nodeCount; i++) {
    const cluster = clusterCenters[Math.floor(rng() * clusterCount)];
    const angle = rng() * Math.PI * 2;
    const dist = rng() * 120 + 10;
    nodes[i] = {
      id: i,
      x: Math.max(20, Math.min(980, cluster.x + Math.cos(angle) * dist)),
      y: Math.max(20, Math.min(580, cluster.y + Math.sin(angle) * dist)),
    };
    adj[i] = [];
  }

  // Connect nearby nodes
  const nodeArr = Object.values(nodes);
  for (let i = 0; i < nodeArr.length; i++) {
    const distances = [];
    for (let j = 0; j < nodeArr.length; j++) {
      if (i === j) continue;
      const dx = nodeArr[i].x - nodeArr[j].x;
      const dy = nodeArr[i].y - nodeArr[j].y;
      distances.push({ j, dist: Math.sqrt(dx * dx + dy * dy) });
    }
    distances.sort((a, b) => a.dist - b.dist);
    const connectCount = Math.max(1, Math.floor(rng() * edgeDensity * 3) + 1);
    for (let k = 0; k < Math.min(connectCount, distances.length); k++) {
      const j = distances[k].j;
      const length = distances[k].dist;
      if (length > 300) continue;
      const maxspeed = 30 + rng() * 90;
      const traffic = rng();
      const safety = 0.3 + rng() * 0.7;
      const cost = length * (0.001 + rng() * 0.003);
      const edge = {
        u: i, v: j, length, maxspeed, traffic, safety, cost,
        travel_time: length / (maxspeed / 3.6)
      };
      edges.push(edge);
      adj[i].push(edges.length - 1);
    }
  }

  return { nodes, edges, adj };
}

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    var t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ─── MCDM ───
function computeMinMax(edges) {
  const mm = { traffic: [Infinity, -Infinity], safety: [Infinity, -Infinity], cost: [Infinity, -Infinity] };
  edges.forEach(e => {
    for (const k of ['traffic', 'safety', 'cost']) {
      mm[k][0] = Math.min(mm[k][0], e[k]);
      mm[k][1] = Math.max(mm[k][1], e[k]);
    }
  });
  for (const k in mm) {
    if (mm[k][0] === Infinity) mm[k] = [0, 1];
    if (mm[k][0] === mm[k][1]) mm[k][1] += 1;
  }
  return mm;
}

function mcdmCost(edge, weights, mm) {
  const tr = mm.traffic[1] - mm.traffic[0];
  const sr = mm.safety[1] - mm.safety[0];
  const cr = mm.cost[1] - mm.cost[0];
  const nt = tr > 0 ? (edge.traffic - mm.traffic[0]) / tr : 0.5;
  const ns = sr > 0 ? (mm.safety[1] - edge.safety) / sr : 0.5;
  const nc = cr > 0 ? (edge.cost - mm.cost[0]) / cr : 0.5;
  return Math.max(1e-6, weights.traffic * Math.max(0, nt) + weights.safety * Math.max(0, ns) + weights.cost * Math.max(0, nc));
}

// ─── Heuristics ───
function euclideanH(u, goal, nodes) {
  const dx = nodes[u].x - nodes[goal].x;
  const dy = nodes[u].y - nodes[goal].y;
  return Math.sqrt(dx * dx + dy * dy) * 0.001;
}

// Simple learned heuristic simulator (polynomial regression on distance features)
function trainSimpleModel(graph, weights, mm) {
  const { nodes, edges, adj } = graph;
  const nodeIds = Object.keys(nodes).map(Number);
  const rng = mulberry32(123);
  const samples = [];
  
  // Sample random pairs and compute actual shortest path cost
  const sampleCount = Math.min(200, nodeIds.length * 2);
  for (let s = 0; s < sampleCount; s++) {
    const src = nodeIds[Math.floor(rng() * nodeIds.length)];
    const tgt = nodeIds[Math.floor(rng() * nodeIds.length)];
    if (src === tgt) continue;
    
    // Quick Dijkstra
    const dist = {};
    const heap = new MinHeap();
    dist[src] = 0;
    heap.push([0, src]);
    let found = false;
    
    while (heap.size > 0) {
      const [d, u] = heap.pop();
      if (u === tgt) { found = true; break; }
      if (d > (dist[u] ?? Infinity)) continue;
      for (const ei of (adj[u] || [])) {
        const e = edges[ei];
        const w = mcdmCost(e, weights, mm);
        const nd = d + w;
        if (nd < (dist[e.v] ?? Infinity)) {
          dist[e.v] = nd;
          heap.push([nd, e.v]);
        }
      }
    }
    
    if (found && dist[tgt] !== undefined) {
      const dx = nodes[src].x - nodes[tgt].x;
      const dy = nodes[src].y - nodes[tgt].y;
      const eucDist = Math.sqrt(dx * dx + dy * dy);
      samples.push({ eucDist, cost: dist[tgt] });
    }
  }
  
  if (samples.length < 5) return null;
  
  // Simple linear regression: cost ≈ a * eucDist + b
  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  for (const s of samples) {
    sumX += s.eucDist;
    sumY += s.cost;
    sumXX += s.eucDist * s.eucDist;
    sumXY += s.eucDist * s.cost;
  }
  const n = samples.length;
  const a = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const b = (sumY - a * sumX) / n;
  
  return { a: Math.max(0, a), b: Math.max(0, b), sampleCount: n };
}

function dnnH(u, goal, nodes, model) {
  if (!model) return 0;
  const dx = nodes[u].x - nodes[goal].x;
  const dy = nodes[u].y - nodes[goal].y;
  const eucDist = Math.sqrt(dx * dx + dy * dy);
  return Math.max(0, (model.a * eucDist + model.b) * 0.9);
}

// ─── A* with step tracking ───
function astarSteps(graph, start, goal, weights, mm, heuristicFn) {
  const { nodes, edges, adj } = graph;
  const steps = [];
  const gScore = {};
  const cameFrom = {};
  const heap = new MinHeap();
  const visited = new Set();
  
  gScore[start] = 0;
  heap.push([heuristicFn(start, goal, nodes), start]);
  
  while (heap.size > 0) {
    const [fVal, current] = heap.pop();
    
    if (visited.has(current)) continue;
    visited.add(current);
    
    steps.push({
      type: 'visit',
      node: current,
      fScore: fVal,
      gScore: gScore[current],
      visited: new Set(visited),
      frontier: new Set(heap.heap.map(h => h[1]))
    });
    
    if (current === goal) {
      const path = [];
      let c = goal;
      while (c !== undefined) {
        path.unshift(c);
        c = cameFrom[c];
      }
      let totalCost = 0;
      for (let i = 0; i < path.length - 1; i++) {
        const ei = (adj[path[i]] || []).find(idx => edges[idx].v === path[i + 1]);
        if (ei !== undefined) totalCost += mcdmCost(edges[ei], weights, mm);
      }
      steps.push({ type: 'done', path, cost: totalCost, visited: new Set(visited) });
      return steps;
    }
    
    for (const ei of (adj[current] || [])) {
      const e = edges[ei];
      if (visited.has(e.v)) continue;
      const w = mcdmCost(e, weights, mm);
      const tentative = gScore[current] + w;
      if (tentative < (gScore[e.v] ?? Infinity)) {
        gScore[e.v] = tentative;
        cameFrom[e.v] = current;
        const f = tentative + heuristicFn(e.v, goal, nodes);
        heap.push([f, e.v]);
      }
    }
  }
  
  steps.push({ type: 'no_path', visited: new Set(visited) });
  return steps;
}

// ─── PRESETS ───
const WEIGHT_PRESETS = {
  traffic_focused: { traffic: 0.8, safety: 0.1, cost: 0.1 },
  cost_focused: { traffic: 0.1, safety: 0.1, cost: 0.8 },
  safety_focused: { traffic: 0.1, safety: 0.8, cost: 0.1 },
  balanced: { traffic: 0.33, safety: 0.33, cost: 0.33 },
};

// ─── SVG Graph Renderer ───
function GraphCanvas({ graph, width, height, step, path, startNode, goalNode, onNodeClick, selectMode }) {
  const svgRef = useRef(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const visited = step?.visited || new Set();
  const frontier = step?.frontier || new Set();
  const currentPath = step?.type === 'done' ? step.path : (path || []);
  const pathSet = new Set(currentPath);
  const pathEdges = new Set();
  for (let i = 0; i < currentPath.length - 1; i++) {
    pathEdges.add(`${currentPath[i]}-${currentPath[i + 1]}`);
  }

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setTransform(t => ({
      ...t,
      scale: Math.max(0.3, Math.min(5, t.scale * delta))
    }));
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (svg) svg.addEventListener('wheel', handleWheel, { passive: false });
    return () => { if (svg) svg.removeEventListener('wheel', handleWheel); };
  }, [handleWheel]);

  const handleMouseDown = (e) => {
    if (e.target.tagName === 'circle') return;
    setDragging(true);
    setDragStart({ x: e.clientX - transform.x, y: e.clientY - transform.y });
  };

  const handleMouseMove = (e) => {
    if (!dragging) return;
    setTransform(t => ({ ...t, x: e.clientX - dragStart.x, y: e.clientY - dragStart.y }));
  };

  const handleMouseUp = () => setDragging(false);

  const { nodes, edges } = graph;
  const nodeArr = Object.values(nodes);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ background: '#0a0e17', borderRadius: '8px', cursor: dragging ? 'grabbing' : 'grab' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <marker id="arrowGray" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="6" markerHeight="6" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#1e2a3a" />
        </marker>
        <marker id="arrowPath" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="8" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#00e5ff" />
        </marker>
      </defs>
      <g transform={`translate(${transform.x},${transform.y}) scale(${transform.scale})`}>
        {/* Edges */}
        {edges.map((e, i) => {
          const u = nodes[e.u], v = nodes[e.v];
          if (!u || !v) return null;
          const isPath = pathEdges.has(`${e.u}-${e.v}`);
          return (
            <line
              key={`e-${i}`}
              x1={u.x} y1={u.y} x2={v.x} y2={v.y}
              stroke={isPath ? '#00e5ff' : '#1e2a3a'}
              strokeWidth={isPath ? 2.5 : 0.7}
              opacity={isPath ? 1 : 0.4}
              markerEnd={isPath ? 'url(#arrowPath)' : ''}
              filter={isPath ? 'url(#glow)' : ''}
            />
          );
        })}
        {/* Nodes */}
        {nodeArr.map(n => {
          const isStart = n.id === startNode;
          const isGoal = n.id === goalNode;
          const isVisited = visited.has(n.id);
          const isFrontier = frontier.has(n.id);
          const isOnPath = pathSet.has(n.id);

          let fill = '#1a2332';
          let r = 4;
          let strokeColor = 'none';
          let strokeW = 0;

          if (isFrontier) { fill = '#ffd54f'; r = 5; }
          if (isVisited) { fill = '#5c6bc0'; r = 5; }
          if (isOnPath) { fill = '#00e5ff'; r = 6; strokeColor = '#00e5ff'; strokeW = 1; }
          if (isStart) { fill = '#00e676'; r = 10; strokeColor = '#00c853'; strokeW = 2; }
          if (isGoal) { fill = '#ff1744'; r = 10; strokeColor = '#d50000'; strokeW = 2; }

          return (
            <circle
              key={`n-${n.id}`}
              cx={n.x} cy={n.y} r={r}
              fill={fill}
              stroke={strokeColor}
              strokeWidth={strokeW}
              style={{ cursor: selectMode ? 'crosshair' : 'pointer', transition: 'fill 0.2s' }}
              onClick={(e) => { e.stopPropagation(); onNodeClick(n.id); }}
              filter={(isStart || isGoal) ? 'url(#glow)' : ''}
            >
              <title>Node {n.id}</title>
            </circle>
          );
        })}
      </g>
    </svg>
  );
}

// ─── Stats Bar ───
function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: '8px',
      padding: '12px 16px',
      minWidth: '120px',
      flex: 1,
    }}>
      <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1.5px', color: '#6b7b8d', marginBottom: '4px', fontFamily: "'JetBrains Mono', monospace" }}>{label}</div>
      <div style={{ fontSize: '22px', fontWeight: 700, color: color || '#e0e6ed', fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
      {sub && <div style={{ fontSize: '11px', color: '#4a5568', marginTop: '2px', fontFamily: "'JetBrains Mono', monospace" }}>{sub}</div>}
    </div>
  );
}

// ─── Main App ───
export default function PathfindingSimulator() {
  const [nodeCount, setNodeCount] = useState(60);
  const [edgeDensity, setEdgeDensity] = useState(2);
  const [graph, setGraph] = useState(null);
  const [mm, setMm] = useState(null);
  const [weights, setWeights] = useState({ ...WEIGHT_PRESETS.balanced });
  const [activePreset, setActivePreset] = useState('balanced');
  const [startNode, setStartNode] = useState(null);
  const [goalNode, setGoalNode] = useState(null);
  const [selectMode, setSelectMode] = useState(null); // 'start' | 'goal'
  const [algorithm, setAlgorithm] = useState('both');
  
  // Simulation state
  const [tradSteps, setTradSteps] = useState([]);
  const [dnnSteps, setDnnSteps] = useState([]);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(50);
  const [dnnModel, setDnnModel] = useState(null);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [showPanel, setShowPanel] = useState('controls');
  const timerRef = useRef(null);

  // Generate graph
  useEffect(() => {
    const g = generateGraph(nodeCount, edgeDensity);
    setGraph(g);
    setMm(computeMinMax(g.edges));
    setStartNode(0);
    setGoalNode(Math.min(nodeCount - 1, Object.keys(g.nodes).length - 1));
    setTradSteps([]);
    setDnnSteps([]);
    setStepIdx(0);
    setDnnModel(null);
    setTrained(false);
  }, [nodeCount, edgeDensity]);

  const handleNodeClick = useCallback((id) => {
    if (selectMode === 'start') {
      setStartNode(id);
      setSelectMode(null);
    } else if (selectMode === 'goal') {
      setGoalNode(id);
      setSelectMode(null);
    }
  }, [selectMode]);

  // Train DNN model
  const handleTrain = useCallback(() => {
    if (!graph || !mm) return;
    setTraining(true);
    setTimeout(() => {
      const model = trainSimpleModel(graph, weights, mm);
      setDnnModel(model);
      setTrained(true);
      setTraining(false);
    }, 100);
  }, [graph, mm, weights]);

  // Run simulation
  const handleRun = useCallback(() => {
    if (!graph || !mm || startNode === null || goalNode === null) return;
    
    const tSteps = astarSteps(graph, startNode, goalNode, weights, mm, (u, g, n) => euclideanH(u, g, n));
    setTradSteps(tSteps);
    
    if (dnnModel) {
      const dSteps = astarSteps(graph, startNode, goalNode, weights, mm, (u, g, n) => dnnH(u, g, n, dnnModel));
      setDnnSteps(dSteps);
    } else {
      setDnnSteps([]);
    }
    
    setStepIdx(0);
    setPlaying(false);
  }, [graph, mm, startNode, goalNode, weights, dnnModel]);

  // Playback
  useEffect(() => {
    if (playing) {
      const maxSteps = Math.max(tradSteps.length, dnnSteps.length);
      if (stepIdx >= maxSteps - 1) {
        setPlaying(false);
        return;
      }
      timerRef.current = setTimeout(() => {
        setStepIdx(s => s + 1);
      }, Math.max(10, 500 - speed * 4.5));
    }
    return () => clearTimeout(timerRef.current);
  }, [playing, stepIdx, tradSteps.length, dnnSteps.length, speed]);

  const tradStep = tradSteps[Math.min(stepIdx, tradSteps.length - 1)] || null;
  const dnnStep = dnnSteps[Math.min(stepIdx, dnnSteps.length - 1)] || null;
  const maxSteps = Math.max(tradSteps.length, dnnSteps.length);
  
  const tradResult = tradSteps.find(s => s.type === 'done');
  const dnnResult = dnnSteps.find(s => s.type === 'done');

  const presetColors = {
    traffic_focused: '#ff6b35',
    cost_focused: '#ffd54f',
    safety_focused: '#00e676',
    balanced: '#7c4dff',
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#060a10',
      color: '#c8d3de',
      fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif",
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* ─── Header ─── */}
      <header style={{
        padding: '16px 24px',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(6,10,16,0.95)',
        backdropFilter: 'blur(12px)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            background: 'linear-gradient(135deg, #00e5ff, #7c4dff)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '16px', fontWeight: 800, color: '#060a10',
          }}>A*</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '15px', color: '#e8edf2', letterSpacing: '-0.3px' }}>DNN-Based A* Pathfinding Simulator</div>
            <div style={{ fontSize: '11px', color: '#4a5e73', fontFamily: "'JetBrains Mono', monospace" }}>MCDM Multi-Criteria • Euclidean vs Learned Heuristic</div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {['controls', 'results'].map(tab => (
            <button
              key={tab}
              onClick={() => setShowPanel(tab)}
              style={{
                padding: '6px 14px', borderRadius: '6px', border: 'none',
                background: showPanel === tab ? 'rgba(0,229,255,0.12)' : 'transparent',
                color: showPanel === tab ? '#00e5ff' : '#5a6a7a',
                fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                textTransform: 'uppercase', letterSpacing: '0.5px',
              }}
            >{tab}</button>
          ))}
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* ─── Sidebar ─── */}
        <aside style={{
          width: '320px',
          minWidth: '320px',
          borderRight: '1px solid rgba(255,255,255,0.05)',
          overflowY: 'auto',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          background: 'rgba(8,12,20,0.6)',
        }}>
          {showPanel === 'controls' ? (
            <>
              {/* Graph Config */}
              <Section title="Graph Configuration">
                <SliderRow label="Nodes" value={nodeCount} min={10} max={150} step={5}
                  onChange={v => setNodeCount(v)} display={nodeCount} />
                <SliderRow label="Density" value={edgeDensity} min={1} max={5} step={0.5}
                  onChange={v => setEdgeDensity(v)} display={edgeDensity.toFixed(1)} />
                <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                  <ActionBtn
                    label={selectMode === 'start' ? '⦿ Click a node...' : `Start: ${startNode ?? '–'}`}
                    onClick={() => setSelectMode(selectMode === 'start' ? null : 'start')}
                    color="#00e676" active={selectMode === 'start'}
                  />
                  <ActionBtn
                    label={selectMode === 'goal' ? '⦿ Click a node...' : `Goal: ${goalNode ?? '–'}`}
                    onClick={() => setSelectMode(selectMode === 'goal' ? null : 'goal')}
                    color="#ff1744" active={selectMode === 'goal'}
                  />
                </div>
              </Section>

              {/* MCDM Weights */}
              <Section title="MCDM Weights">
                <div style={{ display: 'flex', gap: '6px', marginBottom: '12px', flexWrap: 'wrap' }}>
                  {Object.entries(WEIGHT_PRESETS).map(([name, w]) => (
                    <button
                      key={name}
                      onClick={() => { setWeights({ ...w }); setActivePreset(name); setTrained(false); }}
                      style={{
                        padding: '5px 10px', borderRadius: '5px', border: `1px solid ${presetColors[name]}33`,
                        background: activePreset === name ? `${presetColors[name]}18` : 'transparent',
                        color: activePreset === name ? presetColors[name] : '#5a6a7a',
                        fontSize: '10px', fontWeight: 700, cursor: 'pointer',
                        textTransform: 'uppercase', letterSpacing: '0.5px',
                      }}
                    >{name.replace('_', ' ')}</button>
                  ))}
                </div>
                <SliderRow label="Traffic" value={weights.traffic} min={0} max={1} step={0.05}
                  onChange={v => { setWeights(w => ({ ...w, traffic: v })); setActivePreset(null); setTrained(false); }}
                  display={weights.traffic.toFixed(2)} color="#ff6b35" />
                <SliderRow label="Safety" value={weights.safety} min={0} max={1} step={0.05}
                  onChange={v => { setWeights(w => ({ ...w, safety: v })); setActivePreset(null); setTrained(false); }}
                  display={weights.safety.toFixed(2)} color="#00e676" />
                <SliderRow label="Cost" value={weights.cost} min={0} max={1} step={0.05}
                  onChange={v => { setWeights(w => ({ ...w, cost: v })); setActivePreset(null); setTrained(false); }}
                  display={weights.cost.toFixed(2)} color="#ffd54f" />
              </Section>

              {/* Algorithm */}
              <Section title="Algorithm">
                <div style={{ display: 'flex', gap: '6px' }}>
                  {[['both', 'Compare Both'], ['trad', 'Euclidean Only'], ['dnn', 'DNN Only']].map(([val, label]) => (
                    <button
                      key={val}
                      onClick={() => setAlgorithm(val)}
                      style={{
                        flex: 1, padding: '7px 4px', borderRadius: '6px',
                        border: `1px solid ${algorithm === val ? '#00e5ff33' : 'rgba(255,255,255,0.06)'}`,
                        background: algorithm === val ? 'rgba(0,229,255,0.08)' : 'transparent',
                        color: algorithm === val ? '#00e5ff' : '#5a6a7a',
                        fontSize: '10px', fontWeight: 600, cursor: 'pointer',
                      }}
                    >{label}</button>
                  ))}
                </div>
              </Section>

              {/* Actions */}
              <Section title="Actions">
                <button
                  onClick={handleTrain}
                  disabled={training}
                  style={{
                    width: '100%', padding: '10px', borderRadius: '8px', border: 'none',
                    background: trained ? 'rgba(0,230,118,0.1)' : 'linear-gradient(135deg, #7c4dff, #536dfe)',
                    color: trained ? '#00e676' : '#fff',
                    fontWeight: 700, fontSize: '12px', cursor: training ? 'wait' : 'pointer',
                    letterSpacing: '0.5px', marginBottom: '8px',
                  }}
                >
                  {training ? '⟳ Training...' : trained ? '✓ Model Trained' : '🧠 Train DNN Model'}
                </button>
                {dnnModel && (
                  <div style={{ fontSize: '10px', color: '#4a5e73', textAlign: 'center', marginBottom: '8px', fontFamily: "'JetBrains Mono', monospace" }}>
                    Regression: cost ≈ {dnnModel.a.toFixed(4)} × dist + {dnnModel.b.toFixed(4)} ({dnnModel.sampleCount} samples)
                  </div>
                )}
                <button
                  onClick={handleRun}
                  style={{
                    width: '100%', padding: '12px', borderRadius: '8px', border: 'none',
                    background: 'linear-gradient(135deg, #00e5ff, #00b0ff)',
                    color: '#060a10', fontWeight: 800, fontSize: '13px', cursor: 'pointer',
                    letterSpacing: '0.3px',
                  }}
                >▶ Run Pathfinding</button>
              </Section>

              {/* Playback */}
              {maxSteps > 0 && (
                <Section title="Playback">
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <button onClick={() => setStepIdx(0)} style={playBtn}>⏮</button>
                    <button onClick={() => setStepIdx(Math.max(0, stepIdx - 1))} style={playBtn}>◀</button>
                    <button onClick={() => setPlaying(!playing)} style={{ ...playBtn, background: playing ? 'rgba(255,23,68,0.15)' : 'rgba(0,229,255,0.15)', color: playing ? '#ff1744' : '#00e5ff', flex: 1 }}>
                      {playing ? '⏸ Pause' : '▶ Play'}
                    </button>
                    <button onClick={() => setStepIdx(Math.min(maxSteps - 1, stepIdx + 1))} style={playBtn}>▶</button>
                    <button onClick={() => setStepIdx(maxSteps - 1)} style={playBtn}>⏭</button>
                  </div>
                  <div style={{ marginTop: '8px' }}>
                    <input type="range" min={0} max={maxSteps - 1} value={stepIdx}
                      onChange={e => { setStepIdx(Number(e.target.value)); setPlaying(false); }}
                      style={{ width: '100%', accentColor: '#00e5ff' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#4a5e73', fontFamily: "'JetBrains Mono', monospace" }}>
                      <span>Step {stepIdx + 1}/{maxSteps}</span>
                    </div>
                  </div>
                  <SliderRow label="Speed" value={speed} min={1} max={100} step={1}
                    onChange={v => setSpeed(v)} display={speed} color="#00e5ff" />
                </Section>
              )}
            </>
          ) : (
            /* Results Panel */
            <>
              <Section title="Traditional A* (Euclidean)">
                {tradResult ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <MiniStat label="Path Cost" value={tradResult.cost.toFixed(4)} />
                    <MiniStat label="Path Length" value={`${tradResult.path.length} nodes`} />
                    <MiniStat label="Nodes Expanded" value={tradResult.visited.size} />
                    <MiniStat label="Steps" value={tradSteps.length} />
                  </div>
                ) : <div style={{ color: '#3a4a5a', fontSize: '12px' }}>Run simulation first</div>}
              </Section>
              <Section title="DNN A* (Learned Heuristic)">
                {dnnResult ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <MiniStat label="Path Cost" value={dnnResult.cost.toFixed(4)} />
                    <MiniStat label="Path Length" value={`${dnnResult.path.length} nodes`} />
                    <MiniStat label="Nodes Expanded" value={dnnResult.visited.size} />
                    <MiniStat label="Steps" value={dnnSteps.length} />
                  </div>
                ) : <div style={{ color: '#3a4a5a', fontSize: '12px' }}>{trained ? 'Run simulation' : 'Train model first'}</div>}
              </Section>
              {tradResult && dnnResult && (
                <Section title="Comparison">
                  <ComparisonBar
                    label="Cost"
                    tradVal={tradResult.cost}
                    dnnVal={dnnResult.cost}
                    lowerBetter
                  />
                  <ComparisonBar
                    label="Nodes Expanded"
                    tradVal={tradResult.visited.size}
                    dnnVal={dnnResult.visited.size}
                    lowerBetter
                  />
                  <ComparisonBar
                    label="Path Length"
                    tradVal={tradResult.path.length}
                    dnnVal={dnnResult.path.length}
                    lowerBetter
                  />
                </Section>
              )}
            </>
          )}
        </aside>

        {/* ─── Main Canvas Area ─── */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Stats bar */}
          <div style={{
            padding: '10px 16px',
            display: 'flex',
            gap: '10px',
            borderBottom: '1px solid rgba(255,255,255,0.04)',
            overflowX: 'auto',
          }}>
            <StatCard label="Nodes" value={graph ? Object.keys(graph.nodes).length : 0} color="#7c4dff" />
            <StatCard label="Edges" value={graph ? graph.edges.length : 0} color="#536dfe" />
            <StatCard label="Trad Visited" value={tradStep?.visited?.size ?? '–'} color="#5c6bc0" />
            <StatCard label="DNN Visited" value={dnnStep?.visited?.size ?? '–'} color="#00e5ff" />
            <StatCard
              label="Efficiency Δ"
              value={tradResult && dnnResult
                ? `${((1 - dnnResult.visited.size / tradResult.visited.size) * 100).toFixed(1)}%`
                : '–'}
              color={tradResult && dnnResult && dnnResult.visited.size < tradResult.visited.size ? '#00e676' : '#ff6b35'}
            />
          </div>

          {/* Canvas */}
          <div style={{ flex: 1, display: 'flex', gap: '2px', padding: '8px', overflow: 'hidden' }}>
            {(algorithm === 'both' || algorithm === 'trad') && graph && (
              <div style={{ flex: 1, position: 'relative', display: 'flex', flexDirection: 'column' }}>
                <div style={{
                  padding: '6px 12px', fontSize: '11px', fontWeight: 700,
                  color: '#5c6bc0', textTransform: 'uppercase', letterSpacing: '1px',
                  fontFamily: "'JetBrains Mono', monospace",
                }}>Traditional A* (Euclidean)</div>
                <div style={{ flex: 1 }}>
                  <GraphCanvas
                    graph={graph}
                    width={algorithm === 'both' ? 500 : 1000}
                    height={520}
                    step={tradStep}
                    path={tradResult?.path}
                    startNode={startNode}
                    goalNode={goalNode}
                    onNodeClick={handleNodeClick}
                    selectMode={selectMode}
                  />
                </div>
              </div>
            )}
            {(algorithm === 'both' || algorithm === 'dnn') && graph && (
              <div style={{ flex: 1, position: 'relative', display: 'flex', flexDirection: 'column' }}>
                <div style={{
                  padding: '6px 12px', fontSize: '11px', fontWeight: 700,
                  color: '#00e5ff', textTransform: 'uppercase', letterSpacing: '1px',
                  fontFamily: "'JetBrains Mono', monospace",
                }}>DNN A* (Learned Heuristic)</div>
                <div style={{ flex: 1 }}>
                  <GraphCanvas
                    graph={graph}
                    width={algorithm === 'both' ? 500 : 1000}
                    height={520}
                    step={dnnStep}
                    path={dnnResult?.path}
                    startNode={startNode}
                    goalNode={goalNode}
                    onNodeClick={handleNodeClick}
                    selectMode={selectMode}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Legend */}
          <div style={{
            padding: '8px 16px',
            display: 'flex',
            gap: '20px',
            borderTop: '1px solid rgba(255,255,255,0.04)',
            fontSize: '11px',
            color: '#4a5e73',
          }}>
            {[
              ['#00e676', 'Start Node'],
              ['#ff1744', 'Goal Node'],
              ['#5c6bc0', 'Visited'],
              ['#ffd54f', 'Frontier'],
              ['#00e5ff', 'Path'],
              ['#1a2332', 'Unvisited'],
            ].map(([color, label]) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{ width: 10, height: 10, borderRadius: '50%', background: color }} />
                <span>{label}</span>
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}

// ─── Sub-components ───
const playBtn = {
  padding: '6px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.08)',
  background: 'rgba(255,255,255,0.03)', color: '#8a9ab0', fontSize: '12px', cursor: 'pointer',
};

function Section({ title, children }) {
  return (
    <div>
      <div style={{
        fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1.5px',
        color: '#3a4e62', fontWeight: 700, marginBottom: '10px',
        fontFamily: "'JetBrains Mono', monospace",
      }}>{title}</div>
      {children}
    </div>
  );
}

function SliderRow({ label, value, min, max, step, onChange, display, color }) {
  return (
    <div style={{ marginBottom: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
        <span style={{ color: '#6b7b8d' }}>{label}</span>
        <span style={{ color: color || '#8a9ab0', fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{display}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: color || '#00e5ff', height: '4px' }}
      />
    </div>
  );
}

function ActionBtn({ label, onClick, color, active }) {
  return (
    <button onClick={onClick} style={{
      flex: 1, padding: '7px 8px', borderRadius: '6px',
      border: `1px solid ${active ? color : 'rgba(255,255,255,0.08)'}`,
      background: active ? `${color}15` : 'rgba(255,255,255,0.02)',
      color: active ? color : '#6b7b8d',
      fontSize: '10px', fontWeight: 600, cursor: 'pointer',
      fontFamily: "'JetBrains Mono', monospace",
    }}>{label}</button>
  );
}

function MiniStat({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
      <span style={{ color: '#4a5e73' }}>{label}</span>
      <span style={{ color: '#c8d3de', fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function ComparisonBar({ label, tradVal, dnnVal, lowerBetter }) {
  const tradBetter = lowerBetter ? tradVal <= dnnVal : tradVal >= dnnVal;
  const diff = lowerBetter
    ? ((1 - dnnVal / tradVal) * 100)
    : ((dnnVal / tradVal - 1) * 100);
  
  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
        <span style={{ color: '#6b7b8d' }}>{label}</span>
        <span style={{
          color: diff > 0 ? '#00e676' : diff < 0 ? '#ff6b35' : '#6b7b8d',
          fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: '10px',
        }}>
          {diff > 0 ? '+' : ''}{diff.toFixed(1)}% {diff > 0 ? '↑' : diff < 0 ? '↓' : ''}
        </span>
      </div>
      <div style={{ display: 'flex', gap: '4px', height: '20px' }}>
        <div style={{
          flex: 1, background: '#5c6bc022', borderRadius: '4px', position: 'relative', overflow: 'hidden',
        }}>
          <div style={{
            position: 'absolute', left: 0, top: 0, bottom: 0,
            width: `${Math.min(100, (tradVal / Math.max(tradVal, dnnVal)) * 100)}%`,
            background: tradBetter ? '#5c6bc0' : '#5c6bc066',
            borderRadius: '4px',
            display: 'flex', alignItems: 'center', paddingLeft: '6px',
            fontSize: '10px', fontWeight: 600, color: '#fff',
            fontFamily: "'JetBrains Mono', monospace",
          }}>{typeof tradVal === 'number' && tradVal % 1 !== 0 ? tradVal.toFixed(3) : tradVal}</div>
        </div>
        <div style={{
          flex: 1, background: '#00e5ff11', borderRadius: '4px', position: 'relative', overflow: 'hidden',
        }}>
          <div style={{
            position: 'absolute', left: 0, top: 0, bottom: 0,
            width: `${Math.min(100, (dnnVal / Math.max(tradVal, dnnVal)) * 100)}%`,
            background: !tradBetter ? '#00e5ff' : '#00e5ff66',
            borderRadius: '4px',
            display: 'flex', alignItems: 'center', paddingLeft: '6px',
            fontSize: '10px', fontWeight: 600, color: '#060a10',
            fontFamily: "'JetBrains Mono', monospace",
          }}>{typeof dnnVal === 'number' && dnnVal % 1 !== 0 ? dnnVal.toFixed(3) : dnnVal}</div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#3a4e62', marginTop: '2px' }}>
        <span>Euclidean</span><span>DNN</span>
      </div>
    </div>
  );
}
