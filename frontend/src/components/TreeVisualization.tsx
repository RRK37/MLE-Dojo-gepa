'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  GitBranch, 
  Database, 
  Cpu, 
  Zap, 
  Activity, 
  Play, 
  Square, 
  RotateCcw, 
  Plus,
  Terminal,
  Layers,
  Search,
  Settings
} from 'lucide-react';
import './TreeVisualization.css';

export interface TreeNode {
  id: string;
  x: number;
  y: number;
  level: number;
  parent: string | null;
  isActive: boolean;
  isNew: boolean;
  code: string;
  modelType?: string;
  metrics?: {
    accuracy: number;
    loss: number;
  };
  children?: string[]; // Helper for layout
}

export interface Connection {
  from: string;
  to: string;
}

// Sample Python code templates for different model types
const CODE_TEMPLATES = {
  catboost: `cat_model = CatBoostClassifier(
    verbose=0,
    task_type="CPU",
    iterations=500,
    depth=6,
    learning_rate=0.1
)`,
  lgbm: `lgb_model = LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1
)`,
  xgb: `xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss'
)`,
  logistic: `LogisticRegression(
    max_iter=1000,
    random_state=42
)`,
  ensemble: `# Ensemble predictions
stacked_pred = np.column_stack([
    cat_pred_val,
    lgb_pred_val,
    xgb_pred_val
])`,
};

interface TreeVisualizationProps {
  nodes?: TreeNode[];
  connections?: Connection[];
  onStartSimulation?: () => void;
  onStopSimulation?: () => void;
  onReset?: () => void;
  onAddNode?: () => void;
  isSimulatingExternal?: boolean;
}

const TreeVisualization: React.FC<TreeVisualizationProps> = ({
  nodes: externalNodes,
  connections: externalConnections,
  onStartSimulation,
  onStopSimulation,
  onReset,
  onAddNode,
  isSimulatingExternal
}) => {
  const [internalNodes, setInternalNodes] = useState<TreeNode[]>([]);
  const [internalConnections, setInternalConnections] = useState<Connection[]>([]);
  const [isSimulatingInternal, setIsSimulatingInternal] = useState(false);
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);
  const [initialNodeCount, setInitialNodeCount] = useState(3);
  
  // Use refs to access latest state in intervals without triggering re-renders
  const nodesRef = useRef<TreeNode[]>([]);
  
  const nodes = externalNodes || internalNodes;
  const connections = externalConnections || internalConnections;
  const isSimulating = isSimulatingExternal !== undefined ? isSimulatingExternal : isSimulatingInternal;

  // Update ref when nodes change
  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  // Layout Algorithm: Reingold-Tilford inspired (Leaf-based)
  const recalculateLayout = (currentNodes: TreeNode[], currentConnections: Connection[]) => {
    if (currentNodes.length === 0) return currentNodes;

    // 1. Build Hierarchy Map
    const nodeMap = new Map<string, TreeNode>();
    const childrenMap = new Map<string, string[]>();
    
    currentNodes.forEach(n => {
      nodeMap.set(n.id, { ...n, children: [] });
      childrenMap.set(n.id, []);
    });

    currentConnections.forEach(c => {
      const parent = childrenMap.get(c.from);
      if (parent) parent.push(c.to);
    });

    // 2. Identify Leaves (nodes with no children)
    // We need to traverse to find "visual leaves" - i.e., the bottom-most nodes of each branch
    // Actually, a simpler approach for this specific tree structure:
    // Assign X based on "leaf slots".
    
    const leaves: string[] = [];
    
    // Helper to traverse and collect leaves in order
    const collectLeaves = (nodeId: string) => {
      const children = childrenMap.get(nodeId) || [];
      if (children.length === 0) {
        leaves.push(nodeId);
      } else {
        children.forEach(childId => collectLeaves(childId));
      }
    };

    // Find root(s)
    const roots = currentNodes.filter(n => !n.parent);
    roots.forEach(root => collectLeaves(root.id));

    // 3. Assign X to leaves
    const LEAF_SPACING = 80;
    const leafXMap = new Map<string, number>();
    const totalWidth = leaves.length * LEAF_SPACING;
    const startX = 400 - totalWidth / 2; // Center around 400

    leaves.forEach((leafId, index) => {
      leafXMap.set(leafId, startX + index * LEAF_SPACING);
    });

    // 4. Propagate X up the tree (Post-order traversal)
    const newNodes = currentNodes.map(n => ({ ...n }));
    const newNodeMap = new Map(newNodes.map(n => [n.id, n]));

    const calculateX = (nodeId: string): number => {
      const children = childrenMap.get(nodeId) || [];
      
      if (children.length === 0) {
        return leafXMap.get(nodeId) || 400;
      }

      const childrenX = children.map(childId => calculateX(childId));
      const avgX = childrenX.reduce((a, b) => a + b, 0) / childrenX.length;
      
      const node = newNodeMap.get(nodeId);
      if (node) {
        node.x = avgX;
        // Keep Y based on level, but ensure spacing
        node.y = 80 + node.level * 120;
      }
      
      return avgX;
    };

    roots.forEach(root => calculateX(root.id));

    return newNodes;
  };

  // Initialize tree with n top-level nodes
  const initializeTree = useCallback((count: number) => {
    if (externalNodes) return;

    const root: TreeNode = {
      id: 'root',
      x: 400,
      y: 80,
      level: 0,
      parent: null,
      isActive: true,
      isNew: false,
      code: CODE_TEMPLATES.ensemble,
      modelType: 'ensemble',
      metrics: { accuracy: 0.85, loss: 0.32 }
    };

    const topLevelNodes: TreeNode[] = [];
    const modelTypes = ['catboost', 'lgbm', 'xgb'];

    for (let i = 0; i < count; i++) {
      const modelType = modelTypes[i % modelTypes.length];
      topLevelNodes.push({
        id: `top-${i}`,
        x: 0, // Will be calculated
        y: 200,
        level: 1,
        parent: 'root',
        isActive: true,
        isNew: false,
        code: CODE_TEMPLATES[modelType as keyof typeof CODE_TEMPLATES],
        modelType,
        metrics: { 
          accuracy: 0.75 + Math.random() * 0.1, 
          loss: 0.4 + Math.random() * 0.2 
        }
      });
    }

    const newConnections: Connection[] = topLevelNodes.map((node) => ({
      from: 'root',
      to: node.id,
    }));

    const allNodes = [root, ...topLevelNodes];
    const layoutNodes = recalculateLayout(allNodes, newConnections);

    setInternalNodes(layoutNodes);
    setInternalConnections(newConnections);
    setSelectedNode(root);
  }, [externalNodes]);

  // Initialize on mount and start simulation
  useEffect(() => {
    initializeTree(initialNodeCount);
    // Auto-start simulation
    if (isSimulatingExternal === undefined) {
      setIsSimulatingInternal(true);
    }
  }, [initializeTree, initialNodeCount, isSimulatingExternal]);

  // Simulate backend sending new nodes
  const generateNewNode = useCallback(() => {
    if (onAddNode) {
      onAddNode();
      return;
    }

    const currentNodes = nodesRef.current;
    
    // Find nodes that can have children (active nodes in the tree)
    const possibleParents = currentNodes.filter(
      (node) => node.level < 4 // Limit depth slightly more
    );

    if (possibleParents.length === 0) return;

    // Prefer parents with fewer children to balance tree initially
    // But random is fine for simulation
    const parent = possibleParents[Math.floor(Math.random() * possibleParents.length)];

    const modelTypes = ['catboost', 'lgbm', 'xgb', 'logistic'];
    const modelType = modelTypes[Math.floor(Math.random() * modelTypes.length)];

    const newNode: TreeNode = {
      id: `node-${Date.now()}-${Math.random()}`,
      x: parent.x, // Start at parent X for animation
      y: parent.y + 120,
      level: parent.level + 1,
      parent: parent.id,
      isActive: Math.random() > 0.3,
      isNew: true,
      code: CODE_TEMPLATES[modelType as keyof typeof CODE_TEMPLATES],
      modelType,
      metrics: { 
        accuracy: 0.7 + Math.random() * 0.15, 
        loss: 0.3 + Math.random() * 0.3 
      }
    };

    const newConn = { from: parent.id, to: newNode.id };
    
    // Update state with new node AND recalculate layout
    setInternalNodes(prev => {
      const updatedNodes = [...prev, newNode];
      const updatedConnections = [...internalConnections, newConn]; // Need connections for layout
      return recalculateLayout(updatedNodes, updatedConnections);
    });
    
    setInternalConnections(prev => [...prev, newConn]);

    // Remove "new" status after animation
    setTimeout(() => {
      setInternalNodes((nodes) =>
        nodes.map((n) => (n.id === newNode.id ? { ...n, isNew: false } : n))
      );
    }, 800);

  }, [onAddNode, internalConnections]); // Added internalConnections dependency

  // Auto-generate nodes when simulation is running
  useEffect(() => {
    if (!isSimulating) return;

    const interval = setInterval(() => {
      generateNewNode();
    }, 2000); // Slower interval (2s)

    return () => clearInterval(interval);
  }, [isSimulating, generateNewNode]);

  // Helper to get model icon
  const getModelIcon = (type?: string) => {
    switch (type) {
      case 'ensemble': return <Layers size={18} />;
      case 'catboost': return <Cpu size={18} />;
      case 'lgbm': return <Zap size={18} />;
      case 'xgb': return <Activity size={18} />;
      default: return <Database size={18} />;
    }
  };

  // Helper to get model color
  const getModelColor = (type?: string) => {
    switch (type) {
      case 'ensemble': return '#ec4899'; // Pink
      case 'catboost': return '#3b82f6'; // Blue
      case 'lgbm': return '#eab308'; // Yellow
      case 'xgb': return '#22c55e'; // Green
      default: return '#a855f7'; // Purple
    }
  };

  return (
    <div className="tree-container">
      {/* Header */}
      <div className="tree-header">
        <div className="header-left">
          <div className="icon-wrapper">
            <GitBranch size={20} />
          </div>
          <div className="title-group">
            <h1>Model Evolution Tree</h1>
            <p>playground-series-s3e7 • Reservation Cancellation Prediction</p>
          </div>
        </div>
        
        <div className="header-right">
          <div className="status-badge">
            <span className="status-dot"></span>
            <span className="status-text">System Active</span>
          </div>
          <button className="icon-button">
            <Settings size={18} />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Right Panel - Code & Details (Now on Left) */}
        <div className="inspector-panel">
          <div className="inspector-header">
            <h2 className="inspector-title">
              <Terminal size={16} color="#60a5fa" />
              Model Inspector
            </h2>
          </div>

          <div className="inspector-content">
            {selectedNode ? (
              <>
                {/* Node Details Card */}
                <div className="node-card">
                  <div className="node-header">
                    <div className="node-identity">
                      <div 
                        className="node-icon"
                        style={{ 
                          color: getModelColor(selectedNode.modelType),
                          borderColor: `${getModelColor(selectedNode.modelType)}40`,
                          backgroundColor: `${getModelColor(selectedNode.modelType)}10`
                        }}
                      >
                        {getModelIcon(selectedNode.modelType)}
                      </div>
                      <div className="node-info">
                        <h3>{selectedNode.modelType || 'Unknown'}</h3>
                        <p>{selectedNode.id.split('-')[0]}...{selectedNode.id.slice(-4)}</p>
                      </div>
                    </div>
                    <div className={`status-tag ${selectedNode.isActive ? 'active' : 'inactive'}`}>
                      {selectedNode.isActive ? 'Active' : 'Inactive'}
                    </div>
                  </div>

                  <div className="metrics-grid">
                    <div className="metric-box">
                      <div className="metric-label">Accuracy</div>
                      <div className="metric-value accuracy">
                        {(selectedNode.metrics?.accuracy || 0).toFixed(4)}
                      </div>
                    </div>
                    <div className="metric-box">
                      <div className="metric-label">Loss</div>
                      <div className="metric-value loss">
                        {(selectedNode.metrics?.loss || 0).toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Code Block */}
                <div className="code-block">
                  <div className="code-header">
                    <span className="filename">config.py</span>
                    <button 
                      onClick={() => navigator.clipboard.writeText(selectedNode.code)}
                      className="copy-btn"
                    >
                      Copy
                    </button>
                  </div>
                  <pre className="code-content">
                    <code>{selectedNode.code}</code>
                  </pre>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <Search size={48} strokeWidth={1} />
                <p>Select a node to view details</p>
              </div>
            )}
          </div>
        </div>

        {/* Tree Visualization Area */}
        <div className="viz-area">
          <svg className="viz-svg">
            <defs>
              <linearGradient id="lineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.5" />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.2" />
              </linearGradient>
              
              <filter id="glow-node" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>

              <radialGradient id="nodeGradient" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                <stop offset="0%" stopColor="#1f2937" stopOpacity="1" />
                <stop offset="100%" stopColor="#111827" stopOpacity="1" />
              </radialGradient>
            </defs>

            {/* Connections */}
            {connections.map((conn, idx) => {
              const fromNode = nodes.find((n) => n.id === conn.from);
              const toNode = nodes.find((n) => n.id === conn.to);

              if (!fromNode || !toNode) return null;

              // Improved Bezier curve
              const deltaY = toNode.y - fromNode.y;
              // Use a fixed control point offset for more consistent curves
              const cpOffset = Math.max(deltaY * 0.5, 40);
              
              const pathData = `M ${fromNode.x} ${fromNode.y} 
                               C ${fromNode.x} ${fromNode.y + cpOffset}, 
                                 ${toNode.x} ${toNode.y - cpOffset}, 
                                 ${toNode.x} ${toNode.y}`;

              return (
                <path
                  key={`${conn.from}-${conn.to}-${idx}`}
                  d={pathData}
                  fill="none"
                  stroke={toNode.isActive ? "url(#lineGradient)" : "#30363d"}
                  strokeWidth={toNode.isActive ? 2 : 1}
                  strokeLinecap="round"
                  className="transition-all"
                  style={{
                    opacity: toNode.isNew ? 0 : 0.6,
                    animation: toNode.isNew ? 'fadeInLine 1s ease-out forwards' : 'none',
                  }}
                />
              );
            })}

            {/* Nodes */}
            {nodes.map((node) => {
              const color = getModelColor(node.modelType);
              const isSelected = selectedNode?.id === node.id;
              
              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x}, ${node.y})`}
                  style={{
                    opacity: node.isNew ? 0 : 1,
                    animation: node.isNew
                      ? 'popIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards'
                      : 'none',
                    cursor: 'pointer',
                    transition: 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)' // Smooth movement
                  }}
                  onClick={() => setSelectedNode(node)}
                >
                  {/* Selection Ring */}
                  {isSelected && (
                    <circle
                      r="32"
                      fill="none"
                      stroke={color}
                      strokeWidth="2"
                      strokeOpacity="0.4"
                      className="animate-pulse"
                    />
                  )}

                  {/* Glow Effect for Active Nodes */}
                  {node.isActive && (
                    <circle
                      r="24"
                      fill={color}
                      fillOpacity="0.1"
                      filter="url(#glow-node)"
                    />
                  )}

                  {/* Node Background */}
                  <circle
                    r="24"
                    fill="url(#nodeGradient)"
                    stroke={node.isActive ? color : '#374151'}
                    strokeWidth={isSelected ? 3 : 2}
                    className="transition-all"
                  />

                  {/* Icon */}
                  <foreignObject x="-12" y="-12" width="24" height="24">
                    <div className="flex items-center justify-center w-full h-full" style={{ color: node.isActive ? color : '#6b7280' }}>
                      {getModelIcon(node.modelType)}
                    </div>
                  </foreignObject>

                  {/* Label (only for root or top level) */}
                  {node.level <= 1 && (
                    <text
                      y="42"
                      textAnchor="middle"
                      fill="#9ca3af"
                      fontSize="11"
                      fontWeight="500"
                      className="font-sans tracking-wide"
                    >
                      {node.modelType}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
};

export default TreeVisualization;
