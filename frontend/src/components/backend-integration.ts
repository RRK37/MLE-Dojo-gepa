// Example Backend Integration for TreeVisualization
// This file shows how to integrate the component with a real backend

import { TreeNode } from './TreeVisualization';

// ============================================================================
// REST API Integration Example
// ============================================================================

export async function initializeTreeFromAPI(): Promise<{
  initialNodeCount: number;
  nodes: TreeNode[];
}> {
  const response = await fetch('/api/model/initialize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      dataset: 'reservation-cancellation',
      modelTypes: ['catboost', 'lgbm', 'xgb'],
    }),
  });

  const data = await response.json();
  return {
    initialNodeCount: data.baseModels.length,
    nodes: data.nodes,
  };
}

export async function generateNodeFromAPI(parentId: string): Promise<TreeNode> {
  const response = await fetch('/api/model/generate-node', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      parentId,
    }),
  });

  const data = await response.json();
  return data.node;
}

// ============================================================================
// WebSocket Integration Example
// ============================================================================

export class ModelTreeWebSocket {
  private ws: WebSocket | null = null;
  private onNodeGenerated?: (node: TreeNode) => void;
  private onTreeInitialized?: (data: { nodes: TreeNode[]; initialCount: number }) => void;

  constructor(url: string) {
    this.connect(url);
  }

  private connect(url: string) {
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      // Request initial tree structure
      this.send({ type: 'INITIALIZE_TREE' });
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
  }

  private handleMessage(message: any) {
    switch (message.type) {
      case 'TREE_INITIALIZED':
        if (this.onTreeInitialized) {
          this.onTreeInitialized({
            nodes: message.nodes,
            initialCount: message.initialCount,
          });
        }
        break;

      case 'NODE_GENERATED':
        if (this.onNodeGenerated) {
          this.onNodeGenerated(message.node);
        }
        break;

      case 'TRAINING_COMPLETE':
        console.log('Training complete:', message.metrics);
        break;

      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  public onNode(callback: (node: TreeNode) => void) {
    this.onNodeGenerated = callback;
  }

  public onInit(callback: (data: { nodes: TreeNode[]; initialCount: number }) => void) {
    this.onTreeInitialized = callback;
  }

  public send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  public startTraining() {
    this.send({ type: 'START_TRAINING' });
  }

  public stopTraining() {
    this.send({ type: 'STOP_TRAINING' });
  }

  public disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// ============================================================================
// Usage Example in React Component
// ============================================================================

/*
import { ModelTreeWebSocket } from './backend-integration';

function TreePage() {
  const [nodes, setNodes] = useState<TreeNode[]>([]);
  const [wsClient, setWsClient] = useState<ModelTreeWebSocket | null>(null);

  useEffect(() => {
    // Initialize WebSocket connection
    const client = new ModelTreeWebSocket('ws://localhost:8000/ws/model-tree');

    // Handle tree initialization
    client.onInit((data) => {
      setNodes(data.nodes);
      setInitialNodeCount(data.initialCount);
    });

    // Handle new nodes
    client.onNode((node) => {
      setNodes((prev) => [...prev, node]);
      setConnections((prev) => [
        ...prev,
        { from: node.parent!, to: node.id },
      ]);
    });

    setWsClient(client);

    return () => {
      client.disconnect();
    };
  }, []);

  const handleStartSimulation = () => {
    wsClient?.startTraining();
  };

  const handleStopSimulation = () => {
    wsClient?.stopTraining();
  };

  return <TreeVisualization />;
}
*/

// ============================================================================
// Backend API Response Formats
// ============================================================================

// Initialize Tree Response
interface InitializeTreeResponse {
  baseModels: string[];
  nodes: Array<{
    id: string;
    x: number;
    y: number;
    level: number;
    parent: string | null;
    isActive: boolean;
    code: string;
    modelType: string;
  }>;
}

// Generate Node Response
interface GenerateNodeResponse {
  node: {
    id: string;
    x: number;
    y: number;
    level: number;
    parent: string;
    isActive: boolean;
    code: string;
    modelType: string;
  };
}

// WebSocket Message Types
type WebSocketMessage =
  | {
      type: 'TREE_INITIALIZED';
      nodes: TreeNode[];
      initialCount: number;
    }
  | {
      type: 'NODE_GENERATED';
      node: TreeNode;
    }
  | {
      type: 'TRAINING_COMPLETE';
      metrics: {
        accuracy: number;
        precision: number;
        recall: number;
      };
    };

// ============================================================================
// Example Python Backend (FastAPI)
// ============================================================================

/*
from fastapi import FastAPI, WebSocket
from typing import List
import asyncio

app = FastAPI()

class ModelTree:
    def __init__(self):
        self.nodes = []
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)
    
    async def initialize_tree(self, base_models: List[str]):
        # Initialize tree with base models
        nodes = [
            {
                "id": "root",
                "x": 400,
                "y": 50,
                "level": 0,
                "parent": None,
                "isActive": True,
                "code": "# Ensemble meta-model",
                "modelType": "ensemble"
            }
        ]
        
        for i, model_type in enumerate(base_models):
            nodes.append({
                "id": f"base-{i}",
                "x": 300 + i * 100,
                "y": 180,
                "level": 1,
                "parent": "root",
                "isActive": True,
                "code": f"{model_type}_model = ...",
                "modelType": model_type
            })
        
        await self.broadcast({
            "type": "TREE_INITIALIZED",
            "nodes": nodes,
            "initialCount": len(base_models)
        })
    
    async def generate_node(self, parent_id: str):
        # Generate new node logic here
        new_node = {
            "id": f"node-{len(self.nodes)}",
            "x": 400,
            "y": 300,
            "level": 2,
            "parent": parent_id,
            "isActive": True,
            "code": "# Generated model",
            "modelType": "generated"
        }
        
        await self.broadcast({
            "type": "NODE_GENERATED",
            "node": new_node
        })

tree_manager = ModelTree()

@app.websocket("/ws/model-tree")
async def websocket_endpoint(websocket: WebSocket):
    await tree_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "INITIALIZE_TREE":
                await tree_manager.initialize_tree(["catboost", "lgbm", "xgb"])
            
            elif data["type"] == "START_TRAINING":
                # Start generating nodes periodically
                for _ in range(10):
                    await asyncio.sleep(0.8)
                    await tree_manager.generate_node("root")
            
            elif data["type"] == "STOP_TRAINING":
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await tree_manager.disconnect(websocket)
*/
