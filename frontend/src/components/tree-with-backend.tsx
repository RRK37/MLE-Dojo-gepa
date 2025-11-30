'use client';

import { useEffect, useState } from 'react';
import TreeVisualization from '@/components/TreeVisualization';
import { ModelTreeWebSocket } from './backend-integration';

/**
 * Example page showing how to use TreeVisualization with backend integration
 * This demonstrates both WebSocket and REST API approaches
 */

// ============================================================================
// Option 1: Using WebSocket for Real-Time Updates
// ============================================================================

export function TreePageWithWebSocket() {
  const [wsClient, setWsClient] = useState<ModelTreeWebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Initialize WebSocket connection
    const client = new ModelTreeWebSocket('ws://localhost:8000/ws/model-tree');

    // Set up event listeners
    client.onInit((data) => {
      console.log('Tree initialized:', data);
      setIsConnected(true);
    });

    client.onNode((node) => {
      console.log('New node generated:', node);
    });

    setWsClient(client);

    // Cleanup on unmount
    return () => {
      client.disconnect();
      setIsConnected(false);
    };
  }, []);

  return (
    <div className="w-full h-screen">
      {/* Connection status indicator */}
      <div className="absolute top-4 right-4 z-10 bg-white rounded-lg shadow-lg px-4 py-2">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            }`}
          />
          <span className="text-sm">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <TreeVisualization />
    </div>
  );
}

// ============================================================================
// Option 2: Using REST API with Polling
// ============================================================================

export function TreePageWithRestAPI() {
  const [isPolling, setIsPolling] = useState(false);

  useEffect(() => {
    if (!isPolling) return;

    // Poll for new nodes every 2 seconds
    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/model/get-updates');
        const data = await response.json();
        
        if (data.newNodes && data.newNodes.length > 0) {
          console.log('New nodes received:', data.newNodes);
          // Update tree with new nodes
        }
      } catch (error) {
        console.error('Error polling for updates:', error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isPolling]);

  return <TreeVisualization />;
}

// ============================================================================
// Option 3: Server-Side Props (Next.js)
// ============================================================================

interface TreePageProps {
  initialNodeCount: number;
  initialNodes: any[];
}

export default function TreePageWithSSR({ initialNodeCount, initialNodes }: TreePageProps) {
  return <TreeVisualization />;
}

// Server-side data fetching
export async function getServerSideProps() {
  // Fetch initial tree configuration from your backend
  const response = await fetch('http://your-backend/api/model/initialize');
  const data = await response.json();

  return {
    props: {
      initialNodeCount: data.baseModels.length,
      initialNodes: data.nodes,
    },
  };
}

// ============================================================================
// Option 4: Hybrid Approach (Initial fetch + WebSocket updates)
// ============================================================================

export function TreePageHybrid() {
  const [initialData, setInitialData] = useState<any>(null);
  const [wsClient, setWsClient] = useState<ModelTreeWebSocket | null>(null);

  useEffect(() => {
    // First, fetch initial configuration
    fetch('/api/model/initialize')
      .then((res) => res.json())
      .then((data) => {
        setInitialData(data);

        // Then connect WebSocket for real-time updates
        const client = new ModelTreeWebSocket('ws://localhost:8000/ws/model-tree');
        
        client.onNode((node) => {
          console.log('New node:', node);
        });

        setWsClient(client);
      })
      .catch((error) => {
        console.error('Error initializing:', error);
      });

    return () => {
      wsClient?.disconnect();
    };
  }, []);

  if (!initialData) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  return <TreeVisualization />;
}

// ============================================================================
// Usage Examples
// ============================================================================

/*
// In your app/page.tsx or pages/index.tsx:

import { TreePageWithWebSocket } from './tree-with-backend';

export default function Page() {
  return <TreePageWithWebSocket />;
}

// Or for REST API:
import { TreePageWithRestAPI } from './tree-with-backend';

export default function Page() {
  return <TreePageWithRestAPI />;
}

// Or for SSR:
import TreePageWithSSR from './tree-with-backend';

export default TreePageWithSSR;

// Or for Hybrid:
import { TreePageHybrid } from './tree-with-backend';

export default function Page() {
  return <TreePageHybrid />;
}
*/
