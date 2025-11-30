"""
WebSocket client for connecting to the MLE-Dojo API server
and receiving real-time data from prepare/mle.py and main.py.

Usage:
    python client.py --host localhost --port 8765
"""

import asyncio
import json
import argparse
import sys
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI
from typing import Optional
from datetime import datetime


class APIClient:
    """Client for connecting to the MLE-Dojo API server."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.running = False
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            print(f"✓ Connected to {self.uri}")
            return True
        except ConnectionRefusedError:
            print(f"✗ Connection refused. Is the server running on {self.uri}?")
            return False
        except InvalidURI:
            print(f"✗ Invalid URI: {self.uri}")
            return False
        except Exception as e:
            print(f"✗ Error connecting: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            print("✓ Disconnected from server")
    
    async def send_command(self, command: dict):
        """Send a command to the server."""
        if not self.websocket:
            print("✗ Not connected to server")
            return False
        
        try:
            await self.websocket.send(json.dumps(command))
            return True
        except Exception as e:
            print(f"✗ Error sending command: {e}")
            return False
    
    async def listen(self):
        """Listen for messages from the server."""
        if not self.websocket:
            return
        
        self.running = True
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    self.handle_message(data)
                except json.JSONDecodeError:
                    print(f"✗ Invalid JSON received: {message}")
                except Exception as e:
                    print(f"✗ Error handling message: {e}")
        except ConnectionClosed:
            print("✗ Connection closed by server")
            self.running = False
        except Exception as e:
            print(f"✗ Error listening: {e}")
            self.running = False
    
    def handle_message(self, data: dict):
        """Handle a message from the server."""
        msg_type = data.get("type", "unknown")
        timestamp = data.get("timestamp", 0)
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp else ""
        
        if msg_type == "connection":
            print(f"[{time_str}] {data.get('message', 'Connected')}")
        
        elif msg_type == "command_start":
            command_id = data.get("command_id", "unknown")
            command = data.get("command", "")
            print(f"\n[{time_str}] 🚀 Starting: {command}")
            print(f"   Command ID: {command_id}")
            print("-" * 80)
        
        elif msg_type == "node_data":
            node = data.get("node", {})
            step = node.get("step", node.get("iteration", "?"))
            node_id = node.get("node_id", "unknown")
            code = node.get("code", "")
            status = node.get("status", "unknown")
            reward = node.get("reward", 0.0)
            parent_id = node.get("parent_id")
            is_buggy = node.get("is_buggy", False)
            node_type = node.get("node_type", "unknown")
            stage_name = node.get("stage_name", "unknown")
            
            print(f"\n{'='*80}")
            print(f"[{time_str}] 🌳 NODE UPDATE - Step {step}")
            print(f"{'='*80}")
            print(f"  Node ID:      {node_id}")
            print(f"  Parent ID:    {parent_id if parent_id else 'None (root)'}")
            print(f"  Stage:        {stage_name} ({node_type})")
            print(f"  Status:       {status}")
            print(f"  Reward:       {reward:.6f}")
            print(f"  Is Buggy:     {'Yes' if is_buggy else 'No'}")
            print(f"  Code Preview: {code[:200]}{'...' if len(code) > 200 else ''}")
            print(f"{'='*80}\n")
        
        elif msg_type == "command_output":
            output = data.get("output", "")
            if output:
                print(output, end="", flush=True)
        
        elif msg_type == "command_complete":
            command_id = data.get("command_id", "unknown")
            return_code = data.get("return_code", -1)
            status = "✓ SUCCESS" if return_code == 0 else f"✗ FAILED (code: {return_code})"
            print(f"\n[{time_str}] {status} - Command completed: {command_id}")
            print("=" * 80)
        
        elif msg_type == "command_error":
            command_id = data.get("command_id", "unknown")
            error = data.get("error", "Unknown error")
            print(f"\n[{time_str}] ✗ ERROR in {command_id}: {error}")
            print("=" * 80)
        
        elif msg_type == "command_stopped":
            command_id = data.get("command_id", "unknown")
            print(f"\n[{time_str}] ⏹ STOPPED: {command_id}")
            print("=" * 80)
        
        elif msg_type == "file_data":
            file_type = data.get("file_type", "unknown")
            filename = data.get("filename", "unknown")
            file_data = data.get("data", {})
            
            print(f"\n[{time_str}] 📄 File received: {filename} ({file_type})")
            
            if isinstance(file_data, dict):
                # JSON file
                print(f"   Data preview (first 500 chars):")
                preview = json.dumps(file_data, indent=2)[:500]
                print(f"   {preview}...")
            elif isinstance(file_data, str):
                # CSV or text file
                lines = file_data.split('\n')
                print(f"   Total lines: {len(lines)}")
                if lines:
                    print(f"   First few lines:")
                    for line in lines[:5]:
                        print(f"   {line}")
                    if len(lines) > 5:
                        print(f"   ... ({len(lines) - 5} more lines)")
        
        elif msg_type == "file_error":
            filename = data.get("filename", "unknown")
            error = data.get("error", "Unknown error")
            print(f"\n[{time_str}] ✗ File error ({filename}): {error}")
        
        elif msg_type == "error":
            error_msg = data.get("message", "Unknown error")
            print(f"\n[{time_str}] ✗ Error: {error_msg}")
        
        else:
            print(f"\n[{time_str}] Unknown message type: {msg_type}")
            print(f"   Data: {json.dumps(data, indent=2)}")
    
    async def prepare_competition(self, competition_name: str):
        """Send a prepare command to the server."""
        command = {
            "command": "prepare",
            "competition_name": competition_name
        }
        return await self.send_command(command)
    
    async def run_main(self, config_path: str = "config.yaml"):
        """Send a run_main command to the server."""
        command = {
            "command": "run_main",
            "config_path": config_path
        }
        return await self.send_command(command)
    
    async def read_files(self, output_dir: str):
        """Request to read output files from a directory."""
        command = {
            "command": "read_files",
            "output_dir": output_dir
        }
        return await self.send_command(command)
    
    async def stop_command(self, command_id: str):
        """Stop a running command."""
        command = {
            "command": "stop",
            "command_id": command_id
        }
        return await self.send_command(command)
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        print("\n" + "=" * 80)
        print("MLE-Dojo API Client - Interactive Mode")
        print("=" * 80)
        print("\nAvailable commands:")
        print("  <competition_name>          - Run prepare and main.py for a competition")
        print("  prepare <competition_name>  - Run prepare/mle.py for a competition")
        print("  run_main [config_path]      - Run main.py with config (default: config.yaml)")
        print("  read_files <output_dir>     - Read output files from directory")
        print("  stop <command_id>           - Stop a running command")
        print("  quit                        - Exit the client")
        print("\n" + "=" * 80 + "\n")
        
        # Start listening in background
        listen_task = asyncio.create_task(self.listen())
        
        try:
            while self.running:
                try:
                    user_input = await asyncio.to_thread(input, "> ")
                    if not user_input.strip():
                        continue
                    
                    parts = user_input.strip().split()
                    cmd = parts[0].lower()
                    
                    if cmd == "quit" or cmd == "exit":
                        print("Exiting...")
                        self.running = False
                        break
                    
                    elif cmd == "prepare":
                        if len(parts) < 2:
                            print("Usage: prepare <competition_name>")
                        else:
                            competition_name = parts[1]
                            await self.prepare_competition(competition_name)
                    
                    elif cmd == "run_main":
                        config_path = parts[1] if len(parts) > 1 else "config.yaml"
                        await self.run_main(config_path)
                    
                    elif cmd == "read_files":
                        if len(parts) < 2:
                            print("Usage: read_files <output_dir>")
                        else:
                            output_dir = parts[1]
                            await self.read_files(output_dir)
                    
                    elif cmd == "stop":
                        if len(parts) < 2:
                            print("Usage: stop <command_id>")
                        else:
                            command_id = parts[1]
                            await self.stop_command(command_id)
                    
                    else:
                        # Treat as competition name - run prepare and main.py
                        competition_name = user_input.strip()
                        await self.run_competition(competition_name)
                
                except EOFError:
                    # Handle Ctrl+D
                    print("\nExiting...")
                    self.running = False
                    break
                except KeyboardInterrupt:
                    print("\nExiting...")
                    self.running = False
                    break
        
        finally:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass


    async def run_competition(self, competition_name: str):
        """Run prepare and main.py for a competition."""
        command = {
            "command": "run_competition",
            "competition_name": competition_name
        }
        return await self.send_command(command)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MLE-Dojo API Client")
    parser.add_argument("competition_name", nargs="?", type=str, help="Competition name to run")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    
    args = parser.parse_args()
    
    client = APIClient(host=args.host, port=args.port)
    
    if not await client.connect():
        sys.exit(1)
    
    try:
        if args.competition_name:
            # Run prepare and main.py for the competition
            await client.run_competition(args.competition_name)
            await client.listen()
        else:
            # Interactive mode if no competition name provided
            await client.interactive_mode()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")

