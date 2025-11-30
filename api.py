"""
Socket API server for consuming data from prepare/mle.py and main.py
and serving it to frontend clients via WebSocket.

Usage:
    python api.py --host localhost --port 8765
"""

import asyncio
import json
import os
import subprocess
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import yaml
import websockets
from websockets.server import serve
from websockets.exceptions import ConnectionClosed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_server')


class DataStreamer:
    """Streams data from prepare/mle.py and main.py commands to connected clients."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.current_processes = {}
        
    async def register_client(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        await self.send_to_client(websocket, {
            "type": "connection",
            "status": "connected",
            "message": "Connected to MLE-Dojo API server"
        })
    
    async def unregister_client(self, websocket):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_to_client(self, websocket, data: Dict[str, Any]):
        """Send data to a specific client."""
        try:
            await websocket.send(json.dumps(data))
        except ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"Error sending data to client: {e}")
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients."""
        if not self.clients:
            return
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(data))
            except ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            await self.unregister_client(client)
    
    async def run_command(
        self,
        command: list,
        command_id: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """Run a command and stream its output in real-time."""
        logger.info(f"Running command: {' '.join(command)}")
        
        await self.broadcast({
            "type": "command_start",
            "command_id": command_id,
            "command": " ".join(command),
            "timestamp": asyncio.get_event_loop().time()
        })
        
        try:
            # Merge with current environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # Start the process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=process_env,
                bufsize=0  # Unbuffered
            )
            
            self.current_processes[command_id] = process
            
            # Stream output line by line
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                output = line.decode('utf-8', errors='replace').rstrip()
                if output:
                    await self.broadcast({
                        "type": "command_output",
                        "command_id": command_id,
                        "output": output,
                        "timestamp": asyncio.get_event_loop().time()
                    })
            
            # Wait for process to complete
            return_code = await process.wait()
            
            await self.broadcast({
                "type": "command_complete",
                "command_id": command_id,
                "return_code": return_code,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            if command_id in self.current_processes:
                del self.current_processes[command_id]
            
            return return_code
            
        except Exception as e:
            logger.error(f"Error running command: {e}")
            await self.broadcast({
                "type": "command_error",
                "command_id": command_id,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            })
            if command_id in self.current_processes:
                del self.current_processes[command_id]
            return -1
    
    async def prepare_competition(self, competition_name: str):
        """Run prepare/mle.py for a competition."""
        command_id = f"prepare_{competition_name}"
        
        # Get the project root directory
        project_root = Path(__file__).parent.resolve()
        prepare_script = project_root / "prepare" / "mle.py"
        
        command = [
            sys.executable,
            str(prepare_script),
            "--competitions",
            competition_name
        ]
        
        return await self.run_command(command, command_id, cwd=str(project_root))
    
    async def run_main(self, config_path: str = "config.yaml", competition_name: str = None):
        """Run main.py with a config file."""
        command_id = f"main_{Path(config_path).stem}"
        
        # Get the project root directory
        project_root = Path(__file__).parent.resolve()
        main_script = project_root / "main.py"
        config_file = project_root / config_path
        
        if not config_file.exists():
            await self.broadcast({
                "type": "command_error",
                "command_id": command_id,
                "error": f"Config file not found: {config_path}",
                "timestamp": asyncio.get_event_loop().time()
            })
            return -1
        
        # Update config with competition name if provided
        if competition_name:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if 'competition' not in config:
                    config['competition'] = {}
                config['competition']['name'] = competition_name
                
                # Write updated config to a temporary file
                temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                yaml.dump(config, temp_config)
                temp_config_path = temp_config.name
                temp_config.close()
                
                config_file = Path(temp_config_path)
            except Exception as e:
                logger.error(f"Error updating config: {e}")
                await self.broadcast({
                    "type": "command_error",
                    "command_id": command_id,
                    "error": f"Failed to update config: {e}",
                    "timestamp": asyncio.get_event_loop().time()
                })
                return -1
        
        command = [
            sys.executable,
            str(main_script),
            "--config",
            str(config_file)
        ]
        
        return_code = await self.run_command(command, command_id, cwd=str(project_root))
        
        # Clean up temp config file if created
        if competition_name and config_file != project_root / config_path:
            try:
                config_file.unlink()
            except:
                pass
        
        return return_code
    
    async def run_competition(self, competition_name: str, config_path: str = "config.yaml"):
        """Run prepare and then main.py for a competition."""
        # First, run prepare
        prepare_code = await self.prepare_competition(competition_name)
        
        if prepare_code != 0:
            await self.broadcast({
                "type": "command_error",
                "command_id": f"run_competition_{competition_name}",
                "error": f"Prepare failed with code {prepare_code}",
                "timestamp": asyncio.get_event_loop().time()
            })
            return prepare_code
        
        # Then run main.py with the competition name
        return await self.run_main(config_path, competition_name)
    
    async def read_output_files(self, output_dir: str):
        """Read and send output files from the output directory."""
        output_path = Path(output_dir)
        
        if not output_path.exists():
            await self.broadcast({
                "type": "file_error",
                "error": f"Output directory not found: {output_dir}",
                "timestamp": asyncio.get_event_loop().time()
            })
            return
        
        # Find all output files
        files_to_read = {
            "trajectory": "agent_trajectory.json",
            "scores": "scores.csv",
            "prompts": "prompts.csv",
            "cost_history": "cost_history.json",
            "fix_parse": "fix_parse_error.json"
        }
        
        for file_type, filename in files_to_read.items():
            file_path = output_path / filename
            if file_path.exists():
                try:
                    if file_path.suffix == '.json':
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                    else:  # CSV
                        with open(file_path, 'r') as f:
                            data = f.read()
                    
                    await self.broadcast({
                        "type": "file_data",
                        "file_type": file_type,
                        "filename": filename,
                        "data": data,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    await self.broadcast({
                        "type": "file_error",
                        "filename": filename,
                        "error": str(e),
                        "timestamp": asyncio.get_event_loop().time()
                    })
    
    async def handle_client(self, websocket, path):
        """Handle a client connection."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get("command")
                    
                    logger.info(f"Received command: {command}")
                    
                    if command == "prepare":
                        competition_name = data.get("competition_name")
                        if not competition_name:
                            await self.send_to_client(websocket, {
                                "type": "error",
                                "message": "competition_name is required for prepare command"
                            })
                        else:
                            await self.prepare_competition(competition_name)
                    
                    elif command == "run_competition":
                        competition_name = data.get("competition_name")
                        if not competition_name:
                            await self.send_to_client(websocket, {
                                "type": "error",
                                "message": "competition_name is required for run_competition command"
                            })
                        else:
                            config_path = data.get("config_path", "config.yaml")
                            return_code = await self.run_competition(competition_name, config_path)
                            
                            # After main.py completes, read output files
                            if return_code == 0:
                                # Get output directory from config
                                project_root = Path(__file__).parent.resolve()
                                config_file = project_root / config_path
                                
                                if config_file.exists():
                                    with open(config_file, 'r') as f:
                                        config = yaml.safe_load(f)
                                    
                                    comp_name = config.get('competition', {}).get('name', competition_name)
                                    output_dir = Path(config.get('output_dir', 'output')) / comp_name
                                    
                                    await self.read_output_files(str(output_dir))
                    
                    elif command == "run_main":
                        config_path = data.get("config_path", "config.yaml")
                        competition_name = data.get("competition_name")
                        return_code = await self.run_main(config_path, competition_name)
                        
                        # After main.py completes, read output files
                        if return_code == 0:
                            # Get output directory from config
                            project_root = Path(__file__).parent.resolve()
                            config_file = project_root / config_path
                            
                            if config_file.exists():
                                with open(config_file, 'r') as f:
                                    config = yaml.safe_load(f)
                                
                                comp_name = config.get('competition', {}).get('name', competition_name or 'default')
                                output_dir = Path(config.get('output_dir', 'output')) / comp_name
                                
                                await self.read_output_files(str(output_dir))
                    
                    elif command == "read_files":
                        output_dir = data.get("output_dir")
                        if not output_dir:
                            await self.send_to_client(websocket, {
                                "type": "error",
                                "message": "output_dir is required for read_files command"
                            })
                        else:
                            await self.read_output_files(output_dir)
                    
                    elif command == "stop":
                        command_id = data.get("command_id")
                        if command_id and command_id in self.current_processes:
                            process = self.current_processes[command_id]
                            process.terminate()
                            await self.broadcast({
                                "type": "command_stopped",
                                "command_id": command_id,
                                "timestamp": asyncio.get_event_loop().time()
                            })
                    
                    else:
                        logger.warning(f"Unknown command received: {command}, data: {data}")
                        await self.send_to_client(websocket, {
                            "type": "error",
                            "message": f"Unknown command: {command}. Available commands: prepare, run_competition, run_main, read_files, stop"
                        })
                
                except json.JSONDecodeError:
                    await self.send_to_client(websocket, {
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
                    await self.send_to_client(websocket, {
                        "type": "error",
                        "message": str(e)
                    })
        
        except ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        async with serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MLE-Dojo Socket API Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    
    args = parser.parse_args()
    
    streamer = DataStreamer(host=args.host, port=args.port)
    
    try:
        asyncio.run(streamer.start_server())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")


if __name__ == "__main__":
    main()

