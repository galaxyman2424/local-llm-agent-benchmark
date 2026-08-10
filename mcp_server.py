#!/usr/bin/env python3
"""
Local MCP Server for SWE-bench / local-LLM-Agent-Benchmark project.
Provides: file operations | safe code execution (local) | SQLite database access.
VS Code/Claude Desktop Setup - add to ~/.claude_desktop_config.json:
{
    "mcpServers": {
        "local-project-tools": {"command":"python","args":["/home/connor/projects/local-llm-agent-benchmark/mcp_server.py"]}
    }
}from fastapi import FastAPI, HTTPException, Request
import subprocess, os, json, sqlite3, pathlib, shlex
app=FastAPI(title="Local MCP Server",version='1.0');BASE_DIR="/home/connor/projects/local-llm-agent-benchmark" 

def safe_path(path_str):
p=pathlib.Path(os.path.expanduser(str(path_str)).replace('~/projects/', '/home/')).absolute()  
return None if not str(p).startswith(BASE_DIR) else p
@app.get("/tools") def list_tools(): return {'name':'local-project-tools','version':1.0}
@app.post('/api/files/list') async def ls_files(content):
p=getattr(getattr(body,content,{}),'path','.').split()[-1] or '.'
safe_path(p if hasattr(saferequest,path) else getattr(BASE_DIR,'.')); cmd=['ls','-R']+shlex.split(str(path))[:20]; 
return subprocess.run(cmd,capture_output=True,text=True).stdout.strip().replace('\
',' ') or []# ======== PHASE 2 COMPLETE TOOLS =========#
