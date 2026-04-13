# Chess Research — Cloud2 + SAIS Setup

## Architecture

Same pattern as SandstoneAutoResearch:
- **Cloud2** (dev desktop) — runs Claude Code in tmux, persistent
- **SAIS** (ml.g5.8xlarge — A10G 24GB) — runs training, eval, inference via MCP
- **MCP server** — ZachliuPersonal sagemaker-mcp-server connects the two

## Prerequisites

- Claude Code installed on cloud2
- ZachliuPersonal sagemaker-mcp-server available
- SAIS notebook instance running (ml.g5.8xlarge)
- AWS credentials refreshed (`ada`)

## 1. Set up MCP on cloud2

Create `.mcp.json` in the chess-coach workspace root:

```json
{
  "mcpServers": {
    "sagemaker-notebook": {
      "command": "python3",
      "args": ["/home/samtkap/workspace/ZachliuPersonal/sagemaker-mcp-server/server.py"]
    }
  }
}
```

(Adjust path to wherever ZachliuPersonal is on cloud2)

## 2. Refresh credentials

```bash
nohup ada credentials update --account=199342434285 --provider=conduit --role=IibsAdminAccess-DO-NOT-DELETE > /tmp/ada_chess.log 2>&1 &
```

## 3. Verify SAIS access

```bash
cd ~/workspace/chess-coach
claude

# In Claude Code:
# > Use sm_execute to run: import torch; print(torch.cuda.is_available())
# > Use sm_terminal_execute to run: nvidia-smi
```

## 4. Launch research

```bash
cd ~/workspace/chess-coach
tmux new-session -s chess-research
claude --agent research/program.md
```

Detach: Ctrl-B D
Reattach: `tmux attach -t chess-research`

## 5. Check progress

```bash
# Experiment count
wc -l research/experiment_log.jsonl

# Current champion
cat research/configs/champion.json

# Research plan
cat research/research_plan.md

# Recent results
tail -5 research/experiment_log.jsonl | python3 -m json.tool
```

## 6. SAIS workspace

All chess research files live under:
```
/home/ec2-user/SageMaker/chess-research/
```

Separate from SandstoneAgent to avoid conflicts.
