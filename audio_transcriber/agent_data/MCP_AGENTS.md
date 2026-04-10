# MCP_AGENTS.md - Dynamic Agent Registry

This file tracks the generated agents from MCP servers. You can manually modify the 'Tools' list to customize agent expertise.

## Agent Mapping Table

| Name | Description | System Prompt | Tools | Tag | Source MCP |
|------|-------------|---------------|-------|-----|------------|
| Audio-Transcriber Audio Processing Specialist | Expert specialist for audio_processing domain tasks. | You are a Audio-Transcriber Audio Processing specialist. Help users manage and interact with Audio Processing functionality using the available tools. | audio-transcriber-mcp_audio_processing_toolset | audio_processing | audio-transcriber-mcp |
| Audio-Transcriber Misc Specialist | Expert specialist for misc domain tasks. | You are a Audio-Transcriber Misc specialist. Help users manage and interact with Misc functionality using the available tools. | audio-transcriber-mcp_misc_toolset | misc | audio-transcriber-mcp |

## Tool Inventory Table

| Tool Name | Description | Tag | Source |
|-----------|-------------|-----|--------|
| audio-transcriber-mcp_audio_processing_toolset | Static hint toolset for audio_processing based on config env. | audio_processing | audio-transcriber-mcp |
| audio-transcriber-mcp_misc_toolset | Static hint toolset for misc based on config env. | misc | audio-transcriber-mcp |
