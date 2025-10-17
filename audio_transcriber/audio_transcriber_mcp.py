#!/usr/bin/python
# coding: utf-8

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from fastmcp import FastMCP, Context
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from audio_transcriber.audio_transcriber import AudioTranscriber, setup_logging

# Initialize logging for MCP server (logs to file, verbose for details)
logger = setup_logging(verbose=True, log_file="audio_transcriber_mcp.log")

mcp = FastMCP(name="AudioTranscriberServer")

# Environment variables for defaults
environment_model = os.environ.get("WHISPER_MODEL", "base")
environment_directory = os.environ.get(
    "TRANSCRIBE_DIRECTORY", str(Path.home() / "Downloads")
)


@mcp.tool(
    annotations={
        "title": "Transcribe Audio",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"audio_processing"},
)
async def transcribe_audio(
    audio_file: Optional[str] = Field(
        description="Path to the audio file to transcribe. If provided, transcription is performed on this file.",
        default=None,
    ),
    record_seconds: int = Field(
        description="Number of seconds to record audio from microphone. Must be positive if no audio_file is provided. 0 or negative not supported for recording in this context.",
        default=0,
    ),
    directory: Optional[str] = Field(
        description="Directory for saving recordings or exports.",
        default=environment_directory,
    ),
    model: str = Field(
        description="Whisper model to use (e.g., 'base', 'small', 'turbo').",
        default=environment_model,
    ),
    language: Optional[str] = Field(
        description="Language code for transcription (e.g., 'en', 'fr'). Auto-detected if not specified.",
        default=None,
    ),
    task: str = Field(
        description="Task to perform: 'transcribe' or 'translate' (to English).",
        default="transcribe",
    ),
    fp16: bool = Field(description="Use FP16 for faster inference.", default=True),
    word_timestamps: bool = Field(
        description="Include word-level timestamps in the output.", default=False
    ),
    temperature: float = Field(
        description="Temperature for sampling diversity (0.0 for deterministic).",
        default=0.0,
    ),
    initial_prompt: Optional[str] = Field(
        description="Initial text prompt to guide the transcription.", default=None
    ),
    export_formats: List[str] = Field(
        description="Formats to export the transcription (e.g., ['txt', 'srt']).",
        default=None,
    ),
    ctx: Context = Field(
        description="MCP context for progress reporting.", default=None
    ),
) -> str:
    """Transcribes audio from a provided file or by recording from the microphone."""
    logger.info(
        f"Starting transcription: audio_file={audio_file}, record_seconds={record_seconds}, "
        f"directory={directory}, model={model}, language={language}, task={task}"
    )

    try:
        if not audio_file and record_seconds <= 0:
            raise ValueError(
                "Either audio_file must be provided or record_seconds must be positive."
            )

        # Create transcriber instance
        transcriber = AudioTranscriber(
            model=model,
            directory=Path(directory),
            file=audio_file if audio_file else None,
            logger=logger,
        )

        # Report initial progress
        if ctx:
            await ctx.report_progress(progress=0, total=100)
            logger.debug("Reported initial progress: 0/100")

        if audio_file:
            # Validate file existence
            file_path = Path(audio_file)
            if not file_path.exists():
                raise ValueError(f"Audio file not found: {audio_file}")
        else:
            # Recording mode (only fixed duration supported)
            logger.info(f"Starting recording for {record_seconds} seconds.")
            transcriber.initiate_stream()

            # Coarse progress for recording (sync call, so limited granularity)
            transcriber.record(seconds=record_seconds)
            transcriber.stop_stream()
            transcriber.save_stream()

            if ctx:
                await ctx.report_progress(
                    progress=40, total=100
                )  # Arbitrary midpoint after recording
                logger.debug("Reported progress after recording: 40/100")

        # Perform transcription
        logger.info("Starting Whisper transcription.")
        result = transcriber.transcribe(
            language=language,
            task=task,
            fp16=fp16,
            word_timestamps=word_timestamps,
            temperature=temperature,
            initial_prompt=initial_prompt,
            verbose=True,  # Enable verbose for logging details
        )

        if ctx:
            await ctx.report_progress(progress=90, total=100)
            logger.debug("Reported progress after transcription: 90/100")

        # Export if requested
        if export_formats:
            transcriber.export(result, formats=export_formats)
            logger.info(f"Exported transcription to formats: {export_formats}")

        # Report completion
        if ctx:
            await ctx.report_progress(progress=100, total=100)
            logger.debug("Reported final progress: 100/100")

        logger.info("Transcription completed successfully.")
        return result["text"]
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {str(e)}")
        raise RuntimeError(f"Failed to transcribe audio: {str(e)}")


def audio_transcriber_mcp():
    parser = argparse.ArgumentParser(
        description="Audio Transcriber MCP - Run in stdio or http mode"
    )
    parser.add_argument(
        "-t",
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="Transport method: 'stdio', 'http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-s",
        "--host",
        default="0.0.0.0",
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port number for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--auth-type",
        default="none",
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    # JWT/Token params
    parser.add_argument(
        "--token-jwks-uri", default=None, help="JWKS URI for JWT verification"
    )
    parser.add_argument(
        "--token-issuer", default=None, help="Issuer for JWT verification"
    )
    parser.add_argument(
        "--token-audience", default=None, help="Audience for JWT verification"
    )
    # OAuth Proxy params
    parser.add_argument(
        "--oauth-upstream-auth-endpoint",
        default=None,
        help="Upstream authorization endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-token-endpoint",
        default=None,
        help="Upstream token endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-id",
        default=None,
        help="Upstream client ID for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-secret",
        default=None,
        help="Upstream client secret for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-base-url", default=None, help="Base URL for OAuth Proxy"
    )
    # OIDC Proxy params
    parser.add_argument(
        "--oidc-config-url", default=None, help="OIDC configuration URL"
    )
    parser.add_argument("--oidc-client-id", default=None, help="OIDC client ID")
    parser.add_argument("--oidc-client-secret", default=None, help="OIDC client secret")
    parser.add_argument("--oidc-base-url", default=None, help="Base URL for OIDC Proxy")
    # Remote OAuth params
    parser.add_argument(
        "--remote-auth-servers",
        default=None,
        help="Comma-separated list of authorization servers for Remote OAuth",
    )
    parser.add_argument(
        "--remote-base-url", default=None, help="Base URL for Remote OAuth"
    )
    # Common
    parser.add_argument(
        "--allowed-client-redirect-uris",
        default=None,
        help="Comma-separated list of allowed client redirect URIs",
    )
    # Eunomia params
    parser.add_argument(
        "--eunomia-type",
        default="none",
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default="mcp_policies.json",
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url", default=None, help="URL for remote Eunomia server"
    )

    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    # Set auth based on type
    auth = None
    allowed_uris = (
        args.allowed_client_redirect_uris.split(",")
        if args.allowed_client_redirect_uris
        else None
    )

    if args.auth_type == "none":
        auth = None
    elif args.auth_type == "static":
        # Internal static tokens (hardcoded example)
        auth = StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
    elif args.auth_type == "jwt":
        if not (args.token_jwks_uri and args.token_issuer and args.token_audience):
            print(
                "Error: jwt requires --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        auth = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
    elif args.auth_type == "oauth-proxy":
        if not (
            args.oauth_upstream_auth_endpoint
            and args.oauth_upstream_token_endpoint
            and args.oauth_upstream_client_id
            and args.oauth_upstream_client_secret
            and args.oauth_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            print(
                "Error: oauth-proxy requires --oauth-upstream-auth-endpoint, --oauth-upstream-token-endpoint, --oauth-upstream-client-id, --oauth-upstream-client-secret, --oauth-base-url, --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = OAuthProxy(
            upstream_authorization_endpoint=args.oauth_upstream_auth_endpoint,
            upstream_token_endpoint=args.oauth_upstream_token_endpoint,
            upstream_client_id=args.oauth_upstream_client_id,
            upstream_client_secret=args.oauth_upstream_client_secret,
            token_verifier=token_verifier,
            base_url=args.oauth_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "oidc-proxy":
        if not (
            args.oidc_config_url
            and args.oidc_client_id
            and args.oidc_client_secret
            and args.oidc_base_url
        ):
            print(
                "Error: oidc-proxy requires --oidc-config-url, --oidc-client-id, --oidc-client-secret, --oidc-base-url"
            )
            sys.exit(1)
        auth = OIDCProxy(
            config_url=args.oidc_config_url,
            client_id=args.oidc_client_id,
            client_secret=args.oidc_client_secret,
            base_url=args.oidc_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "remote-oauth":
        if not (
            args.remote_auth_servers
            and args.remote_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            print(
                "Error: remote-oauth requires --remote-auth-servers, --remote-base-url, --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        auth_servers = [url.strip() for url in args.remote_auth_servers.split(",")]
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=auth_servers,
            base_url=args.remote_base_url,
        )
    mcp.auth = auth
    if args.eunomia_type != "none":
        from eunomia_mcp import create_eunomia_middleware

        if args.eunomia_type == "embedded":
            if not args.eunomia_policy_file:
                print("Error: embedded Eunomia requires --eunomia-policy-file")
                sys.exit(1)
            middleware = create_eunomia_middleware(policy_file=args.eunomia_policy_file)
            mcp.add_middleware(middleware)
        elif args.eunomia_type == "remote":
            if not args.eunomia_remote_url:
                print("Error: remote Eunomia requires --eunomia-remote-url")
                sys.exit(1)
            middleware = create_eunomia_middleware(
                use_remote_eunomia=args.eunomia_remote_url
            )
            mcp.add_middleware(middleware)

    mcp.add_middleware(
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True)
    )
    mcp.add_middleware(
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20)
    )
    mcp.add_middleware(TimingMiddleware())
    mcp.add_middleware(LoggingMiddleware())

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger = logging.getLogger("ContainerManager")
        logger.error("Transport not supported")
        sys.exit(1)


if __name__ == "__main__":
    audio_transcriber_mcp()
