mod pull;

use anyhow::{bail, Context, Result};
use noddle_proto::{client_service_client::ClientServiceClient, PromptRequest};
use std::io::{self, BufRead, Write};
use tracing::warn;
use uuid::Uuid;

const DEFAULT_GRPC: &str = "http://127.0.0.1:7900";


#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();

    // Handle management subcommands before connecting to the node.
    if args.get(1).map(String::as_str) == Some("pull") {
        let model_id = args.get(2)
            .ok_or_else(|| anyhow::anyhow!("usage: noddle pull <model_id>"))?;
        return pull::pull(model_id).await;
    }
    if args.get(1).map(String::as_str) == Some("models") {
        return pull::list_available();
    }

    let cli_args = parse_args(&args)?;

    let mut client = ClientServiceClient::new(connect().await?);

    // A single-shot prompt was passed on the command line — run it and exit.
    if let Some(prompt) = cli_args.prompt {
        send_and_stream(&mut client, &cli_args.model_id, &prompt, &cli_args.session_id).await?;
        return Ok(());
    }

    // No prompt given — enter interactive session loop.
    run_session(&mut client, &cli_args.model_id, &cli_args.session_id).await
}

async fn run_session(
    client: &mut ClientServiceClient<tonic::transport::Channel>,
    model_id: &str,
    session_id: &str,
) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();

    loop {
        {
            let mut out = stdout.lock();
            write!(out, "> ")?;
            out.flush()?;
        }

        let mut line = String::new();
        let bytes_read = stdin.lock().read_line(&mut line)?;

        // EOF (Ctrl+D)
        if bytes_read == 0 {
            break;
        }

        let prompt = line.trim();

        if prompt.is_empty() {
            continue;
        }

        if matches!(prompt, "exit" | "quit" | ":q") {
            break;
        }

        send_and_stream(client, model_id, prompt, session_id).await?;
        println!();
    }

    Ok(())
}

async fn send_and_stream(
    client: &mut ClientServiceClient<tonic::transport::Channel>,
    model_id: &str,
    prompt: &str,
    session_id: &str,
) -> Result<()> {
    let mut stream = client
        .submit_prompt(PromptRequest {
            session_id:  session_id.to_string(),
            model_id:    model_id.to_string(),
            prompt_text: prompt.to_string(),
            private_mode: false,
        })
        .await
        .context("failed to submit prompt")?
        .into_inner();

    let stdout = io::stdout();
    let mut out = stdout.lock();

    loop {
        match stream.message().await {
            Ok(Some(chunk)) => {
                if !chunk.error.is_empty() {
                    bail!("node error: {}", chunk.error);
                }
                write!(out, "{}", chunk.text)?;
                out.flush()?;
                if chunk.done {
                    break;
                }
            }
            Ok(None) => break,
            Err(e) => bail!("stream error: {}", e),
        }
    }

    Ok(())
}

// ── Args ──────────────────────────────────────────────────────────────────────

struct CliArgs {
    model_id:   String,
    session_id: String,
    /// If Some, run single-shot and exit. If None, enter interactive loop.
    prompt:     Option<String>,
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    let mut model_id = "Qwen/Qwen2.5-7B-Instruct".to_string();
    let mut session_id = Uuid::new_v4().to_string();
    let mut prompt_parts: Vec<String> = Vec::new();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                model_id = args.get(i).context("--model requires a value")?.clone();
            }
            "--session" | "-s" => {
                i += 1;
                session_id = args.get(i).context("--session requires a value")?.clone();
            }
            arg if arg.starts_with('-') => {
                bail!("unknown flag: {}", arg);
            }
            _ => prompt_parts.push(args[i].clone()),
        }
        i += 1;
    }

    let prompt = if prompt_parts.is_empty() {
        None
    } else {
        Some(prompt_parts.join(" "))
    };

    Ok(CliArgs { model_id, session_id, prompt })
}

// ── Connection ────────────────────────────────────────────────────────────────

/// Build a connected gRPC channel.
///
/// On Unix: tries the local Unix socket first (no TLS needed for local IPC),
/// falls back to plain TCP if the socket isn't present.
/// On Windows: connects directly over TCP.
async fn connect() -> Result<tonic::transport::Channel> {
    #[cfg(unix)]
    {
        let socket_path = local_socket_path();
        if std::path::Path::new(&socket_path).exists() {
            return connect_unix(socket_path).await;
        }
        warn!("unix socket not found at {}, falling back to {}", socket_path, DEFAULT_GRPC);
    }

    tonic::transport::Channel::from_static(DEFAULT_GRPC)
        .connect()
        .await
        .context("could not connect to noddle-node — is it running?")
}

/// Connect via a Unix domain socket (local IPC, no TLS required).
///
/// Platform behaviour matches `local_socket_path` in noddle-node:
///   - Linux  — `$XDG_RUNTIME_DIR/noddle.sock`
///   - macOS  — `/tmp/noddle.sock`
#[cfg(unix)]
async fn connect_unix(socket_path: String) -> Result<tonic::transport::Channel> {
    use hyper_util::rt::TokioIo;
    use tokio::net::UnixStream;
    use tonic::transport::{Endpoint, Uri};

    Endpoint::from_static("http://[::]:50051")
        .connect_with_connector(tower::service_fn(move |_: Uri| {
            let path = socket_path.clone();
            async move {
                let stream = UnixStream::connect(path).await?;
                Ok::<_, std::io::Error>(TokioIo::new(stream))
            }
        }))
        .await
        .context("could not connect to noddle-node unix socket")
}

#[cfg(unix)]
fn local_socket_path() -> String {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| "/tmp".to_string());
    format!("{}/noddle.sock", runtime_dir)
}
