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
    let cli_args = parse_args(&args)?;

    let endpoint = resolve_endpoint();
    let mut client = ClientServiceClient::connect(endpoint)
        .await
        .context("could not connect to noddle-node — is it running? (`noddle-node`)")?;

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
    let mut model_id = "meta-llama/Llama-3-8B".to_string();
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

fn resolve_endpoint() -> String {
    let uid = std::env::var("UID")
        .ok()
        .and_then(|u| u.parse::<u32>().ok())
        .unwrap_or(1000);

    let socket_path = format!("/run/user/{}/noddle.sock", uid);

    if std::path::Path::new(&socket_path).exists() {
        format!("unix://{}", socket_path)
    } else {
        warn!("unix socket not found at {}, falling back to {}", socket_path, DEFAULT_GRPC);
        DEFAULT_GRPC.to_string()
    }
}
