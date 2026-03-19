use crate::state::NodeState;
use noddle_proto::{
    admin_service_server::AdminService,
    node_service_server::NodeService,
    client_service_server::ClientService,
    AnnounceAck, AnnounceRequest,
    CancelAck, CancelRequest,
    JobMessage,
    NodeStatusRequest, NodeStatusResponse,
    PingRequest, PingResponse,
    PromptRequest, RegistrySyncAck, RegistrySyncRequest,
    SetActiveRequest, SetActiveResponse,
    TokenChunk,
};
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

type BoxStream<T> = Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send>>;

// ── NodeService implementation ────────────────────────────────────────────────

pub struct NodeServiceImpl {
    pub state: Arc<NodeState>,
}

#[tonic::async_trait]
impl NodeService for NodeServiceImpl {
    type ExecuteLayerStream = BoxStream<JobMessage>;

    async fn execute_layer(
        &self,
        request: Request<JobMessage>,
    ) -> Result<Response<Self::ExecuteLayerStream>, Status> {
        let job = request.into_inner();
        info!(job_id = %job.job_id, "execute_layer received");

        // Dedup guard: if we've seen this job before, drop it
        if self.state.router.is_duplicate(&job.job_id) {
            warn!(job_id = %job.job_id, "duplicate job dropped");
            return Err(Status::already_exists("duplicate job"));
        }
        self.state.router.mark_seen(&job.job_id);

        // Apply any piggybacked registry diff
        if let Some(ref diff) = job.registry_diff {
            let mut reg = self.state.registry.write().await;
            let n = reg.merge(diff);
            if n > 0 {
                info!(merged = n, "registry updated from piggybacked diff");
            }
        }

        // ADMIN nodes do not run inference — they should not receive ExecuteLayer
        if self.state.role == noddle_proto::NodeRole::Admin {
            return Err(Status::unimplemented("admin node cannot execute layers"));
        }

        let state = self.state.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(16);

        tokio::spawn(async move {
            match state.run_job(job).await {
                Ok(result) => {
                    let _ = tx.send(Ok(result)).await;
                }
                Err(e) => {
                    warn!(error = %e, "job execution failed");
                    let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn cancel_job(
        &self,
        request: Request<CancelRequest>,
    ) -> Result<Response<CancelAck>, Status> {
        let req = request.into_inner();
        info!(job_id = %req.job_id, "cancel_job received");
        self.state.cancel_job(&req.job_id);
        Ok(Response::new(CancelAck { accepted: true }))
    }

    async fn sync_registry(
        &self,
        request: Request<RegistrySyncRequest>,
    ) -> Result<Response<RegistrySyncAck>, Status> {
        let req = request.into_inner();

        // Merge their diff into our registry
        if let Some(ref diff) = req.diff {
            let mut reg = self.state.registry.write().await;
            reg.merge(diff);
        }

        // Return our diff back to them
        let our_diff = {
            let reg = self.state.registry.read().await;
            reg.diff_since(req.since_ms, &self.state.node_id)
        };

        Ok(Response::new(RegistrySyncAck { diff: Some(our_diff) }))
    }

    async fn ping(
        &self,
        _request: Request<PingRequest>,
    ) -> Result<Response<PingResponse>, Status> {
        let load = self.state.current_load();
        Ok(Response::new(PingResponse {
            node_id:      self.state.node_id.clone(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            current_load: load,
        }))
    }

    async fn announce(
        &self,
        request: Request<AnnounceRequest>,
    ) -> Result<Response<AnnounceAck>, Status> {
        if let Some(cap) = request.into_inner().capability {
            info!(node_id = %cap.node_id, "announce received");
            let mut reg = self.state.registry.write().await;
            reg.upsert(cap);
        }
        Ok(Response::new(AnnounceAck { accepted: true }))
    }
}

// ── AdminService implementation ───────────────────────────────────────────────

pub struct AdminServiceImpl {
    pub state: Arc<NodeState>,
}

#[tonic::async_trait]
impl AdminService for AdminServiceImpl {
    async fn set_active(
        &self,
        request: Request<SetActiveRequest>,
    ) -> Result<Response<SetActiveResponse>, Status> {
        let active = request.into_inner().active;
        self.state
            .set_active(active)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(SetActiveResponse { active }))
    }

    async fn get_status(
        &self,
        _request: Request<NodeStatusRequest>,
    ) -> Result<Response<NodeStatusResponse>, Status> {
        let active = self.state.is_active().await;
        let model_ids = if active {
            self.state.discovered_model_ids.clone()
        } else {
            vec![]
        };
        Ok(Response::new(NodeStatusResponse {
            node_id:      self.state.node_id.clone(),
            role:         self.state.role as i32,
            active,
            current_load: self.state.current_load(),
            model_ids,
            vram_mb:      Some(self.state.vram_mb()).filter(|&v| v > 0),
        }))
    }
}

// ── ClientService implementation ──────────────────────────────────────────────

pub struct ClientServiceImpl {
    pub state: Arc<NodeState>,
}

#[tonic::async_trait]
impl ClientService for ClientServiceImpl {
    type SubmitPromptStream = BoxStream<TokenChunk>;

    async fn submit_prompt(
        &self,
        request: Request<PromptRequest>,
    ) -> Result<Response<Self::SubmitPromptStream>, Status> {
        let req = request.into_inner();
        info!(model_id = %req.model_id, "submit_prompt received");

        let state = self.state.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(async move {
            match state.handle_prompt(req).await {
                Ok(text) => {
                    // Stream the response word by word so the CLI can render
                    // tokens as they arrive. The real implementation will stream
                    // directly from the final layer's decode loop.
                    for word in text.split_whitespace() {
                        let chunk = TokenChunk {
                            session_id: String::new(),
                            text: format!("{} ", word),
                            done: false,
                            error: String::new(),
                        };
                        if tx.send(Ok(chunk)).await.is_err() {
                            break;
                        }
                    }
                    let _ = tx.send(Ok(TokenChunk {
                        session_id: String::new(),
                        text:       String::new(),
                        done:       true,
                        error:      String::new(),
                    })).await;
                }
                Err(e) => {
                    let _ = tx.send(Ok(TokenChunk {
                        session_id: String::new(),
                        text:       String::new(),
                        done:       true,
                        error:      e.to_string(),
                    })).await;
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}
