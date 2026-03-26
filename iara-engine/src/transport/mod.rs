// ============================================================================
// websocket.rs — Servidor WebSocket para streaming de telemetria
// ============================================================================
// Protocolo:
//
// Frontend → Backend:
//   {"type": "stimulus", "text": "carro"}      — Enviar estímulo
//   {"type": "select_layer", "name": "..."}    — Trocar tensor ativo
//   {"type": "reset"}                          — Resetar simulação
//   {"type": "list_layers"}                    — Listar tensores disponíveis
//
// Backend → Frontend:
//   {"type": "frame", "grid_size": 128, "data": [...], "tick": N,
//    "layer": "...", "energy": 0.0, "active_nodes": N, "total_nodes": N}
//   {"type": "layers", "names": [...]}
//   {"type": "status", "message": "..."}
//   {"type": "loading", "layer": "...", "progress": "..."}
// ============================================================================

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

/// Mensagens que o frontend envia para o backend
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "stimulus")]
    Stimulus { text: String },

    #[serde(rename = "resonate")]
    Resonate { text: String },

    #[serde(rename = "select_layer")]
    SelectLayer { name: String },

    #[serde(rename = "reset")]
    Reset,

    #[serde(rename = "list_layers")]
    ListLayers,
}

/// Frame de telemetria enviado pelo backend
#[derive(Debug, Serialize, Clone)]
pub struct TelemetryFrame {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub grid_size: usize,
    pub data: Vec<f32>,
    pub tick: u64,
    pub layer: String,
    pub energy: f32,
    pub active_nodes: usize,
    pub total_nodes: usize,
}

/// Mensagem genérica do backend para o frontend
#[derive(Debug, Serialize)]
pub struct ServerMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub names: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<String>,
}

/// Comandos internos enviados do handler WS para o loop principal
#[derive(Debug)]
pub enum EngineCommand {
    Stimulus(String),
    Resonate(String),
    SelectLayer(String),
    Reset,
    ListLayers,
}

/// Inicia o servidor WebSocket
pub async fn start_ws_server(
    addr: SocketAddr,
    cmd_tx: mpsc::Sender<EngineCommand>,
    frame_rx: tokio::sync::broadcast::Receiver<String>,
) {
    let listener = TcpListener::bind(addr)
        .await
        .expect("❌ Falha ao bind do WebSocket server");

    log::info!("🌐 WebSocket server escutando em ws://{}", addr);
    log::info!("🖥️  Abra frontend/index.html no navegador para visualizar");

    loop {
        match listener.accept().await {
            Ok((stream, peer)) => {
                log::info!("🔗 Nova conexão WebSocket: {}", peer);
                let cmd_tx = cmd_tx.clone();
                let frame_rx = frame_rx.resubscribe();
                tokio::spawn(handle_connection(stream, peer, cmd_tx, frame_rx));
            }
            Err(e) => {
                log::error!("❌ Erro ao aceitar conexão: {}", e);
            }
        }
    }
}

/// Handler de uma conexão WebSocket individual
async fn handle_connection(
    stream: TcpStream,
    peer: SocketAddr,
    cmd_tx: mpsc::Sender<EngineCommand>,
    mut frame_rx: tokio::sync::broadcast::Receiver<String>,
) {
    let ws_stream = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            log::error!("❌ Falha no handshake WS com {}: {}", peer, e);
            return;
        }
    };

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Enviar mensagem de boas-vindas
    let welcome = ServerMessage {
        msg_type: "status".to_string(),
        message: Some("🧬 IARA Resonance Engine — Conectado".to_string()),
        names: None,
        layer: None,
        progress: None,
    };
    if let Ok(json) = serde_json::to_string(&welcome) {
        let _ = ws_sender
            .send(tokio_tungstenite::tungstenite::Message::Text(json.into()))
            .await;
    }

    // Task para receber frames da simulação e enviar ao cliente
    let send_task = tokio::spawn(async move {
        loop {
            match frame_rx.recv().await {
                Ok(frame_json) => {
                    if ws_sender
                        .send(tokio_tungstenite::tungstenite::Message::Text(frame_json.into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    log::warn!("⚠️  Cliente {} atrasado, {} frames perdidos", peer, n);
                    continue;
                }
                Err(_) => break,
            }
        }
    });

    // Loop para receber mensagens do cliente
    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(client_msg) => {
                        let cmd = match client_msg {
                            ClientMessage::Stimulus { text } => EngineCommand::Stimulus(text),
                            ClientMessage::Resonate { text } => EngineCommand::Resonate(text),
                            ClientMessage::SelectLayer { name } => EngineCommand::SelectLayer(name),
                            ClientMessage::Reset => EngineCommand::Reset,
                            ClientMessage::ListLayers => EngineCommand::ListLayers,
                        };
                        if cmd_tx.send(cmd).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        log::warn!("⚠️  Mensagem inválida de {}: {} — {}", peer, text, e);
                    }
                }
            }
            Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => {
                log::info!("👋 Cliente {} desconectou", peer);
                break;
            }
            Err(e) => {
                log::error!("❌ Erro WS de {}: {}", peer, e);
                break;
            }
            _ => {}
        }
    }

    send_task.abort();
    log::info!("🔌 Conexão encerrada: {}", peer);
}
