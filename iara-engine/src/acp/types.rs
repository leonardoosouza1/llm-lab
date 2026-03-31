// ============================================================================
// acp/types.rs — ACP Protocol Data Types
// ============================================================================
// Implementa tipos para o protocolo Agent Communication Protocol (ACP v1).
// Inclui mensagens, manifestos de agentes, estados de execução e eventos.
// ============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parte de uma mensagem ACP (text, json, arquivo, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcpMessagePart {
    /// Nome opcional da parte (ex: "image", "code", "explanation")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tipo MIME: "text/plain", "text/json", "application/octet-stream", etc.
    pub content_type: String,

    /// Conteúdo inline (texto ou JSON serializado)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// URL para conteúdo externo (ex: arquivo em disco ou S3)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_url: Option<String>,

    /// Codificação: "utf-8", "base64", "gzip", etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_encoding: Option<String>,
}

/// Mensagem no protocolo ACP — representa texto/prompt do usuário ou resposta do agente
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcpMessage {
    /// "user" ou "agent"
    pub role: String,

    /// Partes multimodais (texto, json, imagens, arquivos, etc.)
    pub parts: Vec<AcpMessagePart>,
}

/// Descrição de um agente (manifesto — capacidades, modelos suportados, metadados)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentManifest {
    /// Nome único do agente (ex: "qwen-3-8b")
    pub name: String,

    /// Descrição legível (ex: "Qwen 3 com 8B parâmetros")
    pub description: String,

    /// Tipos MIME de conteúdo suportados (ex: ["text/plain", "text/json"])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_content_types: Option<Vec<String>>,

    /// Metadados customizados (versão, arquitetura, limite de tokens, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Status de execução de uma run ACP
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Criada, ainda não iniciada
    Created,
    /// Em progresso
    InProgress,
    /// Completada com sucesso
    Completed,
    /// Aguardando entrada do usuário (modo interactive)
    Awaiting,
    /// Erro durante execução
    Failed,
    /// Em processo de cancelamento
    Cancelling,
    /// Cancelada pelo usuário
    Cancelled,
}

/// Requisição para criar e executar uma run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    /// Nome do agente a usar (ex: "qwen-3-8b")
    pub agent_name: String,

    /// Histórico de mensagens (input para o agente)
    pub input: Vec<AcpMessage>,

    /// Modo: "sync" (aguarda resposta), "async" (retorna imediatamente), "stream" (SSE)
    #[serde(default = "default_mode")]
    pub mode: String,

    /// ID da sessão (para manter estado entre runs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Metadados customizados (temperature, top_k, top_p, etc.)
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Resposta de uma run ACP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    /// ID único da run (UUID v4)
    pub run_id: String,

    /// Nome do agente usado
    pub agent_name: String,

    /// Status atual
    pub status: RunStatus,

    /// Mensagens de saída (output do agente) — presente se status = Completed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<AcpMessage>>,

    /// Timestamp ISO 8601 de criação
    pub created_at: String,

    /// Timestamp ISO 8601 de início da execução
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,

    /// Timestamp ISO 8601 de conclusão
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,

    /// Mensagem de erro (se status = Failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Evento rastreável de uma run (ex: geração de token, cache reset, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvent {
    /// Tipo do evento: "token_generated", "layer_processed", "cache_reset", etc.
    pub event_type: String,

    /// Timestamp ISO 8601
    pub timestamp: String,

    /// Dados adicionais (contexto do evento)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Função auxiliar — modo padrão para RunRequest
fn default_mode() -> String {
    "sync".to_string()
}
