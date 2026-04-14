use tokio::process::Command;
use anyhow::Result;

/// Orquestrador para o NotebookLM (NLM).
/// Permite que a IARA consulte e expanda sua base de conhecimento externa.
pub struct NlmOrchestrator;

impl NlmOrchestrator {
    /// Cria um novo notebook no NLM.
    pub async fn create_notebook(title: &str) -> Result<String> {
        let output = Command::new("nlm")
            .arg("notebook")
            .arg("create")
            .arg("--title")
            .arg(title)
            .output()
            .await?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            anyhow::bail!("NLM Create Error: {}", String::from_utf8_lossy(&output.stderr))
        }
    }

    /// Adiciona uma URL como fonte a um notebook existente.
    pub async fn add_source(notebook_id: &str, url: &str) -> Result<String> {
        let output = Command::new("nlm")
            .arg("source")
            .arg("add")
            .arg("--notebook")
            .arg(notebook_id)
            .arg("--type")
            .arg("url")
            .arg("--url")
            .arg(url)
            .output()
            .await?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            anyhow::bail!("NLM Add Source Error: {}", String::from_utf8_lossy(&output.stderr))
        }
    }

    /// Consulta o NotebookLM para extrair conhecimento (RAG).
    pub async fn extract_knowledge(notebook_id: &str, query: &str) -> Result<String> {
        let output = Command::new("nlm")
            .arg("notebook")
            .arg("query")
            .arg("--notebook")
            .arg(notebook_id)
            .arg("--query")
            .arg(query)
            .output()
            .await?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            anyhow::bail!("NLM Query Error: {}", String::from_utf8_lossy(&output.stderr))
        }
    }
}
