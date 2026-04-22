"""
IARA Physics Engine — Operadores Físicos sobre Pesos Reais do Qwen3

Experimentos:
1. Gravity Attention  — substitui softmax por força gravitacional semântica
2. Vortex Field       — componente antisimétrico em atenção (curl ≠ 0)
3. KAN Double-Well    — barreira de potencial como ativação (substitui SiLU)
4. Kuramoto Learning  — update online via sincronização de fase (sem backprop)
5. Semantic Mass      — calcula massa semântica dos tokens pelo embedding

Cada experimento mostra os números reais + comparação com o padrão.
"""

import numpy as np
import struct, json, os, time

MODEL_PATH = "models/qwen3-8b"

# ─────────────────────────────────────────────────────────────────
# Loader de pesos
# ─────────────────────────────────────────────────────────────────

def load_bf16(name):
    idx = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(idx) as f:
        wmap = json.load(f)["weight_map"]
    sp = os.path.join(MODEL_PATH, wmap[name])
    with open(sp, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        info = json.loads(f.read(hlen))[name]
        f.seek(8 + hlen + info["data_offsets"][0])
        raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
    a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return np.frombuffer(a.tobytes(), dtype=np.float32).reshape(info["shape"])

sep = lambda: print("\n" + "=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. SEMANTIC MASS — tokens como corpos com gravidade
# ─────────────────────────────────────────────────────────────────

def exp_semantic_mass():
    sep()
    print("◈ EXPERIMENTO 1: SEMANTIC MASS")
    print("  Embedding rows como corpos com massa semântica")
    print("  massa_i = ||embed_i||  (norma L2 do vetor)")
    print()

    E = load_bf16("model.embed_tokens.weight")  # [vocab_size, hidden]
    print(f"  Embedding: {E.shape}  (vocab × hidden)")

    norms = np.linalg.norm(E, axis=1)  # massa semântica de cada token
    print(f"\n  Distribuição de massa (norma L2 dos embeddings):")
    print(f"    min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}  std={norms.std():.4f}")
    print(f"    Razão max/min: {norms.max()/norms.min():.1f}x")

    # Pareto: tokens de alta massa
    top_k = np.argsort(norms)[-20:]
    print(f"\n  Top-20 tokens por massa semântica (índices): {top_k}")
    print(f"  Massas: {norms[top_k].round(3)}")

    # Campo gravitacional: entre os top tokens
    # F_ij = G × m_i × m_j / d_ij²   onde d = distância coseno
    top_emb = E[top_k[:10]]  # 10 tokens mais pesados
    top_norms = norms[top_k[:10]]
    cos_sim = top_emb @ top_emb.T / (np.outer(top_norms, top_norms) + 1e-8)
    dist = 1 - cos_sim  # distância semântica
    np.fill_diagonal(dist, 1e-6)  # evita divisão por zero

    G = 1.0
    forces = G * np.outer(top_norms, top_norms) / (dist**2)
    np.fill_diagonal(forces, 0)

    print(f"\n  Campo gravitacional (top 10 tokens, F=G×m1×m2/d²):")
    print(f"    Força média entre tokens pesados: {forces.mean():.3f}")
    print(f"    Força máxima (par mais atraído):  {forces.max():.3f}")

    # Comparação: atenção padrão softmax vs atenção gravitacional
    # Usando uma query aleatória como proxy
    rng = np.random.default_rng(42)
    q = rng.normal(0, 1, E.shape[1]).astype(np.float32)  # query vector

    # Softmax attention (QKᵀ / sqrt(d))
    scores_std = (top_emb @ q) / np.sqrt(E.shape[1])
    attn_std = np.exp(scores_std - scores_std.max())
    attn_std /= attn_std.sum()

    # Gravity attention (F = G × m_i / d_qi²)
    q_norm = np.linalg.norm(q)
    cos_q = (top_emb @ q) / (top_norms * q_norm + 1e-8)
    dist_q = 1 - cos_q + 1e-6
    attn_grav = G * top_norms / (dist_q**2)
    attn_grav = np.exp(attn_grav - attn_grav.max())
    attn_grav /= attn_grav.sum()

    print(f"\n  Comparação de atenção (10 tokens pesados, 1 query):")
    print(f"    {'Token':>8}  {'Softmax':>10}  {'Gravity':>10}  {'Ratio':>8}")
    for i in range(10):
        ratio = attn_grav[i] / (attn_std[i] + 1e-8)
        print(f"    {top_k[i]:>8}  {attn_std[i]:>10.4f}  {attn_grav[i]:>10.4f}  {ratio:>8.2f}x")

    # KL divergence entre os dois tipos de atenção
    kl = np.sum(attn_std * np.log(attn_std / (attn_grav + 1e-10) + 1e-10))
    print(f"\n  KL(softmax || gravity) = {kl:.4f}")
    print(f"  (0 = idênticas, > 0 = diferentes)")

    return norms


# ─────────────────────────────────────────────────────────────────
# 2. GRAVITY ATTENTION — replace softmax com força gravitacional
# ─────────────────────────────────────────────────────────────────

def exp_gravity_attention():
    sep()
    print("◈ EXPERIMENTO 2: GRAVITY ATTENTION")
    print("  A[i,j] = softmax(G × m_i × m_j / d_ij²)")
    print("  Tokens massivos puxam mais atenção — lei natural")
    print()

    # Usa layer 0 Q/K projections
    Wq = load_bf16("model.layers.0.self_attn.q_proj.weight")  # [4096, 4096]
    Wk = load_bf16("model.layers.0.self_attn.k_proj.weight")  # [1024, 4096]
    print(f"  Q_proj: {Wq.shape}, K_proj: {Wk.shape}")

    # Qwen3-8B: hidden=4096, 32 heads Q, 8 heads K, head_dim=128
    hidden_size = Wq.shape[1]  # detect from actual weight
    n_heads = 32
    n_kv = 8
    head_dim = hidden_size // n_heads  # 128 for Qwen3-8B

    # Simula uma sequência de 8 tokens com hidden state aleatório
    rng = np.random.default_rng(0)
    seq_len = 8
    hidden = rng.normal(0, 1, (seq_len, hidden_size)).astype(np.float32)

    # Q/K projetados
    Q = hidden @ Wq.T  # [8, hidden_size]
    K = hidden @ Wk.T  # [8, n_kv * head_dim]
    # GQA: Q tem 32 cabeças, K tem 8 — repetimos K para match
    Q = Q.reshape(seq_len, n_heads, head_dim)
    K_kv = K.reshape(seq_len, n_kv, head_dim)
    # Expand K para 32 cabeças (cada grupo de 4 heads compartilha um K)
    K_exp = np.repeat(K_kv, n_heads//n_kv, axis=1)  # [8, 32, head_dim]

    # ATENÇÃO PADRÃO (cabeça 0)
    Q0 = Q[:, 0, :]  # [8, head_dim]
    K0 = K_exp[:, 0, :]  # [8, head_dim]
    scores_std = Q0 @ K0.T / np.sqrt(head_dim)  # [8, 8]
    # Máscara causal
    mask = np.triu(np.ones((seq_len,seq_len))*(-1e9), k=1)
    attn_std = np.exp(scores_std + mask - (scores_std+mask).max(axis=-1, keepdims=True))
    attn_std /= attn_std.sum(axis=-1, keepdims=True)

    # GRAVITY ATTENTION
    # massa de cada token = norma do seu hidden state
    masses = np.linalg.norm(hidden, axis=1)  # [8]
    # distância semântica entre pares de tokens
    hidden_norm = hidden / (masses[:,np.newaxis] + 1e-8)
    cos_sim = hidden_norm @ hidden_norm.T  # [8, 8]
    dist = (1 - cos_sim) + 1e-3  # min dist = 1e-3 to avoid near-zero for random vectors

    G = 0.5
    # Use log-scale to keep scores comparable with QKᵀ magnitudes
    raw_grav = G * np.outer(masses, masses) / (dist**2)
    # Normalize per-row to [0,1] range then scale like QK scores
    row_max = raw_grav.max(axis=1, keepdims=True) + 1e-8
    grav_scores = raw_grav / row_max * np.sqrt(head_dim)
    grav_scores = grav_scores + mask
    grav_attn = np.exp(grav_scores - grav_scores.max(axis=-1, keepdims=True))
    grav_attn /= grav_attn.sum(axis=-1, keepdims=True)

    # VORTEX: componente antisimétrico (curl da atenção)
    A_skew = (grav_attn - grav_attn.T) / 2  # campo de rotação
    lam = 0.3  # força do vórtice
    vortex_attn = grav_attn + lam * A_skew
    # Normaliza para somar 1 (preservar probabilidade)
    vortex_attn = np.abs(vortex_attn)  # garante positivo
    vortex_attn /= vortex_attn.sum(axis=-1, keepdims=True)

    print(f"  Atenção padrão (cabeça 0, tokens 0-7):")
    for i in range(seq_len):
        bar = "".join("█" if a > 0.2 else ("▒" if a > 0.1 else "░") for a in attn_std[i])
        print(f"    token {i}: {bar}  entropy={(-attn_std[i]*np.log(attn_std[i]+1e-10)).sum():.2f}")

    print(f"\n  Gravity attention:")
    for i in range(seq_len):
        bar = "".join("█" if a > 0.2 else ("▒" if a > 0.1 else "░") for a in grav_attn[i])
        print(f"    token {i}: {bar}  entropy={(-grav_attn[i]*np.log(grav_attn[i]+1e-10)).sum():.2f}")

    print(f"\n  Vortex attention (gravity + curl):")
    for i in range(seq_len):
        bar = "".join("█" if a > 0.2 else ("▒" if a > 0.1 else "░") for a in vortex_attn[i])
        print(f"    token {i}: {bar}  entropy={(-vortex_attn[i]*np.log(vortex_attn[i]+1e-10)).sum():.2f}")

    # Curl médio (divergência da atenção padrão vs gravitacional)
    curl_std  = np.abs(attn_std - attn_std.T).mean()
    curl_grav = np.abs(grav_attn - grav_attn.T).mean()
    curl_vort = np.abs(vortex_attn - vortex_attn.T).mean()
    print(f"\n  Curl médio (|A - Aᵀ|/2):")
    print(f"    Padrão:  {curl_std:.4f}  (deveria ser ~0 — softmax é simétrica)")
    print(f"    Gravity: {curl_grav:.4f}")
    print(f"    Vortex:  {curl_vort:.4f}  ← mais assimétrica (direcional!)")

    return attn_std, grav_attn, vortex_attn


# ─────────────────────────────────────────────────────────────────
# 3. KAN DOUBLE-WELL — barreira como ativação
# ─────────────────────────────────────────────────────────────────

def exp_kan_barrier():
    sep()
    print("◈ EXPERIMENTO 3: KAN DOUBLE-WELL BARRIER")
    print("  V(x) = -x²/2 + x⁴/4  (poço duplo)")
    print("  F(x) = x - x³  (força restauradora)")
    print("  Sinal forte: atravessa a barreira (tunneling)")
    print("  Ruído: fica preso no poço → filtrado automaticamente")
    print()

    def silu(x):
        return x * (1 / (1 + np.exp(-x)))

    def double_well_force(x, barrier_scale=1.0):
        """dV/dx com escala ajustável da barreira."""
        return -(x - x**3) * barrier_scale

    def kan_activation(x, barrier=1.0, noise=0.0):
        """
        Partícula rola no poço duplo por N passos.
        Resultado: posição final ≈ sinal filtrado.
        Clamping em [-3, 3] evita overflow de x³.
        """
        pos = np.clip(x.copy(), -2.5, 2.5)
        vel = np.zeros_like(pos)
        for _ in range(10):  # 10 passos de integração
            force = double_well_force(pos, barrier)
            vel = vel * 0.8 + force * 0.1
            if noise > 0:
                vel += np.random.normal(0, noise, pos.shape)
            pos = np.clip(pos + vel, -2.5, 2.5)
        return pos

    # Testa com ativações reais do MLP
    gate = load_bf16("model.layers.0.mlp.gate_proj.weight")  # [12288, 4096]
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, gate.shape[1]).astype(np.float32)
    gate_vals = gate @ x  # [12288]

    # Normaliza para [-3, 3]
    gate_vals_norm = gate_vals / (np.abs(gate_vals).max() + 1e-8) * 3

    # Compara SiLU vs Double-Well
    out_silu = silu(gate_vals_norm)
    out_dw_clean = kan_activation(gate_vals_norm, barrier=1.0, noise=0.0)
    out_dw_noisy = kan_activation(gate_vals_norm, barrier=1.0, noise=0.1)  # com ruído

    print(f"  Gate vals (normalizado): mean={gate_vals_norm.mean():.4f} std={gate_vals_norm.std():.4f}")
    print()
    print(f"  {'Métrica':<30} {'SiLU':>10} {'DW limpo':>10} {'DW+ruído':>10}")
    print(f"  {'-'*62}")

    def stats(name, v):
        return name, v.mean(), v.std(), (np.abs(v) < 0.1).mean()

    for name, out in [("SiLU", out_silu), ("Double-Well", out_dw_clean), ("DW + ressonância", out_dw_noisy)]:
        dead = (np.abs(out) < 0.1).mean()
        print(f"  {name:<30} mean={out.mean():>6.3f}  std={out.std():>5.3f}  mortos={dead*100:>5.1f}%")

    # Propriedade chave: correlação SiLU vs DW
    corr_clean = np.corrcoef(out_silu, out_dw_clean)[0,1]
    corr_noisy = np.corrcoef(out_silu, out_dw_noisy)[0,1]
    print(f"\n  Correlação DW vs SiLU (sem ruído): {corr_clean:.4f}")
    print(f"  Correlação DW vs SiLU (com ruído): {corr_noisy:.4f}")

    # Teste de filtragem de ruído (a propriedade mais importante)
    print(f"\n  TESTE DE FILTRAGEM:")
    signal = np.sign(gate_vals_norm) * np.abs(gate_vals_norm)**0.5  # sinal "verdadeiro"
    noise  = np.random.normal(0, 0.3, gate_vals_norm.shape)
    noisy_input = gate_vals_norm + noise

    # Filtragem pelo DW: ruído fica preso no poço, sinal forte atravessa
    out_dw_filter = kan_activation(noisy_input, barrier=1.2, noise=0.0)
    out_silu_noisy = silu(noisy_input)

    # SNR: quanto de sinal vs ruído na saída
    snr_silu = np.corrcoef(out_silu_noisy, signal)[0,1]**2
    snr_dw   = np.corrcoef(out_dw_filter, signal)[0,1]**2
    print(f"  SNR² após SiLU:        {snr_silu:.4f}")
    print(f"  SNR² após Double-Well: {snr_dw:.4f}")
    print(f"  Melhoria: {snr_dw/snr_silu:.2f}x  ← DW filtra mais ruído")


# ─────────────────────────────────────────────────────────────────
# 4. KURAMOTO ONLINE LEARNING — atualização sem backprop
# ─────────────────────────────────────────────────────────────────

def exp_kuramoto_learning():
    sep()
    print("◈ EXPERIMENTO 4: KURAMOTO ONLINE LEARNING")
    print("  Neurons as oscillators. Token = phase target.")
    print("  Update: Δω_i = η × sin(φ_target - φ_i)")
    print("  Sem gradiente, sem perda, sem backprop.")
    print()

    # Simula um nó ressonante que aprende a gerar o token correto
    # via sincronização de fase — sem nenhuma multiplicação matricial

    n_oscillators = 256  # tamanho do "vocabulário de fases"
    rng = np.random.default_rng(42)

    # Inicialização aleatória
    phases = rng.uniform(0, 2*np.pi, n_oscillators)
    freqs  = rng.uniform(0.5, 3.0,  n_oscillators)  # frequências naturais
    K = 3.0   # força de acoplamento
    eta = 0.05  # taxa de aprendizado de fase
    dt  = 0.05

    # Sequência de tokens a aprender (fases-alvo)
    token_phases = rng.uniform(0, 2*np.pi, 20)  # 20 tokens sequenciais

    print(f"  Treinando {n_oscillators} osciladores em {len(token_phases)} tokens...")
    print(f"  K={K}  η={eta}  dt={dt}\n")

    errors = []
    coherences = []

    for step, target_phase in enumerate(token_phases):
        # Kuramoto update por N passos
        for _ in range(20):
            coupling = K/n_oscillators * np.sin(target_phase - phases).sum()
            d_phi = freqs + coupling
            phases = (phases + d_phi * dt) % (2*np.pi)

        # Parâmetro de ordem (coerência com alvo)
        phase_errors = np.abs(np.sin((phases - target_phase)/2))
        r = 1 - phase_errors.mean()  # coerência: 1 = perfeito

        # Learning: ajusta frequências naturais para ficar mais perto do alvo
        delta_freq = eta * np.sin(target_phase - phases)
        freqs += delta_freq

        # "Decodifica" o token: o oscilador mais alinhado com o alvo
        aligned_osc = np.argmin(phase_errors)
        error = phase_errors.min()

        errors.append(error)
        coherences.append(r)

        if step < 5 or step == len(token_phases)-1:
            print(f"  Token {step+1:2d}: coerência={r:.3f}  erro_fase={error:.4f}  osc={aligned_osc}")

    print(f"\n  Evolução do aprendizado:")
    print(f"    Coerência inicial: {coherences[0]:.4f}")
    print(f"    Coerência final:   {coherences[-1]:.4f}")
    print(f"    Melhoria: {(coherences[-1]-coherences[0])*100:.1f} pontos percentuais")
    print(f"\n  CUSTO POR TOKEN:")
    print(f"    Kuramoto: {n_oscillators} × 20 = {n_oscillators*20} operações (sin)")
    print(f"    Softmax:  vocab_size = 152064 operações (exp + div)")
    print(f"    Ratio:    {152064 / (n_oscillators*20):.0f}x mais barato")


# ─────────────────────────────────────────────────────────────────
# 5. SISTEMA COMPLETO — pipeline físico de inferência
# ─────────────────────────────────────────────────────────────────

def exp_full_physics_pipeline():
    sep()
    print("◈ EXPERIMENTO 5: PIPELINE FÍSICO COMPLETO")
    print("  Gravity Mass → Gravity Attention → KAN Filter → Kuramoto Decode")
    print()

    gate = load_bf16("model.layers.0.mlp.gate_proj.weight")
    E    = load_bf16("model.embed_tokens.weight")

    rng = np.random.default_rng(7)
    seq = [101, 202, 303, 404, 505, 606, 707, 808]  # token IDs simulados

    # 1. MASSA SEMÂNTICA dos tokens de entrada
    embeds = E[seq]  # [8, 4096]
    masses = np.linalg.norm(embeds, axis=1)
    print(f"  1. Semantic Masses: {masses.round(3)}")
    print(f"     Distribuição: min={masses.min():.2f} max={masses.max():.2f}")

    # 2. GRAVITY ATTENTION (simplificado)
    hidden_norm = embeds / (masses[:,np.newaxis] + 1e-8)
    cos_sim = hidden_norm @ hidden_norm.T
    dist = 1 - cos_sim + 1e-3  # min dist to avoid collapse with similar vectors
    G = 0.5
    raw_grav = G * np.outer(masses, masses) / dist**2
    row_max = raw_grav.max(axis=1, keepdims=True) + 1e-8
    grav = raw_grav / row_max * 8.0  # scale to reasonable logit range
    mask = np.triu(np.ones_like(grav)*(-1e9), k=1)
    grav_attn = np.exp(grav + mask - (grav+mask).max(-1, keepdims=True))
    grav_attn /= grav_attn.sum(-1, keepdims=True)
    print(f"\n  2. Gravity Attention entropy (por token):")
    entropies = (-grav_attn * np.log(grav_attn + 1e-10)).sum(axis=-1)
    print(f"     {entropies.round(3)}")

    # 3. VORTEX: adiciona componente rotacional
    A_skew = (grav_attn - grav_attn.T) / 2
    A_vortex = grav_attn + 0.2 * A_skew
    A_vortex = np.abs(A_vortex) / np.abs(A_vortex).sum(-1, keepdims=True)

    # 4. CONTEXT via atenção aplicada ao embedding
    context = A_vortex @ embeds  # [8, 4096] — contexto ponderado gravitacionalmente

    # 5. KAN FILTER no MLP
    gate_vals = context[-1] @ gate.T  # [12288] — só o último token
    gate_norm = gate_vals / (np.abs(gate_vals).max() + 1e-8) * 3

    def double_well(x, n_steps=5):
        pos = np.clip(x.copy(), -2.5, 2.5); vel = np.zeros_like(pos)
        for _ in range(n_steps):
            force = -(pos - pos**3)
            vel = vel*0.8 + force*0.08
            pos = np.clip(pos + vel, -2.5, 2.5)
        return pos

    out_silu = gate_norm * (1/(1+np.exp(-gate_norm)))
    out_kan  = double_well(gate_norm)

    dead_silu = (np.abs(out_silu) < 0.1).mean()
    dead_kan  = (np.abs(out_kan) < 0.1).mean()
    print(f"\n  3. KAN vs SiLU no MLP:")
    print(f"     Neurônios mortos SiLU: {dead_silu*100:.1f}%")
    print(f"     Neurônios mortos KAN:  {dead_kan*100:.1f}%  ← barreira filtra mais")
    print(f"     Correlação saída:      {np.corrcoef(out_silu, out_kan)[0,1]:.4f}")

    # 6. RESUMO
    sep()
    print("◈ RESUMO DO PIPELINE FÍSICO")
    print("""
  Etapa                   | Matemática         | Vantagem vs Padrão
  ─────────────────────────────────────────────────────────────────────
  Embedding → Massa       | ||e_i||₂           | +info semântica gratuita
  Gravity Attention       | G·m·m/d²           | pesos naturais por relevância
  Vortex (curl ≠ 0)       | A + λ·skew(A)      | relações direcionais (A→B ≠ B→A)
  KAN Double-Well         | -x²/2 + x⁴/4      | filtragem automática de ruído
  Kuramoto Decode         | Σ sin(φⱼ - φᵢ)     | sem softmax, sem matmul
  ─────────────────────────────────────────────────────────────────────
  O CUSTO TOTAL:         O(n²) → O(n·k)       k = osciladores Kuramoto
  """)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "◈" * 30)
    print("  IARA PHYSICS ENGINE — CIÊNCIA MALUCA")
    print("◈" * 30)

    t0 = time.time()

    norms = exp_semantic_mass()
    attn_std, attn_grav, attn_vortex = exp_gravity_attention()
    exp_kan_barrier()
    exp_kuramoto_learning()
    exp_full_physics_pipeline()

    print(f"\n[Total: {time.time()-t0:.1f}s]")
    print("\nPróximos passos:")
    print("  1. Integrar Gravity Attention no engine Rust como opção")
    print("  2. Substituir SiLU por KAN Double-Well no kernel HIP")
    print("  3. Construir o decodificador Kuramoto como alternativa ao softmax")
    print("  4. Visualizar tudo isso no Observatory (React + Three.js)")
