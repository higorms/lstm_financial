"""
Visualização 3D interativa de TODAS as arquiteturas LSTM do projeto.
Gera um HTML interativo por modelo + um índice geral.

Modelos: A (Baseline), B1 (Attention), B2 (Conv1D+LSTM),
         B3 (Bidirectional), B4 (LSTM+GRU), D (Final = B4)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

# ══════════════════════════════════════════════════════════════════════
#  Registro de modelos
# ══════════════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join("..", "models")
REPORTS_DIR = os.path.join("..", "reports")

MODELS = {
    "A_baseline": {
        "file": "etapa_A_baseline_best.keras",
        "title": "Etapa A — Baseline LSTM",
        "desc": "LSTM(128) → BN → Drop → LSTM(64) → BN → Drop → Dense(32, ReLU) → Sigmoid",
    },
    "B1_attention": {
        "file": "etapa_B1_attention_best.keras",
        "title": "Etapa B1 — Attention LSTM",
        "desc": "LSTM → BN → LSTM(ret_seq) → BN → MultiHeadAttention → GlobalAvgPool → Dense → Sigmoid",
    },
    "B2_conv1d_lstm": {
        "file": "etapa_B2_conv1d_lstm_best.keras",
        "title": "Etapa B2 — Conv1D + LSTM",
        "desc": "Conv1D → MaxPool → BN → LSTM → BN → LSTM → BN → Dense → Sigmoid",
    },
    "B3_bidirectional": {
        "file": "etapa_B3_bidirectional_best.keras",
        "title": "Etapa B3 — Bidirectional LSTM",
        "desc": "BiLSTM → BN → Drop → BiLSTM → BN → Drop → Dense(ReLU) → Sigmoid",
    },
    "B4_lstm_gru": {
        "file": "etapa_B4_lstm_gru_best.keras",
        "title": "Etapa B4 — LSTM + GRU Híbrido",
        "desc": "LSTM(128) → BN → Drop → GRU(64) → BN → Drop → Dense(32, ReLU) → Sigmoid",
    },
    "D_modelo_final": {
        "file": "etapa_D_modelo_final_best.keras",
        "title": "Etapa D — Modelo Final (LSTM+GRU retreinado)",
        "desc": "Modelo B4 retreinado como modelo final selecionado",
    },
}

# ══════════════════════════════════════════════════════════════════════
#  Cores por tipo de camada
# ══════════════════════════════════════════════════════════════════════
COLOR_MAP = {
    "InputLayer": "#4FC3F7",
    "LSTM": "#FF7043",
    "GRU": "#E040FB",
    "Bidirectional": "#AB47BC",
    "Conv1D": "#26C6DA",
    "MaxPooling1D": "#80DEEA",
    "BatchNormalization": "#AED581",
    "LayerNormalization": "#C5E1A5",
    "Dropout": "#FFD54F",
    "Dense": "#42A5F5",
    "MultiHeadAttention": "#FF8A65",
    "Add": "#A1887F",
    "GlobalAveragePooling1D": "#CE93D8",
    "output": "#EF5350",
}

TYPE_MAP = {
    "InputLayer": "input",
    "LSTM": "recurrent",
    "GRU": "recurrent",
    "Bidirectional": "recurrent",
    "Conv1D": "conv",
    "MaxPooling1D": "pool",
    "BatchNormalization": "norm",
    "LayerNormalization": "norm",
    "Dropout": "reg",
    "Dense": "dense",
    "MultiHeadAttention": "attention",
    "Add": "merge",
    "GlobalAveragePooling1D": "pool",
}


# ══════════════════════════════════════════════════════════════════════
#  Funções auxiliares
# ══════════════════════════════════════════════════════════════════════
def extract_layer_info(model):
    """Extrai metadados detalhados de cada camada do modelo Keras."""
    layers = []
    for layer in model.layers:
        cfg = layer.get_config()
        class_name = layer.__class__.__name__        # output_shape: TF 2.21 InputLayer não tem .output_shape
        try:
            out_shape = layer.output_shape
        except AttributeError:
            # Fallback: config batch_shape ou output.shape
            out_shape = cfg.get("batch_shape") or getattr(
                getattr(layer, "output", None), "shape", None
            )
        # Converter TensorShape para tupla
        if out_shape is not None and not isinstance(out_shape, (tuple, list)):
            out_shape = tuple(out_shape)

        info = {
            "name": layer.name,
            "class": class_name,
            "output_shape": out_shape,
            "params": layer.count_params(),
            "trainable": sum(
                tf.keras.backend.count_params(w) for w in layer.trainable_weights
            ),
            "non_trainable": sum(
                tf.keras.backend.count_params(w) for w in layer.non_trainable_weights
            ),
            "config": cfg,
        }

        # Extrair unidades e ativação
        if "cell" in cfg:  # LSTM / GRU
            cell = cfg["cell"]["config"]
            info["units"] = cell.get("units")
            info["activation"] = cell.get("activation", "?")
            info["recurrent_activation"] = cell.get("recurrent_activation", "")
            kr = cell.get("kernel_regularizer")
            if kr:
                info["l2_kernel"] = kr.get("config", {}).get("l2")
            rr = cell.get("recurrent_regularizer")
            if rr:
                info["l2_recurrent"] = rr.get("config", {}).get("l2")
        elif class_name == "Bidirectional":
            fwd = cfg.get("layer", {}).get("config", {})
            cell = fwd.get("cell", fwd)
            if isinstance(cell, dict) and "config" in cell:
                cell = cell["config"]
            info["units"] = cell.get("units")
            info["activation"] = cell.get("activation", "?")
            info["recurrent_activation"] = cell.get("recurrent_activation", "")
            info["note"] = "bidirectional -> output = 2 x units"
        else:
            info["units"] = cfg.get("units") or cfg.get("filters")
            info["activation"] = cfg.get("activation", "-")

        if class_name == "Dropout":
            info["dropout_rate"] = cfg.get("rate")

        if class_name == "Dense":
            kr = cfg.get("kernel_regularizer")
            if kr:
                info["l2_kernel"] = kr.get("config", {}).get("l2")

        if class_name == "Conv1D":
            info["kernel_size"] = cfg.get("kernel_size")
            info["padding"] = cfg.get("padding")
            kr = cfg.get("kernel_regularizer")
            if kr:
                info["l2_kernel"] = kr.get("config", {}).get("l2")

        if class_name == "MultiHeadAttention":
            info["num_heads"] = cfg.get("num_heads")
            info["key_dim"] = cfg.get("key_dim")

        if class_name == "MaxPooling1D":
            info["pool_size"] = cfg.get("pool_size")

        layers.append(info)
    return layers


def compute_block_size(info, max_params):
    """Retorna (sx, sy, sz) do bloco 3D baseado no tipo e parâmetros."""
    cls = info["class"]
    ltype = TYPE_MAP.get(cls, "other")
    params = info.get("params", 0)
    log_ratio = np.log1p(params) / np.log1p(max(max_params, 1))

    if ltype == "input":
        return 0.35, 1.2, 0.8
    elif ltype == "recurrent":
        sz = 0.4 + 0.8 * log_ratio
        return 0.4, sz * 1.5, sz
    elif ltype == "conv":
        return 0.35, 1.0, 0.6
    elif ltype == "pool":
        return 0.2, 0.7, 0.4
    elif ltype == "attention":
        return 0.4, 1.0, 0.7
    elif ltype == "merge":
        return 0.2, 0.5, 0.35
    elif ltype == "dense":
        sz = 0.3 + 0.5 * log_ratio
        return 0.35, sz * 1.2, sz
    elif ltype == "norm":
        return 0.2, 0.6, 0.4
    elif ltype == "reg":
        return 0.15, 0.5, 0.35
    else:
        return 0.2, 0.5, 0.4


def build_hover(info):
    """Constrói o texto HTML de hover detalhado."""
    cls = info["class"]
    h = f"<b>{info['name']}</b>  ({cls})<br>"

    # Tratamento especial para InputLayer
    if cls == "InputLayer":
        shape = info.get("output_shape")
        if shape and len(shape) == 3:
            h += f"<br><b>Timesteps:</b> {shape[1]}<br>"
            h += f"<b>Features:</b> {shape[2]}<br>"
            h += f"<b>Batch Shape:</b> {shape}<br>"
            h += f"<br>Cada amostra e uma janela de <b>{shape[1]}</b> dias<br>"
            h += f"com <b>{shape[2]}</b> indicadores/features por dia.<br>"
            h += f"<br>Dtype: float32<br>"
            h += f"Parametros: 0 (camada de entrada)<br>"
        elif shape:
            h += f"Shape: {shape}<br>"
        return h

    if info.get("units"):
        h += f"Unidades: {info['units']}<br>"
    if info.get("note"):
        h += f"<i>{info['note']}</i><br>"
    if info.get("output_shape"):
        h += f"Output Shape: {info['output_shape']}<br>"

    act = info.get("activation")
    if act and act not in ("-", "?"):
        h += f"Ativacao: <b>{act}</b><br>"
    ract = info.get("recurrent_activation")
    if ract:
        h += f"Ativacao Recorrente: <b>{ract}</b><br>"

    h += f"<br>Parametros: {info['params']:,}<br>"
    h += f"  Treinaveis: {info['trainable']:,}<br>"
    h += f"  Nao-treinaveis: {info['non_trainable']:,}<br>"

    if info.get("l2_kernel"):
        h += f"<br>L2 kernel: {info['l2_kernel']}<br>"
    if info.get("l2_recurrent"):
        h += f"L2 recurrent: {info['l2_recurrent']}<br>"
    if info.get("dropout_rate") is not None:
        h += f"<br>Dropout Rate: <b>{info['dropout_rate']:.1%}</b><br>"
    if info.get("num_heads"):
        h += f"<br>Attention Heads: {info['num_heads']}, Key Dim: {info['key_dim']}<br>"
    if info.get("kernel_size"):
        h += f"<br>Kernel: {info['kernel_size']}, Padding: {info['padding']}<br>"
    if info.get("pool_size"):
        h += f"<br>Pool Size: {info['pool_size']}<br>"

    return h


def build_label(info):
    """Label curta para exibir acima do bloco."""
    cls = info["class"]
    name = info["name"]
    units = info.get("units")

    if cls == "InputLayer":
        shape = info.get("output_shape")
        if shape and len(shape) == 3:
            return f"Input<br>({shape[1]}x{shape[2]})"
        return "Input"
    elif cls == "Bidirectional":
        return f"BiLSTM<br>({units}x2={units*2})" if units else "BiLSTM"
    elif cls in ("LSTM", "GRU"):
        return f"{cls}<br>({units})" if units else cls
    elif cls == "Dense":
        act = info.get("activation", "")
        if name == "output" or act == "sigmoid":
            return f"Output<br>(1, sigmoid)"
        return f"Dense<br>({units}, {act})" if units else "Dense"
    elif cls == "Conv1D":
        f_ = info.get("units", "?")
        ks = info.get("kernel_size", "?")
        return f"Conv1D<br>({f_}f, k={ks})"
    elif cls == "Dropout":
        rate = info.get("dropout_rate")
        return f"Dropout<br>({rate:.0%})" if rate else "Dropout"
    elif cls == "BatchNormalization":
        return "BatchNorm"
    elif cls == "LayerNormalization":
        return "LayerNorm"
    elif cls == "MultiHeadAttention":
        nh = info.get("num_heads", "?")
        return f"Attention<br>({nh} heads)"
    elif cls == "MaxPooling1D":
        return f"MaxPool<br>(size={info.get('pool_size', '?')})"
    elif cls == "GlobalAveragePooling1D":
        return "GlobalAvgPool"
    elif cls == "Add":
        return "Add (residual)"
    else:
        return name


def make_cube_mesh(cx, cy, cz, sx, sy, sz):
    """Retorna vértices e faces de um cubo centrado em (cx,cy,cz)."""
    vx = [cx - sx, cx + sx, cx + sx, cx - sx, cx - sx, cx + sx, cx + sx, cx - sx]
    vy = [cy - sy, cy - sy, cy + sy, cy + sy, cy - sy, cy - sy, cy + sy, cy + sy]
    vz = [cz - sz, cz - sz, cz - sz, cz - sz, cz + sz, cz + sz, cz + sz, cz + sz]
    faces = [
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ]
    ii = [f[0] for f in faces]
    jj = [f[1] for f in faces]
    kk = [f[2] for f in faces]
    return vx, vy, vz, ii, jj, kk


# ══════════════════════════════════════════════════════════════════════
#  Gerar visualização 3D de um modelo
# ══════════════════════════════════════════════════════════════════════
def generate_3d_viz(model, title, description, output_path):
    """Gera e salva HTML interativo 3D de um modelo Keras."""
    layers_info = extract_layer_info(model)
    n = len(layers_info)
    max_params = max((li["params"] for li in layers_info), default=1)

    fig = go.Figure()
    x_positions = np.linspace(0, max(10, n * 1.1), n)

    for i, info in enumerate(layers_info):
        cx = x_positions[i]
        cls = info["class"]
        color = COLOR_MAP.get(cls, "#78909C")
        if info["name"] == "output" or (
            cls == "Dense" and info.get("activation") == "sigmoid"
        ):
            color = COLOR_MAP["output"]

        sx, sy, sz = compute_block_size(info, max_params)
        vx, vy, vz, ii, jj, kk = make_cube_mesh(cx, 0, 0, sx, sy, sz)
        hover = build_hover(info)
        label = build_label(info)

        fig.add_trace(
            go.Mesh3d(
                x=vx, y=vy, z=vz, i=ii, j=jj, k=kk,
                color=color, opacity=0.7, flatshading=True,
                hovertext=hover, hoverinfo="text",
                name=f"{info['name']} ({cls})", showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[cx], y=[0], z=[sz + 0.18],
                mode="text", text=[label],
                textfont=dict(size=10, color="white"),
                hoverinfo="skip", showlegend=False,
            )
        )

    # Conexões
    for i in range(n - 1):
        x0, x1 = x_positions[i], x_positions[i + 1]
        fig.add_trace(
            go.Scatter3d(
                x=[x0 + 0.35, x1 - 0.35], y=[0, 0], z=[0, 0],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.4)", width=3),
                hoverinfo="skip", showlegend=False,
            )
        )

    total = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_train = total - trainable

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    ts = input_shape[1] if len(input_shape) >= 3 else "?"
    feat = input_shape[2] if len(input_shape) >= 3 else "?"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<sub>{description}<br>"
                f"Total: {total:,} params ({trainable:,} treinaveis, "
                f"{non_train:,} nao-treinaveis) | "
                f"Input: ({ts} timesteps x {feat} features) -> Output: 1 (sigmoid)</sub>"
            ),
            font=dict(size=15, color="white"),
        ),
        scene=dict(
            xaxis=dict(
                title="Fluxo de Dados ->", showgrid=False,
                zeroline=False, showticklabels=False,
                backgroundcolor="rgba(0,0,0,0)",
            ),
            yaxis=dict(
                title="", showgrid=False, zeroline=False,
                showticklabels=False, backgroundcolor="rgba(0,0,0,0)",
            ),
            zaxis=dict(
                title="", showgrid=False, zeroline=False,
                showticklabels=False, backgroundcolor="rgba(0,0,0,0)",
            ),
            bgcolor="rgb(20,20,30)",
            camera=dict(eye=dict(x=1.8, y=1.5, z=0.8)),
        ),
        paper_bgcolor="rgb(20,20,30)",
        plot_bgcolor="rgb(20,20,30)",
        font=dict(color="white"),
        legend=dict(
            title="Camadas (hover para detalhes)",
            bgcolor="rgba(40,40,60,0.8)",
            bordercolor="rgba(255,255,255,0.3)", borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=0, r=0, t=100, b=0),
        width=1400,
        height=800,
    )

    fig.write_html(output_path, include_plotlyjs=True)
    print(f"  -> {output_path}")


# ══════════════════════════════════════════════════════════════════════
#  Gerar HTML índice com links para todos os modelos
# ══════════════════════════════════════════════════════════════════════
def generate_index_html(generated_files, output_path):
    """Cria um HTML indice com links para todos os modelos."""
    cards = ""
    for key, info in generated_files.items():
        fname = os.path.basename(info["html"])
        cards += f"""
        <div class="card">
            <h2>{info['title']}</h2>
            <p class="desc">{info['desc']}</p>
            <p class="stats">{info['stats']}</p>
            <a href="{fname}" target="_blank" class="btn">Abrir Visualizacao 3D Interativa</a>
        </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Arquiteturas LSTM - Visualizacoes 3D</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0; min-height: 100vh; padding: 40px 20px;
        }}
        h1 {{
            text-align: center; font-size: 2rem; margin-bottom: 10px;
            background: linear-gradient(90deg, #4FC3F7, #E040FB);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ text-align: center; color: #aaa; margin-bottom: 40px; font-size: 0.95rem; }}
        .grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 24px; max-width: 1400px; margin: 0 auto;
        }}
        .card {{
            background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px; padding: 28px; transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(79,195,247,0.15);
        }}
        .card h2 {{ font-size: 1.25rem; margin-bottom: 8px; color: #4FC3F7; }}
        .card .desc {{ font-size: 0.85rem; color: #bbb; margin-bottom: 12px; font-family: monospace; }}
        .card .stats {{ font-size: 0.82rem; color: #999; margin-bottom: 16px; }}
        .btn {{
            display: inline-block; padding: 10px 24px; border-radius: 8px;
            background: linear-gradient(90deg, #4FC3F7, #E040FB);
            color: #fff; text-decoration: none; font-weight: 600; font-size: 0.9rem;
            transition: opacity 0.2s;
        }}
        .btn:hover {{ opacity: 0.85; }}
        footer {{ text-align: center; margin-top: 50px; color: #666; font-size: 0.8rem; }}
    </style>
</head>
<body>
    <h1>Arquiteturas LSTM - Projeto PETR4.SA</h1>
    <p class="subtitle">FIAP Pos-Tech - Tech Challenge Fase 4 - Classificacao Binaria de Direcao de Preco</p>
    <div class="grid">
{cards}
    </div>
    <footer>
        Gerado automaticamente - Visualizacoes interativas com Plotly 3D<br>
        Passe o mouse sobre cada bloco para ver detalhes (unidades, ativacao, parametros, regularizacao)
    </footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nIndice salvo em: {os.path.abspath(output_path)}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    generated = {}

    for key, minfo in MODELS.items():
        model_path = os.path.join(MODELS_DIR, minfo["file"])
        if not os.path.exists(model_path):
            print(f"  [SKIP] {model_path} nao encontrado")
            continue

        print(f"\nCarregando {key}: {minfo['file']}")
        model = tf.keras.models.load_model(model_path)
        model.summary()

        total = model.count_params()
        trainable = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        n_layers = len(model.layers)

        html_path = os.path.join(REPORTS_DIR, f"arquitetura_{key}_3d.html")
        generate_3d_viz(model, minfo["title"], minfo["desc"], html_path)

        generated[key] = {
            "html": html_path,
            "title": minfo["title"],
            "desc": minfo["desc"],
            "stats": f"{n_layers} camadas | {total:,} params ({trainable:,} treinaveis)",
        }

        del model
        tf.keras.backend.clear_session()

    # Gerar indice
    if generated:
        index_path = os.path.join(REPORTS_DIR, "index_arquiteturas_3d.html")
        generate_index_html(generated, index_path)
        print(f"\nConcluido! {len(generated)} visualizacoes geradas.")
        print(f"Abra: {os.path.abspath(index_path)}")
