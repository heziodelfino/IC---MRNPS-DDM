# app.py
import hashlib
import subprocess
from pathlib import Path
import streamlit as st
import os

OUTDIR = Path("renders")
OUTDIR.mkdir(exist_ok=True)

st.title("Pêndulo amortecido -- animações Manim no navegador")

gamma1 = st.slider("gamma_1", 0.0, 1.5, 0.05, 0.01)
gamma2 = st.slider("gamma_2", 0.0, 1.5, 0.20, 0.01)
tcut   = st.slider("t_cut (s)", 0.0, 20.0, 6.60, 0.01)

params = f"g1={gamma1:.4f}_g2={gamma2:.4f}_tc={tcut:.4f}"
vid_name = hashlib.md5(params.encode()).hexdigest() + ".mp4"
vid_path = OUTDIR / vid_name


st.caption(f"Preset: {params}")

if st.button("Renderizar animação (Manim)"):
    if not vid_path.exists():
        with st.spinner("Renderizando..."):

            env = os.environ.copy()
            env["GAMMA1"] = str(gamma1)
            env["GAMMA2"] = str(gamma2)
            env["TCUT"] = str(tcut)
            cmd = [
                    "manim",
                    "-qm",
                    "manim_sub_interesting.py",
                    "PhaseSpacePendulumTwoGammas",
                    "--format", "mp4",
                ]
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error("Erro no Manim:")
                st.code(result.stderr)
                st.stop()

            video_path = Path("media/videos/manim_sub_interesting/1080p60/PhaseSpacePendulumTwoGammas.mp4")

            if video_path.exists():
                st.video(str(video_path))
            else:
                st.error("Vídeo não encontrado")
                st.success("Pronto.")

    if vid_path.exists():
        st.video(str(vid_path))
    else:
        st.warning("Render terminou, mas não encontrei o mp4 no caminho esperado. Ajuste o caminho do output.")