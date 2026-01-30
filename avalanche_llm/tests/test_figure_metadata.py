from __future__ import annotations

from pathlib import Path


def test_save_figure_embeds_provenance_in_png_and_pdf(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from avalanche_llm.plotting.savefig import save_figure

    prov = {
        "run_id": "RUN_TEST",
        "model_id": "MODEL_TEST",
        "dataset_role": "A",
        "spike_def_id": "SPIKE_ONE_SIDED_POS",
        "target_rate": "0.01",
        "g": "1.0",
        "config_hash": "deadbeef",
    }

    out_pdf = tmp_path / "fig.pdf"
    out_png = tmp_path / "fig.png"
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    save_figure(fig=fig, out_pdf=out_pdf, out_png=out_png, provenance=prov, dpi=72)
    plt.close(fig)

    img = Image.open(out_png)
    try:
        for k, v in prov.items():
            assert img.info.get(k) == str(v)
    finally:
        img.close()

    b = out_pdf.read_bytes()
    # We store provenance JSON in the /Keywords field; verify it is present.
    assert b"/Keywords" in b
    assert b"RUN_TEST" in b

