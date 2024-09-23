default: plot_sub_evals


compare_sub_evals:
    python run_analysis.py compare_sub_evals data/encoding_cfg.json output/test/

plot_sub_evals:
    python run_analysis.py plot_sub_evals data/encoding_cfg.json output/test/


plot_selection:
    python run_analysis.py plot_selection data/encoding_cfg.json output/test/
