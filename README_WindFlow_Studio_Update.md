# Portfolio update — add WindFlow Studio v1.6

This update adds the new tool page/folder:

- `windflow-studio/` — modular WindFlow Studio v1.6 application
- `download/index.html` — portfolio tools section updated with a new WindFlow Studio card
- `index.html` — same updated portfolio page at repository root for GitHub Pages root-source deployments

## Live URL after deployment

If deployed to the repository root, the new tool will open at:

```text
https://kvelmurugan77.github.io/portfolio/windflow-studio/
```

The portfolio tools section will show a new card:

```text
WindFlow Studio v1.6
```

## Deploy

```bash
git clone https://github.com/kvelmurugan77/portfolio.git
cd portfolio
cp -r /path/to/portfolio_update_with_windflow_studio/* .
git add index.html download/index.html windflow-studio README_WindFlow_Studio_Update.md
git commit -m "Add WindFlow Studio v1.6 tool to portfolio"
git push origin main
```

Wait 1–3 minutes for GitHub Pages to refresh.
