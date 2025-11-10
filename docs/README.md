# GHOST Diagrams (Mermaid)

This folder contains Mermaid diagrams to visualize how GHOST works.

- `ghost_system_tree.mmd` — Family tree style view of the system
- `ghost_prediction_flow.mmd` — End-to-end prediction flow, from request to forecast

## How to view in VS Code

- Install the "Markdown Preview Mermaid Support" or a Mermaid preview extension.
- Open the `.mmd` files and use the extension's preview command.
- Or embed in a Markdown file like this:

```mermaid
flowchart LR
  Start --> Stop
```

## How to export to PNG/SVG (optional)

Using Node and mermaid-cli (requires Node >= 18):

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i docs/ghost_system_tree.mmd -o docs/ghost_system_tree.svg
mmdc -i docs/ghost_prediction_flow.mmd -o docs/ghost_prediction_flow.svg
```

If `mmdc` is not found, ensure Node is installed or use VS Code extensions that can export.

## Notes

- Colors are chosen for readability on dark backgrounds.
- Update diagrams if subsystems change (e.g., providers/MCP servers).
