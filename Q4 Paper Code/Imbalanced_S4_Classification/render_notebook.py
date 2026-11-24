"""Export actual saved notebook outputs to standalone HTML; no ML is executed."""
import argparse
import html
import json
from pathlib import Path


def render(notebook_path, destination):
    notebook = json.loads(notebook_path.read_text())
    manifest = notebook['metadata'].get('q4_execution', {})
    chunks = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<title>S4 supervised experiment — executed outputs</title>',
        '<style>body{max-width:1180px;margin:32px auto;padding:0 24px;',
        'font:16px/1.55 system-ui;color:#172333;background:#fff}',
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;',
        'background:#f3f5f8;padding:16px}img{max-width:100%;height:auto}',
        'table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}',
        'td,th{padding:5px;border:1px solid #ccd3dd}section{margin:32px 0}',
        '.notice{padding:18px;background:#fff2d2}.error{color:#a11}',
        'summary{cursor:pointer;font-weight:600}</style>',
        '<h1>S4 supervised experiment: captured notebook outputs</h1>',
        '<p class="notice">Both classes were used for supervised fitting.',
        'All nine OCC predictors include Corrected_S4, which also defines the label.',
        'These are S4-reference agreement results. The final period was excluded',
        'from model fitting/tuning but was evaluated previously; it is not a',
        'genuinely untouched independent test.</p>',
        '<p>Execution status: <strong>' + html.escape(manifest.get('status', 'unknown')) + '</strong></p>',
    ]
    for number, cell in enumerate(notebook['cells'], 1):
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            chunks.append('<section><pre>' + html.escape(source) + '</pre></section>')
            continue
        chunks.append(f'<section><details><summary>Code cell {number}</summary><pre>'
                      + html.escape(source) + '</pre></details>')
        for output in cell.get('outputs', []):
            if output['output_type'] == 'stream':
                text = output.get('text', '')
                chunks.append('<pre>' + html.escape(''.join(text)) + '</pre>')
            elif output['output_type'] == 'error':
                chunks.append('<pre class="error">' + html.escape(
                    '\n'.join(output.get('traceback', []))) + '</pre>')
            else:
                data = output.get('data', {})
                if 'image/png' in data:
                    chunks.append('<img alt="Captured scientific figure" src="data:image/png;base64,'
                                  + ''.join(data['image/png']) + '">')
                elif 'text/html' in data:
                    chunks.append(''.join(data['text/html']))
                elif 'text/plain' in data:
                    chunks.append('<pre>' + html.escape(''.join(data['text/plain'])) + '</pre>')
        chunks.append('</section>')
    chunks.append('</html>')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    destination.write_text('\n'.join(chunks), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('html', type=Path)
    args = parser.parse_args()
    render(args.notebook, args.html)
