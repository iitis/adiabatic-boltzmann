"""
Generate HTML summary report of benchmark results.

Usage:
    python generate_report.py --results-dir results/ --output report.html
"""

import json
from pathlib import Path
import argparse
from datetime import datetime


def generate_html_report(results_dir, output_file):
    """
    Generate an HTML report summarizing all results.
    
    Args:
        results_dir: Results directory path
        output_file: Output HTML file path
    """
    results_dir = Path(results_dir)
    
    # Load data
    with open(results_dir / 'summary.json') as f:
        summary = json.load(f)
    
    with open(results_dir / 'statistics.json') as f:
        stats = json.load(f)
    
    with open(results_dir / 'best_configurations.json') as f:
        best = json.load(f)
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RBM Ising Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #0066cc; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #0066cc; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .success {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>RBM Ising Model Benchmark Report</h1>
    
    <div class="timestamp">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    
    <h2>Executive Summary</h2>
    <div class="metric">
        <p><strong>Total Tests:</strong> {summary['totals']['total_tests']}</p>
        <p><strong>Successful:</strong> <span class="success">{summary['totals']['successful']}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{summary['totals']['failed']}</span></p>
    </div>
    
    <h2>Test Configuration</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Values</th>
        </tr>
        <tr>
            <td>System Sizes</td>
            <td>{', '.join(map(str, summary['test_matrix']['system_sizes']))}</td>
        </tr>
        <tr>
            <td>h Values</td>
            <td>{', '.join(map(str, summary['test_matrix']['h_values']))}</td>
        </tr>
        <tr>
            <td>Architectures</td>
            <td>{', '.join(summary['test_matrix']['architectures'])}</td>
        </tr>
        <tr>
            <td>Runs per Config</td>
            <td>{summary['test_matrix']['runs_per_config']}</td>
        </tr>
    </table>
    
    <h2>Best Configurations</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Configuration</th>
            <th>Final Energy</th>
            <th>Improvement</th>
        </tr>
        <tr>
            <td>1</td>
            <td><strong>{best['overall_best']['config']}</strong></td>
            <td>{best['overall_best']['final_energy']:.6f}</td>
            <td>{best['overall_best']['improvement']:.6f}</td>
        </tr>
    </table>
    
    <h2>Results per Configuration</h2>
    <table>
        <tr>
            <th>Configuration</th>
            <th>N</th>
            <th>h</th>
            <th>Architecture</th>
            <th>Runs</th>
            <th>Final Energy (mean ± std)</th>
            <th>Improvement (mean ± std)</th>
        </tr>
"""
    
    for key in sorted(stats.keys()):
        config = stats[key]
        E_final = config['final_energy']['mean']
        E_final_std = config['final_energy']['std']
        E_imp = config['energy_improvement']['mean']
        E_imp_std = config['energy_improvement']['std']
        
        html += f"""
        <tr>
            <td>{key}</td>
            <td>{config['n_spins']}</td>
            <td>{config['h']:.2f}</td>
            <td>{config['architecture']}</td>
            <td>{config['n_runs']}</td>
            <td>{E_final:.6f} ± {E_final_std:.6f}</td>
            <td>{E_imp:.6f} ± {E_imp_std:.6f}</td>
        </tr>
"""
    
    html += """
    </table>
    
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report")
    parser.add_argument('--results-dir', default='results/',
                       help='Results directory')
    parser.add_argument('--output', default='report.html',
                       help='Output HTML file')
    
    args = parser.parse_args()
    
    generate_html_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
