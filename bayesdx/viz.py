import matplotlib.pyplot as plt
import json, os

def reliability_plot(bins_json_path, out_path):
    with open(bins_json_path) as f:
        d = json.load(f)
    bins = d['bins']
    acc = d['acc']
    conf = d['conf']
    plt.figure()
    plt.plot([0,1],[0,1],'--')
    plt.plot(conf, acc, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def risk_coverage_plot(points_json_path, out_path):
    with open(points_json_path) as f:
        d = json.load(f)
    cov = d['coverage']
    risk = d['risk']
    plt.figure()
    plt.plot(cov, risk, marker='o')
    plt.xlabel('Coverage')
    plt.ylabel('Risk (1-Accuracy)')
    plt.title('Risk-Coverage Curve')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
