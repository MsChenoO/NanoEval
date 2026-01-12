import click
import asyncio
import json
from nanoeval.core.pipeline import SmallModelEvaluationPipeline
from nanoeval.evaluators.standard.refusal_rate import RefusalRateEvaluator

@click.group()
def cli():
    """NanoEval: Safety Certification for Small Models"""
    pass

@cli.command()
@click.option('--model-path', required=True, help='Local path or HF hub ID of the model')
@click.option('--output', default='report.json', help='Output JSON report path')
def evaluate(model_path, output):
    """Run standard safety evaluation on a single model"""
    click.echo(f"[*] Initializing NanoEval Pipeline...")
    
    pipeline = SmallModelEvaluationPipeline()
    
    # Register standard evaluators
    # In a real scenario, this would be driven by config
    refusal_eval = RefusalRateEvaluator(dataset_path="benchmarks/safety_critical_prompts.jsonl")
    pipeline.register_evaluator(refusal_eval)
    
    results = asyncio.run(pipeline.evaluate_model(model_path))
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    click.echo(f"[+] Evaluation complete. Report saved to: {output}")

if __name__ == '__main__':
    cli()
