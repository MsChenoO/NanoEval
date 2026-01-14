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

@cli.command()
@click.option('--teacher', required=True, help='Teacher model path (HF/Local)')
@click.option('--student', required=True, help='Student model path (HF/Local)')
@click.option('--output', default='distillation_report.json', help='Output JSON report path')
def compare_distillation(teacher, student, output):
    """Compare Teacher vs. Student safety alignment"""
    click.echo(f"[*] Initializing Distillation Audit...")
    click.echo(f"    Teacher: {teacher}")
    click.echo(f"    Student: {student}")
    
    pipeline = SmallModelEvaluationPipeline()
    results = asyncio.run(pipeline.evaluate_model_pair(teacher, student))
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    preservation = results['results'].get('preservation_score', 0)
    click.echo(f"\n[+] Audit Complete. Safety Preservation Score: {preservation:.1%}")
    click.echo(f"    Full report saved to: {output}")

if __name__ == '__main__':
    cli()
