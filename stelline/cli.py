"""Stelline CLI interface."""
import logging
import click
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import StellineConfig
from .discovery import SessionDiscovery, SessionFile
from .context import ContextLoader
from .pipeline import StellinePipeline
from .tracker import SessionTracker

LOG_DIR = Path("~/.config/stelline/logs").expanduser()


def setup_logging(verbose: bool = False):
    """Configure logging to file + stderr."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"stelline-{datetime.now().strftime('%Y-%m-%d')}.log"

    log = logging.getLogger("stelline")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.handlers.clear()

    # File handler (always)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)

    # Stderr handler (harvest only, not noisy for scan/status)
    if verbose:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(sh)

    return log


@click.group()
@click.option('--config', '-c', help='Config file path')
@click.pass_context
def cli(ctx, config: Optional[str]):
    """Stelline — Session Intelligence Tool."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = StellineConfig.load(config) if config else StellineConfig()


@cli.command()
@click.option('--batch', '-b', type=int, help='Process N sessions maximum')
@click.option('--source', '-s', help='Process only specific source')
@click.option('--file', '-f', type=click.Path(exists=True), help='Process specific file')
@click.option('--dry-run', '-n', is_flag=True, help='Preview only, no processing')
@click.option('--backend', type=click.Choice(['sse', 'pi']), default=None,
              help='LLM backend: sse (direct API) or pi (pi -p subprocess)')
@click.pass_context
def harvest(ctx, batch: Optional[int], source: Optional[str], 
           file: Optional[str], dry_run: bool, backend: Optional[str]):
    """Process unprocessed sessions through intelligence pipeline."""
    log = setup_logging(verbose=True)
    config = ctx.obj['config']
    if backend:
        config.backend = backend
    tracker = SessionTracker(config.db_path)
    context_loader = ContextLoader(config)
    pipeline = StellinePipeline(config, tracker, context_loader)

    log.info(f"\U0001f31f Stelline waking up — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Backend: {config.backend} (fallback: pi) | Model: {config.model} | Batch: {batch or config.batch_size}")

    if file:
        session_file = SessionFile.from_path(Path(file), "manual")
        result = pipeline.process_session(session_file, dry_run=dry_run)
        click.echo(f"Processed {session_file.session_id}: {result['status']}")
        return

    # Discover unprocessed sessions
    discovery = SessionDiscovery(config, tracker)
    unprocessed = discovery.discover_unprocessed(source)

    if not unprocessed:
        log.info("No unprocessed sessions found. Going back to sleep.")
        click.echo("No unprocessed sessions found.")
        return

    if batch:
        unprocessed = unprocessed[:batch]
    else:
        unprocessed = unprocessed[:config.batch_size]

    log.info(f"Found {len(unprocessed)} sessions to process")
    click.echo(f"Processing {len(unprocessed)} sessions...")

    run_id = tracker.start_harvest_run({
        "batch_size": len(unprocessed),
        "source_filter": source,
        "dry_run": dry_run
    })

    processed = 0
    failed = 0
    total_memories = 0

    for session_file in unprocessed:
        result = pipeline.process_session(session_file, dry_run=dry_run)

        if result['status'] in ('success', 'dry_run'):
            processed += 1
            total_memories += result.get('memories_extracted', 0)
        elif result['status'] == 'failed':
            failed += 1

        # Rate limiting between sessions
        if not dry_run and processed > 0 and processed % 3 == 0:
            import time
            time.sleep(5)

    tracker.complete_harvest_run(run_id, processed, failed, total_memories)

    action = "Would process" if dry_run else "Processed"
    summary = f"{action} {processed} sessions, {failed} failed, {total_memories} memories extracted"
    log.info(f"\U0001f634 Stelline going back to sleep — {summary}")
    click.echo(f"\n{summary}")


@cli.command()
@click.pass_context
def scan(ctx):
    """Show unprocessed session count by source."""
    config = ctx.obj['config']
    tracker = SessionTracker(config.db_path)
    discovery = SessionDiscovery(config, tracker)
    
    stats = discovery.get_source_stats()
    
    click.echo("Session Status by Source:\n")
    for source, counts in stats.items():
        click.echo(f"{source:15} {counts['unprocessed']:3} unprocessed / "
                  f"{counts['total']:3} total")
    
    total_unprocessed = sum(s['unprocessed'] for s in stats.values())
    click.echo(f"\nTotal unprocessed: {total_unprocessed}")


@cli.command()
@click.pass_context  
def status(ctx):
    """Show overall system status and recent activity."""
    config = ctx.obj['config']
    tracker = SessionTracker(config.db_path)
    
    stats = tracker.get_stats()
    
    click.echo("Stelline Status\n")
    
    # Overall stats
    overall = stats['overall']
    click.echo(f"Sessions processed: {overall['total_processed']}")
    click.echo(f"Memories extracted: {overall['total_memories']}")
    click.echo(f"Failed sessions: {overall['failed_count']}")
    if overall['avg_duration']:
        click.echo(f"Avg processing time: {overall['avg_duration']:.1f}s")
    
    # By source breakdown
    if stats['by_source']:
        click.echo("\nBy Source:")
        for source_stats in stats['by_source']:
            click.echo(f"  {source_stats['source']:15} "
                      f"{source_stats['processed']:3} sessions, "
                      f"{source_stats['memories']:3} memories")


@cli.command()
@click.option('--limit', '-l', default=10, help='Number of runs to show')
@click.pass_context
def history(ctx, limit: int):
    """Show recent harvest run history."""
    config = ctx.obj['config']
    tracker = SessionTracker(config.db_path)
    
    runs = tracker.get_recent_runs(limit)
    
    if not runs:
        click.echo("No harvest runs found.")
        return
    
    click.echo("Recent Harvest Runs:\n")
    click.echo(f"{'Started':<20} {'Status':<10} {'Sessions':<8} {'Memories':<8}")
    click.echo("-" * 50)
    
    for run in runs:
        started = run['started_at'][:19]  # Trim to datetime
        status = run['status']
        sessions = run['sessions_processed'] or 0
        memories = run['total_memories'] or 0
        
        click.echo(f"{started:<20} {status:<10} {sessions:<8} {memories:<8}")


@cli.command()
@click.pass_context
def sources(ctx):
    """Show configured sources and their stats."""
    config = ctx.obj['config']
    
    click.echo("Configured Sources:\n")
    
    for source in config.sources:
        status = "enabled" if source.enabled else "disabled"
        click.echo(f"{source.name:15} {source.pattern:25} {status}")
        if source.memkoshi_storage:
            click.echo(f"                Custom storage: {source.memkoshi_storage}")
    
    click.echo(f"\nDefault storage: {config.memkoshi_storage}")
    click.echo(f"Batch size: {config.batch_size}")
    click.echo(f"Model: {config.model}")


if __name__ == '__main__':
    cli()