"""
Batch processing script for transcript analysis.

This script processes multiple transcript files through the thematic analysis pipeline,
saving results to designated output folders without manual intervention.

Usage:
    python batch_process.py --source "1 on 1 Transcripts" --output "1 on 1 Transcripts - first pass"
    python batch_process.py --source "Post survey Transcripts" --output "Post survey Transcripts - first pass" --resume
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from api_utils import analyze_transcript, format_results_as_text
from file_utils import load_api_params, parse_file_from_path


@dataclass
class FileProcessingResult:
    """Result of processing a single transcript file."""
    filename: str
    source_path: Path
    output_path: Optional[Path]
    status: str  # 'success' or 'failed'
    error_message: Optional[str]
    processing_time: float
    theme_count: Optional[int] = None
    quote_count: Optional[int] = None


def discover_transcript_files(source_folder: Path) -> List[Path]:
    """Discover all transcript files in the source folder.

    Args:
        source_folder: Path to folder containing transcript files

    Returns:
        List of Path objects for transcript files, sorted alphabetically
    """
    extensions = ['.txt', '.docx']
    files = []

    for ext in extensions:
        files.extend(source_folder.glob(f'*{ext}'))

    # Sort alphabetically for consistent ordering
    files.sort()
    return files


def check_already_processed(source_file: Path, output_folder: Path) -> bool:
    """Check if a file has already been processed.

    Args:
        source_file: Source transcript file
        output_folder: Output folder to check

    Returns:
        True if output files already exist, False otherwise
    """
    base_name = source_file.stem  # filename without extension
    json_output = output_folder / f"{base_name}_results.json"
    txt_output = output_folder / f"{base_name}_results.txt"

    return json_output.exists() and txt_output.exists()


def save_batch_results(results: dict, output_folder: Path, base_filename: str):
    """Save analysis results in both JSON and text formats.

    Args:
        results: Analysis results dictionary
        output_folder: Folder to save results to
        base_filename: Base filename (without extension)

    Returns:
        Tuple of (json_path, text_path)
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_folder / f"{base_filename}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save text
    text_content = format_results_as_text(results)
    txt_path = output_folder / f"{base_filename}_results.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text_content)

    return json_path, txt_path


def count_themes_and_quotes(results: dict) -> tuple[int, int]:
    """Count total themes and quotes in analysis results.

    Args:
        results: Analysis results dictionary

    Returns:
        Tuple of (theme_count, quote_count)
    """
    themes = results.get('themes', [])
    theme_count = len(themes)
    quote_count = 0

    for theme in themes:
        for subtheme in theme.get('subthemes', []):
            quote_count += len(subtheme.get('supporting_quotes', []))

    return theme_count, quote_count


def process_single_file(
    file_path: Path,
    output_folder: Path,
    client: OpenAI,
    model: str,
    verbose: bool = False
) -> FileProcessingResult:
    """Process a single transcript file.

    Args:
        file_path: Path to transcript file
        output_folder: Folder to save results to
        client: OpenAI client instance
        model: Model name to use
        verbose: Show detailed processing logs

    Returns:
        FileProcessingResult with processing outcome
    """
    start_time = time.time()
    filename = file_path.name
    base_filename = file_path.stem

    try:
        # Parse file content
        transcript_text = parse_file_from_path(file_path)

        # Run analysis
        if verbose:
            print(f"[ANALYSIS] Analyzing {filename}...")

        results = analyze_transcript(client, model, transcript_text)

        # Save results
        json_path, txt_path = save_batch_results(results, output_folder, base_filename)

        # Count statistics
        theme_count, quote_count = count_themes_and_quotes(results)

        processing_time = time.time() - start_time

        return FileProcessingResult(
            filename=filename,
            source_path=file_path,
            output_path=json_path,
            status='success',
            error_message=None,
            processing_time=processing_time,
            theme_count=theme_count,
            quote_count=quote_count
        )

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)

        return FileProcessingResult(
            filename=filename,
            source_path=file_path,
            output_path=None,
            status='failed',
            error_message=error_msg,
            processing_time=processing_time
        )


def print_progress(current: int, total: int, filename: str):
    """Print progress indicator.

    Args:
        current: Current file number (1-indexed)
        total: Total number of files
        filename: Name of current file
    """
    print(f"\n[{current}/{total}] Processing: {filename}")


def print_summary_report(results: List[FileProcessingResult], total_time: float):
    """Print summary report of batch processing.

    Args:
        results: List of processing results
        total_time: Total processing time in seconds
    """
    total_files = len(results)
    successful = [r for r in results if r.status == 'success']
    failed = [r for r in results if r.status == 'failed']

    success_count = len(successful)
    failed_count = len(failed)
    success_rate = (success_count / total_files * 100) if total_files > 0 else 0

    # Calculate average time per file
    avg_time = total_time / total_files if total_files > 0 else 0

    # Format total time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{seconds}s"

    print("\n" + "=" * 50)
    print("Processing Complete")
    print("=" * 50)
    print(f"Total files:     {total_files}")
    print(f"Successful:      {success_count} ({success_rate:.1f}%)")
    print(f"Failed:          {failed_count} ({100-success_rate:.1f}%)")
    print(f"Total time:      {time_str}")
    print(f"Avg per file:    {avg_time:.1f}s")

    # Print statistics for successful files
    if successful:
        total_themes = sum(r.theme_count for r in successful if r.theme_count)
        total_quotes = sum(r.quote_count for r in successful if r.quote_count)
        print(f"\nTotal themes extracted:  {total_themes}")
        print(f"Total quotes extracted:  {total_quotes}")

    # Print failed files if any
    if failed:
        print("\nFailed Files:")
        print("-" * 50)
        for result in failed:
            print(f"  - {result.filename}")
            print(f"    Error: {result.error_message}")


def process_batch(
    source_folder: Path,
    output_folder: Path,
    client: OpenAI,
    model: str,
    resume: bool = False,
    verbose: bool = False
) -> List[FileProcessingResult]:
    """Process a batch of transcript files.

    Args:
        source_folder: Folder containing transcript files
        output_folder: Folder to save results to
        client: OpenAI client instance
        model: Model name to use
        resume: Skip already processed files
        verbose: Show detailed processing logs

    Returns:
        List of FileProcessingResult objects
    """
    # Discover files
    files = discover_transcript_files(source_folder)

    if not files:
        print(f"No transcript files found in {source_folder}")
        return []

    print("\nBatch Processing Started")
    print("=" * 50)
    print(f"Source: {source_folder}")
    print(f"Output: {output_folder}")
    print(f"Found {len(files)} transcript files")

    if resume:
        print("Resume mode: Skipping already processed files")

    results = []

    for idx, file_path in enumerate(files, start=1):
        print_progress(idx, len(files), file_path.name)

        # Check if already processed (resume mode)
        if resume and check_already_processed(file_path, output_folder):
            print("⏭️  Skipped (already processed)")
            continue

        # Process file
        result = process_single_file(file_path, output_folder, client, model, verbose)
        results.append(result)

        # Print result
        if result.status == 'success':
            print(f"✓ Success ({result.processing_time:.1f}s) - {result.theme_count} themes, {result.quote_count} quotes")
        else:
            print(f"✗ Failed ({result.processing_time:.1f}s) - {result.error_message}")

    return results


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description='Batch process transcript files through thematic analysis pipeline'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source folder containing transcript files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output folder for analysis results'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip files that have already been processed'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed processing logs'
    )

    args = parser.parse_args()

    # Convert to Path objects
    source_folder = Path(args.source)
    output_folder = Path(args.output)

    # Validate source folder exists
    if not source_folder.exists():
        print(f"Error: Source folder does not exist: {source_folder}")
        sys.exit(1)

    if not source_folder.is_dir():
        print(f"Error: Source path is not a directory: {source_folder}")
        sys.exit(1)

    # Load API configuration
    try:
        api_params = load_api_params()
    except Exception as e:
        print(f"Error loading API configuration: {e}")
        print("Make sure .secrets.toml exists with API_KEY, API_URL, and MODEL")
        sys.exit(1)

    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_params['API_KEY'],
        base_url=api_params['API_URL']
    )

    # Process batch
    start_time = time.time()
    results = process_batch(
        source_folder=source_folder,
        output_folder=output_folder,
        client=client,
        model=api_params['MODEL'],
        resume=args.resume,
        verbose=args.verbose
    )
    total_time = time.time() - start_time

    # Print summary
    if results:
        print_summary_report(results, total_time)

        # Exit with error code if any files failed
        failed_count = sum(1 for r in results if r.status == 'failed')
        if failed_count > 0:
            sys.exit(1)
    else:
        print("\nNo files were processed.")


if __name__ == '__main__':
    main()
