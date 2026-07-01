import concurrent.futures
import sys
import io
import traceback
import logging


def get_pool_results(futures):
    """Get results from futures, capturing stdout only if logging is DEBUG."""
    results = {}
    capture_output = logging.getLogger().isEnabledFor(logging.DEBUG)

    for future in concurrent.futures.as_completed(futures):
        if capture_output:
            print_buffer = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = print_buffer
        else:
            print_buffer = None

        try:
            result = future.result()
        except Exception:
            if capture_output:
                sys.stdout = original_stdout
            traceback.print_exc()
            sys.exit(1)
        finally:
            if capture_output:
                sys.stdout = original_stdout

        if capture_output:
            output = print_buffer.getvalue().strip()
            if output:
                print(output)

        key = futures[future]
        results[key] = result

    return results


def run_parallel(*, func, key, items, **kwargs):
    """
    Run func(**{key: item}, **kwargs) for each item in items using ThreadPoolExecutor.
    
    Parameters
    ----------
    func : callable
        Function to run in parallel. Must accept keyword arguments.
    key : str
        The keyword name to use when passing each item (default "item").
    items : iterable
        Values to pass in under the keyword name `key`.
    kwargs : dict
        Additional keyword arguments forwarded to func.
    
    Returns
    -------
    dict
        Mapping from item to result (or exception).
    """    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(func, **{key: item}, **kwargs): item
            for item in items
        }
        return get_pool_results(futures)