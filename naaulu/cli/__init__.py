import importlib
import sys

COMMANDS = {
    "estimate": "naaulu.cli.estimate",
    "combine": "naaulu.cli.combine",
    "verify": "naaulu.cli.verify",
    "plot": "naaulu.cli.plot",
    "clean": "naaulu.cli.clean",
    "export": "naaulu.cli.export",
    "list": "naaulu.cli.list",
    "set": "naaulu.cli.set",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        sys.exit(0)

    command = sys.argv[1]

    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        _print_help()
        sys.exit(1)

    import importlib
    module = importlib.import_module(COMMANDS[command])
    sys.argv[:] = [sys.argv[0]] + sys.argv[2:]
    module.main()


def _print_help():
    print("Usage: naaulu <command> [options]")
    print()
    print("Commands:")
    for name, path in sorted(COMMANDS.items()):
        mod = importlib.import_module(path)
        doc = (mod.__doc__ or "").strip().split("\n")[0]
        print(f"  {name:12s} {doc}")
