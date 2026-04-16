from rich import print


def cprint(message: str, color: str = 'white') -> None:
    """Simple colored print function"""
    print(f'[{color}]{message}[/{color}]')
